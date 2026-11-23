"""SQLite owns identities, atomic writes and execution evidence."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..core.types import SessionBusy, StorageError, dumps


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, user TEXT, session TEXT, status TEXT, result TEXT,
                created TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS active_session ON runs(user,session)
                WHERE status='running';
            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY, id TEXT UNIQUE, user TEXT, session TEXT,
                run TEXT, body TEXT);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, run TEXT, kind TEXT, body TEXT, created TEXT);
            CREATE TABLE IF NOT EXISTS attempts (
                call TEXT, attempt INTEGER, run TEXT, user TEXT, session TEXT, name TEXT,
                read_only INTEGER, status TEXT, body TEXT,
                PRIMARY KEY(call,attempt));
            CREATE TABLE IF NOT EXISTS memories (
                user TEXT, key TEXT, body TEXT, PRIMARY KEY(user,key));
            CREATE TABLE IF NOT EXISTS memory_changes (
                seq INTEGER PRIMARY KEY, user TEXT, key TEXT, action TEXT, body TEXT, created TEXT);
            CREATE TABLE IF NOT EXISTS memory_progress (
                user TEXT, session TEXT, last_seq INTEGER, PRIMARY KEY(user,session));
            CREATE TABLE IF NOT EXISTS memory_batches (
                id TEXT PRIMARY KEY, user TEXT, session TEXT, status TEXT, body TEXT, created TEXT);
            -- L3 evidence. An external authority declares when a whole session/trial
            -- stopped; the runtime never decides it and never reads this to steer a run.
            CREATE TABLE IF NOT EXISTS session_terminations (
                seq INTEGER PRIMARY KEY, user TEXT, session TEXT, reason TEXT, source TEXT,
                terminal_run TEXT, run_count INTEGER, created TEXT);
            -- Escalation ledger for unknown-outcome writes: one row per call, written
            -- when the runtime has told the user/environment that the effect is
            -- unknown. ``unresolved()`` only returns rows without an entry here, so
            -- an escalation is delivered once instead of stopping every later turn.
            CREATE TABLE IF NOT EXISTS write_escalations (
                call TEXT PRIMARY KEY, user TEXT, session TEXT, run TEXT, created TEXT);
        """)

    @contextmanager
    def transaction(self):
        try:
            with self.db:
                yield self.db
        except sqlite3.Error as exc:
            raise StorageError(type(exc).__name__) from exc

    def close(self):
        self.db.close()

    def begin(self, user, session, message):
        run = uuid4().hex
        with self.transaction() as db:
            if db.execute(
                "SELECT 1 FROM runs WHERE user=? AND session=? AND status='running'",
                (user, session),
            ).fetchone():
                raise SessionBusy(
                    "Session already running; recover only after its owner has stopped"
                )
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?)",
                (run, user, session, "running", dumps({}), now()),
            )
            self._append(db, user, session, run, {"role": "user", "content": message})
        return run

    def _append(self, db, user, session, run, body):
        mid = uuid4().hex
        db.execute(
            "INSERT INTO messages(id,user,session,run,body) VALUES (?,?,?,?,?)",
            (mid, user, session, run, dumps(body)),
        )
        return mid

    def append(self, user, session, run, body):
        with self.transaction() as db:
            return self._append(db, user, session, run, body)

    def messages(self, user, session):
        return [
            {"id": r["id"], "message": json.loads(r["body"])}
            for r in self.db.execute(
                "SELECT id,body FROM messages WHERE user=? AND session=? ORDER BY seq",
                (user, session),
            )
        ]

    def source(self, user, mid):
        row = self.db.execute(
            "SELECT session,body FROM messages WHERE user=? AND id=?", (user, mid)
        ).fetchone()
        return (
            None if row is None else {"session": row["session"], "message": json.loads(row["body"])}
        )

    def event(self, run, kind, body):
        with self.transaction() as db:
            db.execute(
                "INSERT INTO events(run,kind,body,created) VALUES (?,?,?,?)",
                (run, kind, dumps(body), now()),
            )

    def events(self, run):
        return [
            {"kind": r[0], **json.loads(r[1])}
            for r in self.db.execute(
                "SELECT kind,body FROM events WHERE run=? ORDER BY seq", (run,)
            )
        ]

    def finish(self, result):
        """Write the terminal row. Refuses to run without a real exit status.

        ``RunResult.status`` starts as ``None`` so that a code path which forgets
        to assign one cannot write ``"running"`` and leave a row that blocks the
        session forever (``active_session`` is unique on ``status='running'``).
        Raising here turns that bug into a failed write at the exact point it
        happened, instead of a session that can never be started again.
        """
        if result.status is None:
            raise StorageError(
                f"Run {result.run_id} reached finish() without an exit status"
            )
        with self.transaction() as db:
            db.execute(
                "UPDATE runs SET status=?, result=? WHERE id=? AND status='running'",
                (result.status, dumps(result.to_dict()), result.run_id),
            )
            db.execute(
                "INSERT INTO events(run,kind,body,created) VALUES (?,?,?,?)",
                (result.run_id, "run_end", dumps(result.to_dict()), now()),
            )

    def finish_session(self, user, session, *, reason, source):
        """Record an externally declared session/trial terminal state.

        Observation only: no part of the agent loop reads this row. `reason` and
        `source` are supplied by an external authority (the official orchestrator);
        the runtime computes solely the structural facts it can prove itself, so a
        caller cannot assert a run count or a terminal run that did not happen.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Termination reason must be a nonempty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Declaration source must be a nonempty string")
        with self.transaction() as db:
            runs = db.execute(
                "SELECT id FROM runs WHERE user=? AND session=? ORDER BY rowid",
                (user, session),
            ).fetchall()
            db.execute(
                "INSERT INTO session_terminations"
                "(user,session,reason,source,terminal_run,run_count,created) "
                "VALUES (?,?,?,?,?,?,?)",
                (user, session, reason, source,
                 runs[-1]["id"] if runs else None, len(runs), now()),
            )

    def session_terminal(self, user, session):
        """Latest declaration for a session, or None. Prior declarations are retained."""
        row = self.db.execute(
            "SELECT reason,source,terminal_run,run_count,created FROM session_terminations "
            "WHERE user=? AND session=? ORDER BY seq DESC LIMIT 1",
            (user, session),
        ).fetchone()
        return None if row is None else dict(row)

    def attempt(self, run, user, session, call, name, read_only, attempt, status, body):
        with self.transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO attempts VALUES (?,?,?,?,?,?,?,?,?)",
                (call, attempt, run, user, session, name, int(read_only), status, dumps(body)),
            )

    def unknown_writes(self, user, session):
        """Every write in this session whose outcome was never established.

        This is the context view: escalated calls stay listed so the model can
        verify their effect before repeating anything. Reads are excluded; their
        unknown outcome was handed back as a tool result and needs no ledger.
        """
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT call,name,attempt,run FROM attempts WHERE user=? AND session=? "
                "AND read_only=0 AND status='unknown' ORDER BY rowid",
                (user, session),
            )
        ]

    def unresolved(self, user, session):
        """Unknown-outcome writes that have not been escalated yet.

        ``Agent.run`` escalates each one exactly once; after that the session
        must be able to continue. The old contract returned every unknown row on
        every turn, and nothing could ever clear it, so a single write timeout
        welded the session shut: no model call was made again.
        """
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT a.call,a.name,a.attempt,a.run FROM attempts a WHERE a.user=? "
                "AND a.session=? AND a.read_only=0 AND a.status='unknown' "
                "AND NOT EXISTS (SELECT 1 FROM write_escalations e WHERE e.call=a.call) "
                "ORDER BY a.rowid",
                (user, session),
            )
        ]

    def escalate_writes(self, user, session, run, calls):
        """Record that the unknown outcome of each call has been surfaced.

        Idempotent by construction: a second escalation for the same call is a
        no-op, so a retried or duplicated escalation path cannot record the same
        delivery twice.
        """
        with self.transaction() as db:
            for call in calls:
                db.execute(
                    "INSERT OR IGNORE INTO write_escalations(call,user,session,run,created) "
                    "VALUES (?,?,?,?,?)",
                    (call, user, session, run, now()),
                )

    def recover(self, user, session):
        """Explicit crash recovery, never replays a tool. Ensure the old owner stopped first."""
        with self.transaction() as db:
            runs = db.execute(
                "SELECT id FROM runs WHERE user=? AND session=? AND status='running'",
                (user, session),
            ).fetchall()
            for row in runs:
                run = row[0]
                records = self.messages(user, session)
                completed = {r["message"].get("tool_call_id") for r in records}
                for record in records:
                    for call in record["message"].get("tool_calls", []):
                        if call["id"] in completed:
                            continue
                        attempt = db.execute(
                            "SELECT status,body FROM attempts WHERE call=? ORDER BY attempt DESC",
                            (call["id"],),
                        ).fetchone()
                        body = json.loads(attempt["body"]) if attempt else {}
                        body["status"] = attempt["status"] if attempt else "not_dispatched"
                        if body["status"] != "returned":
                            body["error"] = "Interrupted; inspect execution evidence"
                        self._append(
                            db,
                            user,
                            session,
                            run,
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": dumps(body),
                            },
                        )
                result = {
                    "run_id": run,
                    "status": "failed",
                    "error_source": "infra",
                    "reason": "interrupted",
                    "goal_status": "unverified",
                }
                db.execute(
                    "UPDATE runs SET status='failed',result=? WHERE id=?", (dumps(result), run)
                )
                db.execute(
                    "INSERT INTO events(run,kind,body,created) VALUES (?,?,?,?)",
                    (run, "run_end", dumps(result), now()),
                )
        return len(runs)
