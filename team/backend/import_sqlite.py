"""One-time copy of a local SQLite database into Postgres, so codes issued before the move keep working.

    set DATABASE_URL=postgresql://...
    .venv\\Scripts\\python team\\backend\\import_sqlite.py [path\\to\\team.db]      (default: team/backend/team.db)

Every table is copied row by row; rows that already exist (same id or code) are skipped, so it is safe to run twice.
The serial counters are moved past the highest imported id. Needs psycopg (team/backend/requirements.txt).
"""
import json, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import db  # noqa: E402

TABLES = ('events', 'dance_results', 'dance_chains', 'dance_classes', 'dance_profiles', 'dance_shares', 'dance_ghosts', 'dance_ghost_tokens')
SERIAL = ('events', 'dance_results', 'dance_chains', 'dance_classes', 'dance_shares')


def main(path):
    if db.engine() != 'postgres':
        print('DATABASE_URL is not set (or TEAM_DB is): nothing to import into'); return 2
    if not os.path.exists(path):
        print(f'no SQLite file at {path}'); return 2
    db.init()
    src = sqlite3.connect(path); src.row_factory = sqlite3.Row
    totals = {}
    for t in TABLES:
        cols = [r[1] for r in src.execute(f'PRAGMA table_info({t})').fetchall()]
        if not cols: continue
        rows = src.execute(f'SELECT * FROM {t}').fetchall()
        n = 0
        with db._lock, db._conn() as c:
            for r in rows:
                vals = [r[k] for k in cols]
                cur = c.execute(f'INSERT INTO {t} ({", ".join(cols)}) VALUES ({", ".join("?" * len(cols))}) ON CONFLICT DO NOTHING', vals)
                n += max(0, cur.rowcount)
            if t in SERIAL:
                c.execute(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), GREATEST(COALESCE(MAX(id), 0), 1)) FROM {t}")
        totals[t] = (n, len(rows))
        print(f'{t}: {n} of {len(rows)} rows imported')
    print(json.dumps(totals))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'team.db')))
