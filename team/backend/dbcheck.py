"""Round trip against the configured database: create the tables, write a result, read it back by code, delete it.

    set DATABASE_URL=postgresql://...      (Supabase > Connect > Transaction pooler URI; or leave unset to check the SQLite path)
    .venv\\Scripts\\python team\\backend\\dbcheck.py

Run it from the laptop before the first deploy, and against the deployed box with `railway run` if something looks off.
Exit code 0 when everything answered.
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import db  # noqa: E402


def main():
    t = time.time()
    print(f'engine: {db.engine()}' + (f'  ({db.DATABASE_URL.split("@")[-1]})' if db.engine() == 'postgres' else f'  ({db.DB_PATH})'))
    db.init(); print(f'init: ok ({time.time() - t:.2f} s)')
    if not db.ping(): print('ping: FAILED'); return 1
    print('ping: ok')
    code = db.add_result('dbcheck', 'test_squats', 'learn', 42.0, [dict(score=42.0)])
    r = db.get_result(code)
    ok = bool(r) and r['name'] == 'dbcheck' and abs(float(r['total']) - 42.0) < 1e-6
    print(f'write/read: {"ok" if ok else "FAILED"} (code {code})')
    prof = db.award('dbcheck', 10, style='hiphop')
    print(f'profile: {"ok" if prof["xp_after"] >= 10 else "FAILED"} (xp {prof["xp_after"]}, level {prof["level_after"]})')
    gone = db.delete_user('dbcheck')
    print(f'forget me: {gone}')
    print(f'leaderboard rows: {len(db.leaderboard(limit=5))}   funnel: {db.funnel()}')
    print(f'done in {time.time() - t:.2f} s')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
