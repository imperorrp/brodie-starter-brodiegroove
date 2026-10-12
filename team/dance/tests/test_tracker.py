"""The multi-person tracker on synthetic people (a pose plus a shirt colour painted into the frame): crossing, stepping out and back,
a walk-in during a solo routine, a child in a squad, a crowd."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from team.dance.tracker import PlayerTracker

W, H = 384, 288
SHIRTS = dict(red=(220, 40, 40), blue=(40, 60, 220), green=(40, 200, 60), yellow=(230, 210, 40))


def person(cx, top=0.15, h=0.7):
    lm = np.zeros((33, 5), np.float32); lm[:, 3] = 1.0
    y = lambda f: top + f * h; s = h / 0.7
    pts = {0: (cx, y(0.0)), 11: (cx + 0.06 * s, y(0.15)), 12: (cx - 0.06 * s, y(0.15)), 13: (cx + 0.09 * s, y(0.3)), 14: (cx - 0.09 * s, y(0.3)),
           15: (cx + 0.1 * s, y(0.45)), 16: (cx - 0.1 * s, y(0.45)), 23: (cx + 0.04 * s, y(0.5)), 24: (cx - 0.04 * s, y(0.5)),
           25: (cx + 0.04 * s, y(0.75)), 26: (cx - 0.04 * s, y(0.75)), 27: (cx + 0.04 * s, y(1.0)), 28: (cx - 0.04 * s, y(1.0))}
    for j, p in pts.items(): lm[j, :2] = p
    lm[[j for j in range(33) if j not in pts], 3] = 0.0           # joints we don't place are unseen (not at the corner)
    return lm


def frame_of(people):
    """people: [(lm, shirt)]; paints each torso box in its shirt colour on a grey background (RGB)."""
    f = np.full((H, W, 3), 120, np.uint8)
    for lm, shirt in people:
        xs = lm[[11, 12, 23, 24], 0] * W; ys = lm[[11, 12, 23, 24], 1] * H
        f[int(ys.min()):int(ys.max()) + 1, int(xs.min()) - 4:int(xs.max()) + 5] = SHIRTS[shirt]
    return f


def run(tr, script, t0=0.0, dt=1 / 15, n=60):
    """script(i) -> [(lm, shirt)]."""
    t = t0
    for i in range(n):
        people = script(i)
        tr.update(t, [(lm, None) for lm, _ in people], frame_of(people)); t += dt
    return t


def pid_near(tr, x):
    st = [p for p in tr.stable_players() if p.lm is not None]
    return min(st, key=lambda p: abs(float(p.centre[0]) - x)).pid


def test_two_dancers_crossing_keep_their_ids():
    tr = PlayerTracker(); tr.squad = True
    t = run(tr, lambda i: [(person(0.3), 'red'), (person(0.7), 'blue')], n=40)
    red, blue = pid_near(tr, 0.3), pid_near(tr, 0.7)
    def cross(i):                                       # red walks right, blue walks left, through each other
        k = i / 59.0
        return [(person(0.3 + 0.4 * k), 'red'), (person(0.7 - 0.4 * k), 'blue')]
    t = run(tr, cross, t0=t, n=60)
    assert pid_near(tr, 0.7) == red and pid_near(tr, 0.3) == blue


def test_stepping_out_and_back_keeps_the_id_and_colour():
    tr = PlayerTracker(); tr.squad = True
    t = run(tr, lambda i: [(person(0.3), 'red'), (person(0.7), 'blue')], n=40)
    tr.assign_colours()
    blue = pid_near(tr, 0.7); colour = tr.players[blue].colour
    t = run(tr, lambda i: [(person(0.3), 'red')], t0=t, n=60)                 # blue gone for 4 s
    assert blue not in tr.players and blue in tr.lost
    t = run(tr, lambda i: [(person(0.3), 'red'), (person(0.65), 'blue')], t0=t, n=10)
    assert pid_near(tr, 0.65) == blue and tr.players[blue].colour == colour


def test_walk_in_never_takes_a_solo_routine():
    tr = PlayerTracker(); tr.activity = True
    t = run(tr, lambda i: [(person(0.5), 'red')], n=40)
    dancer = tr.primary_id
    t = run(tr, lambda i: [], t0=t, n=45)                                          # the dancer is hidden for 3 s
    t = run(tr, lambda i: [(person(0.45, top=0.05, h=0.85), 'green')], t0=t, n=30)   # a bigger person walks in
    assert tr.primary_id == dancer and tr.solo_pose()[0] is None
    t = run(tr, lambda i: [(person(0.45, top=0.05, h=0.85), 'green'), (person(0.6), 'red')], t0=t, n=10)
    assert tr.primary_id == dancer and tr.primary() is not None


def test_child_counts_in_squad_but_not_solo():
    child = lambda: person(0.7, top=0.45, h=0.21)                                  # 0.3 of the adult's size
    tr = PlayerTracker(); tr.squad = True
    run(tr, lambda i: [(person(0.3), 'red'), (child(), 'yellow')], n=40)
    assert tr.stable_count() == 2
    tr2 = PlayerTracker()
    run(tr2, lambda i: [(person(0.3), 'red'), (child(), 'yellow')], n=40)
    assert tr2.stable_count() == 1


def test_crowd_is_noted():
    tr = PlayerTracker(); tr.squad = True
    run(tr, lambda i: [(person(x), s) for x, s in zip((0.15, 0.38, 0.62, 0.85), SHIRTS)], n=40)
    assert any(e['type'] == 'crowd' for e in tr.pop_events())
