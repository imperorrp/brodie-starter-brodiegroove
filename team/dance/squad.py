"""Squad mode: 2 to 4 dancers in one camera, each scored against the same routine on one shared music clock.

SquadEngine runs one PhraseEngine per player, all started at the same instant (one audio track, one beat grid). The first player's
engine drives the state, ghost and timeline; every engine scores its own dancer. At the end, squad_result() ranks them, picks the
MVP, measures Crew Sync (how alike the dancers moved, beat by beat) and picks the one dancer to coach (the one furthest off).
"""
import math
import numpy as np
from .engine import PhraseEngine
from .features import resample_to_grid, resample_mask
from . import progression

HALF_SYNC_DEG = 20.0


class SquadEngine:
    def __init__(self, key, players, on_event=None, **engine_kw):
        """players: [(pid, colour, name or None), ...] in display order."""
        self.players = list(players); self.on_event_cb = on_event; self.engines = {}
        for pid, colour, name in self.players:
            self.engines[pid] = PhraseEngine(key, on_event=self._tagger(pid, colour), **engine_kw)
        self.lead = self.engines[self.players[0][0]]; self.squad = None; self.key = key

    def _tagger(self, pid, colour):
        def cb(ev):
            ev = dict(ev); ev['pid'] = pid; ev['colour'] = colour
            if self.on_event_cb:
                try: self.on_event_cb(ev)
                except Exception: pass
        return cb

    # the lead engine's clock and state stand for the squad
    def __getattr__(self, name):
        if name in ('engines', 'lead'): raise AttributeError(name)
        return getattr(self.lead, name)

    @property
    def state(self): return self.lead.state

    @property
    def result(self): return self.lead.result

    def start(self, t):
        for e in self.engines.values(): e.start(t)

    def on_frame(self, t, lm_by_pid):
        for pid, e in self.engines.items():
            lm, w = lm_by_pid.get(pid, (None, None)); e.on_frame(t, lm, w)

    def tick(self, t):
        for e in self.engines.values(): e.tick(t)
        if self.squad is None and all(e.state in ('scored', 'feedback', 'done') for e in self.engines.values()) and self.lead.result is not None:
            self.squad = squad_result(self)

    def abort(self, t):
        for e in self.engines.values(): e.abort(t)

    @property
    def aborted(self): return getattr(self.lead, 'aborted', False)

    @property
    def cue(self):
        return (self.squad or {}).get('cue') or self.lead.cue

    def hud(self, t):
        h = self.lead.hud(t); rows = []
        for pid, colour, name in self.players:
            e = self.engines[pid]; eh = e.hud(t)
            rows.append(dict(pid=pid, colour=colour, name=name, running=eh['running_score'], combo=eh['combo'], verdicts=eh['verdicts'],
                             score=(e.result or {}).get('score') if (e.result or {}).get('scored') else None))
        h['squad'] = rows; h['squad_result'] = self.squad
        return h


def crew_sync(engines):
    """How alike the dancers moved: mean pairwise angle difference per beat on the shared grid -> exp(-ln2 * err / 20)."""
    es = [e for e in engines if e.live]
    if len(es) < 2: return dict(sync=None, per_beat=[])
    lead = es[0]; pb = lead.pb; grid = np.arange(lead.n_key) * lead.beat_s / pb
    data = []
    for e in es:
        ts = [r[0] for r in e.live]
        F = resample_to_grid(ts, [r[1] for r in e.live], grid); M = resample_mask(ts, [r[2].astype(np.float32) for r in e.live], grid)
        lo, hi = min(ts), max(ts); M[(grid < lo) | (grid > hi)] = False
        data.append((F, M))
    per_beat = []
    for b in range(lead.n_beats):
        sl = slice(b * pb, (b + 1) * pb); errs = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                both = data[i][1][sl] & data[j][1][sl]
                if both.any(): errs.append(float(np.abs(data[i][0][sl] - data[j][0][sl])[both].mean()))
        per_beat.append(None if not errs else round(math.exp(-math.log(2.0) * float(np.mean(errs)) / HALF_SYNC_DEG), 3))
    vals = [v for v in per_beat if v is not None]
    return dict(sync=round(float(np.mean(vals)), 3) if vals else None, per_beat=per_beat)


def squad_result(sq):
    names = {pid: (name or colour.upper()) for pid, colour, name in sq.players}
    colours = {pid: colour for pid, colour, name in sq.players}
    rows = []
    for pid, colour, name in sq.players:
        r = sq.engines[pid].result or {}
        rows.append(dict(pid=pid, colour=colour, name=name, label=names[pid], scored=bool(r.get('scored')), score=r.get('score') if r.get('scored') else None,
                         timing=r.get('timing'), grade=progression.grade(r.get('score') or 0) if r.get('scored') else None,
                         title=progression.title_for(r) if r.get('scored') else 'Camera Shy (step back)', max_combo=r.get('max_combo', 0)))
    scored = [x for x in rows if x['scored']]
    ranking = sorted(scored, key=lambda x: (-(x['score'] or 0), -(x['timing'] or 0)))
    for i, x in enumerate(ranking): x['rank'] = i + 1
    mvp = ranking[0] if ranking else None
    sync = crew_sync([sq.engines[pid] for pid, _, _ in sq.players])
    crew_mean = round(float(np.mean([x['score'] for x in scored])), 1) if scored else 0.0
    spread = (ranking[0]['score'] - ranking[-1]['score']) if len(ranking) >= 2 else 0.0
    cue = None
    if len(ranking) >= 2:
        worst = ranking[-1]; wc = sq.engines[worst['pid']].cue
        if wc and wc.get('kind') != 'praise':
            t = wc['text']; who = worst['name'] or worst['colour'].capitalize()
            cue = dict(wc, text=f"{who}, {t[0].lower() + t[1:]}", short=f"{worst['colour'].upper()}: {wc.get('short', '')}",
                                       pid=worst['pid'], colour=colours[worst['pid']])
    if cue is None and mvp is not None:
        cue = dict(sq.engines[mvp['pid']].cue or dict(text='Nice crew', kind='praise', group=None, beats=[]))
    unseen = [x for x in rows if not x['scored']]
    if unseen and len(unseen) == len(rows):
        cue = dict(text='Step back so I can see all of you', kind='visibility', group=None, beats=[])
    return dict(players=rows, ranking=ranking, mvp=mvp, crew_sync=sync['sync'], sync_per_beat=sync['per_beat'], crew_mean=crew_mean,
                spread=round(float(spread), 1), squad_title=progression.squad_title(sync['sync'], crew_mean, spread), cue=cue,
                unseen=[x['label'] for x in unseen])
