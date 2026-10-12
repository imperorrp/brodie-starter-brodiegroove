"""Screens and pieces in the BrodieGroove site's style (team/web/groove/index.html, read-only): the daily board, the challenge code
as four volt digit tiles, squad tags (colour pill above each dancer's head, live score under their feet).

  draw_board(img, size, board)          board = dict(title, rows=[dict(rank, name, score, me)], rank, n)
  draw_code_tiles(img, cx, y, code, s0) the end card's code: four 42x52 tiles, ink fill, line border, Archivo volt digits
  draw_squad_tags(img, tags, s0)        tags = [dict(pts (33,2) px, vis, colour, label, score)]
"""
import numpy as np


def _r():
    from team.dance import render as r
    return r


def draw_code_tiles(img, cx, y, code, s0, w=42, h=52, left=None):
    r = _r(); tx = r.tx
    digits = str(code or '')[:4]
    if not digits: return img
    tw, th, gap = int(w * s0), int(h * s0), int(10 * s0)
    x0 = int(left) if left is not None else int(cx - (len(digits) * tw + (len(digits) - 1) * gap) / 2)
    for i, d in enumerate(digits):
        x = x0 + i * (tw + gap)
        r.rounded_rect(img, (x, y), (x + tw, y + th), r.INK, r=int(12 * s0))
        r.rounded_outline(img, (x, y), (x + tw, y + th), r.LINE, r=int(12 * s0), thick=max(1, int(s0)))
        tx.put_text(img, d, (x + tw // 2, y + th // 2), int(26 * s0), r.VOLT, font='display', anchor='mm')
    return img


def draw_board(img, size, board, t=0.0):
    """Today's daily board: an ink2 panel, the top rows (rank, first name, grade, score) and the caller's own row in volt."""
    r = _r(); tx = r.tx
    cw, ch = size; s0 = r.sc_of(size); sx, sy, sx1, sb = r.safe_box(size)
    rows = list(board.get('rows') or [])[:5]
    me = next((x for x in rows if x.get('me')), None)
    if me is None and board.get('rank'):
        rows = rows[:4] + [dict(rank=board['rank'], name=board.get('name') or 'YOU', score=board.get('score'), me=True)]
    pw = sx1 - sx; row_h = int(54 * s0); head_h = int(96 * s0)
    ph = head_h + row_h * max(1, len(rows)) + int(30 * s0)
    py = int(sy + max(r.pip_size(size)[1] + 16 * s0, (sb - sy - ph) / 2))
    r.rounded_rect(img, (sx, py), (sx + pw, py + ph), r.INK2, r=int(22 * s0), alpha=0.94)
    r.rounded_outline(img, (sx, py), (sx + pw, py + ph), r.LINE, r=int(22 * s0), thick=1)
    pad = int(22 * s0)
    tx.put_text(img, (board.get('title') or "TODAY'S DAILY").upper(), (sx + pad, py + int(26 * s0)), int(12 * s0), r.MUTE, font='grotesk')
    tx.put_text(img, 'THE BOARD', (sx + pad, py + int(64 * s0)), int(30 * s0), r.PAPER, font='display')
    if board.get('n'): tx.put_text(img, f"{board['n']} DANCERS", (sx + pw - pad, py + int(64 * s0)), int(12 * s0), r.MUTE, font='grotesk', anchor='rs')
    y = py + head_h
    for row in rows:
        mine = bool(row.get('me'))
        if mine: r.rounded_rect(img, (sx + int(10 * s0), y + int(4 * s0)), (sx + pw - int(10 * s0), y + row_h - int(4 * s0)), r.VOLT, r=int(14 * s0))
        fg = r.INK if mine else r.PAPER
        tx.put_text(img, f"#{row.get('rank', '')}", (sx + pad, y + row_h // 2), int(20 * s0), r.INK if mine else r.MUTE, font='display', anchor='lm')
        name = str(row.get('name') or 'anonymous').split()[0][:12]                 # first names only on boards
        tx.put_text(img, name, (sx + pad + int(66 * s0), y + row_h // 2), int(18 * s0), fg, anchor='lm')
        sc = row.get('score')
        if sc is not None:
            from team.dance import progression
            g = progression.grade(float(sc))
            tx.put_text(img, f"{float(sc):.0f}", (sx + pw - pad, y + row_h // 2), int(22 * s0), fg, font='display', anchor='rm')
            gc = r.GRADE_COL.get(g, r.PAPER)
            r.pill_c(img, g, sx + pw - pad - int(70 * s0), y + row_h // 2, int(14 * s0), r.VOLT if mine else r.INK, r.INK if mine else gc, font='display')
        y += row_h
    return img


def draw_squad_tags(img, tags, s0):
    """Each dancer's colour pill (name or colour) above their head, and their running score under their feet."""
    r = _r(); tx = r.tx
    for tg in tags:
        pts = np.asarray(tg['pts'], np.float32); vis = tg.get('vis')
        seen = np.ones(len(pts), bool) if vis is None else (np.asarray(vis, np.float32) > 0.5)
        if not seen[[11, 12]].all(): continue
        col = r.PLAYER.get(tg.get('colour'), r.VOLT)
        head_y = float(min(pts[0, 1] if seen[0] else pts[[11, 12], 1].min() - 40 * s0, pts[[11, 12], 1].min()))
        cx = float(pts[[11, 12], 0].mean())
        label = str(tg.get('label') or tg.get('colour') or '').upper()
        r.pill_c(img, label, int(cx), int(head_y - 34 * s0), int(15 * s0), r.INK, col, font='display')
        if tg.get('score') is not None:
            low = [j for j in (27, 28, 25, 26, 23, 24) if seen[j]]
            foot_y = float(pts[low, 1].max()) if low else float(pts[[11, 12], 1].max() + 200 * s0)
            foot_y = min(foot_y + 30 * s0, img.shape[0] - 30 * s0)
            tx.put_text(img, f"{float(tg['score']):.0f}", (int(cx), int(foot_y)), int(26 * s0), col, font='display', anchor='ms', stroke=2)
    return img
