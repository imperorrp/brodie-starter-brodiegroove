/* Runtime for the animated pieces in team/promo/anim/, driven frame by frame by animate.py (or just played in a browser).
   ?size=1x1|4x5|9x16|16x9 &v=<variant> &alpha=1. Keeps only the [data-v] / [data-only] blocks that match, fills [data-p] from
   window.PROMO (config.js) and [data-line] with the call line, or "@handle · #hashtag" while config.json has no call_line.
   window.seek(t) pauses every CSS animation at t seconds and calls window.onSeek(t) for anything scripted (counters).
   window.ANIM_READY turns true once the fonts and images have loaded. */
(function () {
  const q = new URLSearchParams(location.search);
  const P = window.PROMO || {};
  const size = q.get('size') || '1x1', v = q.get('v') || '';
  const get = k => k.split('.').reduce((o, p) => (o == null ? o : o[p]), P);
  const root = document.querySelector('.poster');
  root.classList.add('s' + size);
  document.body.dataset.v = v;
  if (q.get('alpha')) document.documentElement.classList.add('alpha');
  // animate.py renders the static background once (?plate=1: only .bg, grain and vignette), then every frame with ?flat=<that PNG>
  // behind the moving layer: redrawing the blurred blobs and the grain blend was 3/4 of the time per frame
  if (q.get('plate')) document.documentElement.classList.add('plate');
  if (q.get('flat')) {
    document.documentElement.classList.add('flat');
    root.insertAdjacentHTML('afterbegin', `<img class="plateimg" src="${q.get('flat').replace(/"/g, '')}">`);
  }
  document.querySelectorAll('[data-v]').forEach(el => { if (!el.dataset.v.split(' ').includes(v)) el.remove(); });
  document.querySelectorAll('[data-only]').forEach(el => { if (!el.dataset.only.split(' ').includes(size)) el.remove(); });
  document.querySelectorAll('[data-p]').forEach(el => { const x = get(el.dataset.p); if (x != null && x !== '') el.textContent = x; });
  const line = (P.call_line || '').trim();
  document.querySelectorAll('[data-line]').forEach(el => {
    el.textContent = line ? (el.dataset.line ? el.dataset.line + ' ' : '') + line : `${P.handle || '@brodiegroove'} · ${P.hashtag || '#BeatMyBrodie'}`;
  });
  window.ANIM_V = v; window.ANIM_SIZE = size;
  window.seek = t => {
    document.getAnimations().forEach(a => { a.pause(); a.currentTime = t * 1000; });
    if (window.onSeek) window.onSeek(t);
  };
  const imgs = [...document.images].map(i => (i.complete ? Promise.resolve() : i.decode()).catch(() => {}));
  Promise.all([document.fonts.ready, ...imgs]).then(() => { window.ANIM_READY = true; });
})();
