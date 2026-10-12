/* Poster runtime. Reads ?size=4x5|9x16|16x9 &slide=N ({slide} in a data-p key is replaced by N) &clean=1 &guides=1, keeps only the matching [data-slide] and
   [data-only] blocks, fills [data-p] text and the [data-cta] slot from window.PROMO (config.js, written by render.py). */
(function () {
  const q = new URLSearchParams(location.search);
  const P = window.PROMO || {};
  const size = q.get('size') || '4x5', slide = q.get('slide') || '1';
  const get = k => k.split('.').reduce((o, p) => (o == null ? o : o[p]), P);
  const root = document.querySelector('.poster');
  root.classList.add('s' + size);
  if (q.get('clean')) document.body.classList.add('clean');
  document.querySelectorAll('[data-slide]').forEach(el => { if (el.dataset.slide !== slide) el.remove(); else el.removeAttribute('data-slide'); });
  document.querySelectorAll('[data-only]').forEach(el => { if (!el.dataset.only.split(' ').includes(size)) el.remove(); });
  document.querySelectorAll('[data-p]').forEach(el => { const v = get(el.dataset.p.replace('{slide}', slide)); if (v != null && v !== '') el.textContent = v; });
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => `&#${c.charCodeAt(0)};`);
  const line = (P.call_line || '').trim();
  document.querySelectorAll('[data-cta]').forEach(el => {
    el.outerHTML = line
      ? `<div class="cta on"><div class="t"><span class="k">${esc(P.cta_kicker)}</span><span class="l">${esc(line)}</span></div>` +
        (P.qr ? `<img class="qr" src="${esc(P.qr)}">` : '') + `</div>`
      : `<div class="cta off"><i class="live"></i><div class="t"><span class="k">${esc(P.cta_pending_kicker)}</span>` +
        `<span class="l">${esc(P.cta_pending)}</span></div></div>`;
  });
  if (q.get('guides') && size === '9x16') {
    root.insertAdjacentHTML('beforeend', '<i class="guide" style="left:0;right:0;top:0;height:11.5%"></i>' +
      '<i class="guide" style="left:0;right:0;bottom:0;height:21.9%"></i><i class="guide" style="right:0;top:11.5%;bottom:21.9%;width:13%"></i>');
  }
})();
