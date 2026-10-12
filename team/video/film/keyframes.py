"""Model sheets and per-shot start keyframes for the pitch film (gemini-3-pro-image, cached).

Run:  python -m team.video.film.keyframes sheets          # Brodie + Maya model sheets, the two locations
      python -m team.video.film.keyframes shots [id ...]  # start frames, using the sheets as references
      add take=N (e.g. s04=2) to force a new variant of one item
Images land in data/tmp/film/cache/, and the chosen ones are copied to data/tmp/film/keys/<id>.png.
"""
import os, sys, json, shutil
from team.video.film.gen import image, FILM, ROOT

KIT = os.path.join(ROOT, 'team', 'dance', 'assets', 'brodie')
BRODIE_REFS = [os.path.join(KIT, 'brodie-flex.png'), os.path.join(KIT, 'brodie-sit-wave.png')]
CANON = json.load(open(os.path.join(KIT, 'brodie.json'), encoding='utf-8'))['prompt']['canon']
KEYS = os.path.join(FILM, 'keys')

WORLD = ('Stylised 3D animated feature-film look (Pixar / Spider-Verse energy), cinematic lighting, rich colour, '
         'vertical 9:16 frame. Palette accents: electric volt yellow-green #D9FF3D, hot pink #FF3DA8, cyan #3DE0FF '
         'against near-black #0A0A0B shadows.')
BRODIE_IN_WORLD = ('Brodie (from the reference images) appears exactly as a FLAT 2D cartoon sticker character come to life '
                   'inside the 3D world: flat cream #EFE7DA fills, thick even near-black outline, no shading, no 3D, never '
                   'pink, black tank top with readable arched cream BRODIE lettering and a dumbbell icon, a '
                   'huge round ball head with a three-lobe cream hair tuft, no ears, no nose, short stubby legs. '
                   'Like a hand-drawn toon in a 3D film.')
MAYA = ('Maya: a woman in her mid-20s, stylised 3D animated character with adult proportions (tall, long limbs, '
        'about 7.5 heads tall), adult face with defined cheekbones, warm brown skin, dark curly hair in one big high '
        'ponytail, small gold hoop earrings, oversized volt-green #D9FF3D hoodie, black cargo pants, chunky white sneakers.')

SHEETS = {
    'brodie_sheet': (BRODIE_REFS, 'Brodie as a dance coach: no hat, no props, a big grin, mid-dance: one fist punched '
                     'up to the sky, the other on his hip, one stubby leg kicked out. Keep his exact head shape from the '
                     'references: a huge round ball head with a three-lobe cream hair tuft, NO ears, NO nose. His face MUST '
                     'show two thick black bar eyebrows and two solid black oval eyes above the grin, exactly like the '
                     'flex reference. ' + CANON),
    'maya_sheet': ([], f'Character model sheet, plain light grey background, four full-body views in a row: front, '
                   f'three-quarter, side, and an awkward stiff dance pose with an embarrassed face. {MAYA} '
                   'She is clearly an adult (college graduate age): adult proportions and height, mature face. '
                   'Stylised 3D animated feature-film character design, clean studio lighting, no text, no blush.'),
}


def save_key(kid, path):
    """copy the chosen image to keys/, cropping any letterbox band the image model added (it would carry into Veo)."""
    import numpy as np
    from PIL import Image
    os.makedirs(KEYS, exist_ok=True)
    im = Image.open(path).convert('RGB')
    rows = np.asarray(im.convert('L')).astype(float)
    dark = rows.max(1) < 14                          # a band is uniformly black, not just a dim ceiling
    top = int(np.argmax(~dark)); bot = len(dark) - int(np.argmax(~dark[::-1]))
    if top > 8 or bot < len(dark) - 8:
        w, h = im.size
        im = im.crop((0, top, w, bot))
        s = h / im.height
        im = im.resize((round(w * s), h), Image.LANCZOS)
        x = (im.width - w) // 2
        im = im.crop((x, 0, x + w, h))
        print(f'{kid}: cropped a letterbox band ({top}px top, {len(dark) - bot}px bottom)')
    im.save(os.path.join(KEYS, kid + '.png'))
    print(f'{kid}: {path}')


def main():
    args = sys.argv[1:]
    takes = dict(a.split('=') for a in args if '=' in a)
    what = [a for a in args if '=' not in a]
    if not what or what[0] == 'sheets':
        for kid, (refs, prompt) in SHEETS.items():
            aspect = '1:1' if kid == 'brodie_sheet' else '16:9'
            save_key(kid, image(prompt, refs, aspect, name=kid, take=int(takes.get(kid, 0))))
    if what and what[0] == 'shots':
        from team.video.film.shots import SHOTS, key_prompt
        for sid in what[1:] or list(SHOTS):
            s = SHOTS[sid]
            refs = []
            if 'b' in s['who']:
                refs += [os.path.join(KEYS, 'brodie_sheet.png'), BRODIE_REFS[0]]
            if 'm' in s['who']:
                refs.append(os.path.join(KEYS, 'maya_sheet.png'))
            if s.get('like') and os.path.exists(os.path.join(KEYS, s['like'] + '.png')):
                refs.append(os.path.join(KEYS, s['like'] + '.png'))
            save_key(sid, image(key_prompt(sid), refs, '9:16', name=sid, take=int(takes.get(sid, 0))))


if __name__ == '__main__':
    main()
