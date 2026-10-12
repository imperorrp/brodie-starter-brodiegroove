"""The film's shots: a start-keyframe prompt (gemini-3-pro-image) and a motion prompt (Veo) for each.

who: which model sheets to attach ('b' Brodie, 'm' Maya). like: another shot's keyframe to attach for continuity.
"""
from team.video.film.keyframes import WORLD, BRODIE_IN_WORLD, MAYA

PARTY = ('A packed neon house party at night: a glowing volt-green LED grid dance floor, a DJ booth with hot pink and cyan '
         'light bars behind, haze, confetti in the air, friends dancing.')
BEDROOM = ('Maya\'s small cosy bedroom at night: a bed with rumpled sheets, a desk with a phone, warm fairy lights, a big '
           'window with a neon city skyline, abstract graphic posters with no lettering, a round rug.')

SHOTS = {
    's01': dict(who='m', secs=6, key=f'{PARTY} Maya in the centre of the crowd mid-flail, arms going in two directions, '
                'eyes squeezed shut, her elbow about to knock the cup of a surprised guy next to her. Medium-wide shot.',
                veo='The DJ drops the beat and the whole crowd dances in sync. Maya flails wildly off the beat, her elbow '
                'knocks the guy\'s cup and the drink splashes; people around her flinch and stare. Handheld camera, quick '
                'push in on Maya. Party music, crowd noise.'),
    's03': dict(who='m', secs=4, key=f'{BEDROOM} Maya lies face-down on the bed, still in her volt hoodie, one arm '
                'hanging off the edge, defeated. On the nightstand her phone lights up with a bright volt-green glow.',
                veo='The phone on the nightstand buzzes and lights up, pulsing volt-green light across the dark room. Maya '
                'slowly lifts her head from the pillow and squints at it. Slow push in. Phone buzzing, quiet room.'),
    's04': dict(who='b', secs=6, key=f'{BEDROOM} Close-up on the desk: the phone lies face up, its screen blasting a beam '
                'of volt-green light upward, and Brodie is halfway out of the screen, bursting out of the light with '
                'both fists up, grinning.',
                veo='Brodie leaps all the way out of the glowing phone screen in a burst of volt-green sparks, lands on '
                'the desk with a bounce, and strikes a double-biceps flex, grinning and talking excitedly to someone off '
                'screen. He stays a flat 2D cartoon with a thick black outline. Camera tilts up with him. Magical whoosh.'),
    's05': dict(who='bm', secs=4, like='s04', key=f'{BEDROOM} Over-the-shoulder from behind tiny Brodie standing on the '
                'desk in the foreground, looking at Maya, who sits bolt upright on the bed in shock, clutching her pillow.',
                veo='Maya yelps and tumbles backwards off the bed onto the rug, pillow flying. In the foreground Brodie '
                'points at her and talks, deadpan, then shrugs. Comedic timing. Thump, pillow whoosh.'),
    's06': dict(who='bm', secs=8, like='s05', key=f'{BEDROOM} Maya stands in the middle of the room facing the camera, '
                'nervous but trying. In front of her a glowing translucent volt-green hologram of a dancer, made of '
                'light, mid hip-hop move with arms in a V. Brodie stands on the bed beside them, clapping a beat.',
                veo='The volt-green hologram dancer performs a smooth hip-hop groove, and Maya copies each move a moment '
                'later, stiff at first, then finding the rhythm. Brodie bounces on the bed to the beat and claps. Slow '
                'orbit around Maya. Hip-hop beat.'),
    's07': dict(who='bm', secs=4, like='s06', key=f'{BEDROOM} Maya mid-stumble, tripping over her own feet, arms '
                'windmilling, a glowing red "MISS" sign made of light floating above her. Brodie on the bed wincing.',
                veo='Maya trips over her own sneakers and catches herself on the desk with a wobble. Brodie facepalms, '
                'then points at his own left arm, raising it high, explaining. Comedic. Thud.'),
    's08': dict(who='bm', secs=6, like='s06', key=f'{BEDROOM} Everything frozen in extreme slow motion: Maya mid-jump, '
                'curly ponytail floating, dust and fairy-light bokeh hanging in the air, and Brodie floating mid-air next '
                'to her in a relaxed lounging pose, calmly talking.',
                veo='Extreme slow motion: Maya drifts through a jump, her ponytail and hoodie floating, dust particles '
                'hanging in the air. Brodie floats beside her, lounging on nothing, calmly talking and gesturing slowly. '
                'Dreamy, time slowed down. Muffled slowed-down sound.'),
    's09a': dict(who='m', secs=4, like='s06', key=f'{BEDROOM} Golden sunset light pouring through the window, Maya '
                 'practising a hip-hop move in the middle of the room, sweaty, focused.',
                 veo='Time-lapse feeling: Maya practises the same hip-hop move again and again while the light through '
                 'the window shifts from golden sunset to deep blue night and the city lights switch on. Locked-off camera.'),
    's09b': dict(who='bm', secs=4, like='s06', key=f'{BEDROOM} Brodie on the desk with a tiny whistle around his neck '
                 'like a gym coach, clapping and cheering, Maya in the background mid-move, blurred.',
                 veo='Brodie claps and cheers like a hype coach, pumping his fist, while Maya dances behind him, more '
                 'confident with every move. Rack focus from Brodie to Maya. Cheering.'),
    's09c': dict(who='m', secs=4, like='s06', key=f'{BEDROOM} Morning light. Full body, front-on, Maya dancing a sharp '
                 'confident hip-hop move, whole body visible head to sneakers, big grin.',
                 veo='Maya dances a sharp, confident hip-hop eight-count, every hit on the beat, full body in frame head '
                 'to toe, front-on, then lands a pose and grins. Static camera. Hip-hop beat.'),
    's10': dict(who='m', secs=8, like='s01', key=f'{PARTY} Same party, same framing as before. Maya in the centre, '
                'confident, hitting a sharp hip-hop pose, the crowd stepping back into a circle around her, jaws dropping.',
                veo='The DJ drops the same beat. Maya nails a sharp, confident hip-hop routine, every move on the beat. The '
                'crowd backs into a circle around her and erupts, cheering, phones up filming. The camera stays in front '
                'of her with a clear view: nobody walks or stands between the camera and Maya, no foreground silhouettes. '
                'Smooth slow push in. Party music, crowd cheering.'),
    's11': dict(who='bm', secs=4, like='s10', key=f'{PARTY} Medium shot: Maya dancing confidently, and tiny Brodie '
                'standing on her shoulder doing the exact same move, both grinning, cheering crowd blurred behind.',
                veo='Maya and tiny Brodie on her shoulder hit the same hip-hop moves in perfect sync, both grinning; '
                'confetti falls, the crowd behind cheers. Slight slow push in. Crowd cheering.'),
    's12': dict(who='b', secs=4, like='s11', key=f'{PARTY} Close-up of Brodie standing on Maya\'s shoulder (only her '
                'shoulder and hoodie visible), turning to look straight into the camera with a confident smirk, party '
                'lights bokeh behind.',
                veo='Brodie turns to the camera, gives a cheeky wink and a point, and talks straight to the viewer with a '
                'grin. Shallow depth of field, party bokeh. Muffled party music.'),
}


def key_prompt(s):
    parts = [WORLD, SHOTS[s]['key']]
    if 'b' in SHOTS[s]['who']:
        parts.append(BRODIE_IN_WORLD + ' Brodie is tiny, about the height of a coffee mug.')
    if 'm' in SHOTS[s]['who']:
        parts.append(MAYA)
    if 'b' in SHOTS[s]['who']:
        parts.append('Brodie\'s face is exactly the reference: a bald round cream ball head with only the three-lobe '
                     'tuft, black bar brows, black oval eyes, no ears, no nose, never a human face.')
    if 'b' in SHOTS[s]['who']:
        parts.append('No text, no captions, no watermark. Only Brodie wears lettering; nobody else has any lettering or '
                     'logos on their clothes.')
    else:                                            # naming the word makes the model print it on a bystander
        parts.append('No text, lettering, logos or words anywhere in the image, plain clothes on everyone.')
    return ' '.join(parts)


def veo_prompt(s):
    p = SHOTS[s]['veo']
    if 'b' in SHOTS[s]['who']:
        p += (' Brodie is a flat 2D hand-drawn cartoon (cream fill, thick black outline, no shading) moving inside the '
              '3D animated scene.')
    return 'Stylised 3D animated film, vertical. ' + p + ' No subtitles, no text.'
