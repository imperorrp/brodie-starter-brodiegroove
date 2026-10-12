"""How each style should *feel*: a small authored vocabulary, keyed by the AIST++ genre code in a phrase's source name
(aist_lh_..., aist_ho_...) or 'kpop' for the custom K-pop keys. The cue layer (cues.py) picks the line whose tags match the move's
measured dynamics (sharp hits vs smooth flow, bouncy, low, big), so a smooth house phrase never gets told to "hit it hard".

Every line is short enough to be spoken once after a demo (12 words or fewer), in second person, no jargon a beginner can't follow.
"""

GENRE_OF = dict(lh='hiphop', mh='hiphop', br='hiphop', ho='house', po='pop', lo='lock', wa='waack', kr='krump', js='jazz', jb='jazz',
                kpop='kpop', fitness='fitness', squats='fitness')

FEEL = dict(
    hiphop=[('bouncy', "Stay low and loose. Let your knees bounce on every beat."),
            ('sharp', "Hit each pose on the beat, then relax into the next."),
            ('smooth', "Keep it loose and groovy, like you're nodding to the music."),
            ('big', "Make it big. Reach all the way out on every count.")],
    house=[('bouncy', "Light on your feet. Let your chest bounce with the beat."),
           ('smooth', "Keep it flowing. Your feet stay light and quick."),
           ('sharp', "Small, quick steps. Snap your chest on the beat."),
           ('big', "Let your whole body ride the groove.")],
    pop=[('sharp', "Hit, then freeze. Tighten every muscle for a split second."),
         ('smooth', "Glide between poses. Control every inch."),
         ('bouncy', "Pop on the beat, release between beats."),
         ('big', "Big clean lines. Hit, hold, move on.")],
    lock=[('sharp', "Point, then freeze. Sharp stops, big smile."),
          ('bouncy', "Keep the bounce in your knees between the locks."),
          ('smooth', "Roll your wrists smoothly, then lock it in."),
          ('big', "Big arms, loud energy. Show off every point.")],
    waack=[('sharp', "Whip from the shoulder. Loose wrists, sharp finish."),
           ('smooth', "Let the arms flow. Pose at the end of each line."),
           ('big', "Big circles with your arms. Strike the pose proud."),
           ('bouncy', "Whip on the beat, bounce between.")],
    krump=[('sharp', "Heavy and strong. Pop your chest like you mean it."),
           ('big', "Go big. Stomp, chest out, all the energy."),
           ('bouncy', "Stay grounded. Bend your knees and stay heavy."),
           ('smooth', "Build it slow, then explode on the hit.")],
    jazz=[('smooth', "Long lines. Finish every arm all the way out."),
          ('sharp', "Clean and sharp. Point your fingers at the end."),
          ('big', "Reach long and tall. Show the full line."),
          ('bouncy', "Keep it light and springy on your feet.")],
    kpop=[('sharp', "Crisp and in sync. Every pose lands right on the beat."),
          ('smooth', "Smooth and confident. Keep your face to the camera."),
          ('big', "Big, clean arms. Perform it to the camera."),
          ('bouncy', "Light bounce in the knees, sharp arms on top.")],
    fitness=[('bouncy', "Keep moving and keep breathing. Soft knees."),
             ('big', "Big moves, full range. Push through every rep."),
             ('sharp', "Strong and steady. Squeeze at the end of each move."),
             ('smooth', "Steady pace. Breathe out on the effort.")],
)
GENRE_WORD = dict(hiphop='hip-hop', house='house', pop='popping', lock='locking', waack='waacking', krump='krump', jazz='jazz', kpop='K-pop',
                  fitness='cardio')


def genre_of_src(src, fallback='hiphop'):
    """'aist_lh_110_ch03' -> 'hiphop'; 'cc_huntrix' -> 'kpop'."""
    s = str(src or '')
    if s.startswith('cc_') or 'kpop' in s: return 'kpop'
    parts = s.split('_')
    if len(parts) > 1 and parts[0] == 'aist': return GENRE_OF.get(parts[1], fallback)
    return GENRE_OF.get(s, fallback)


def feel_line(genre, tags):
    """The first line for this genre whose tag is in `tags` (ordered by how strongly the move shows it)."""
    lines = FEEL.get(genre) or FEEL['hiphop']
    for t in tags:
        for tag, line in lines:
            if tag == t: return line
    return lines[0][1]
