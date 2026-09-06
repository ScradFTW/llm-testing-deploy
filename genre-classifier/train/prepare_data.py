import csv
import json
import random
from collections import defaultdict, Counter

GENRE_MAP = {
    "acoustic": "Country/Folk", "afrobeat": "World", "alt-rock": "Rock",
    "alternative": "Rock", "ambient": None, "anime": None,
    "black-metal": "Metal", "bluegrass": "Country/Folk", "blues": "Jazz/Blues/Classical",
    "brazil": "Latin", "breakbeat": "Electronic", "british": None,
    "cantopop": "Pop", "chicago-house": "Electronic", "children": None,
    "chill": None, "classical": "Jazz/Blues/Classical", "club": "Electronic",
    "comedy": None, "country": "Country/Folk", "dance": "Electronic",
    "dancehall": "Reggae", "death-metal": "Metal", "deep-house": "Electronic",
    "detroit-techno": "Electronic", "disco": "Electronic", "disney": None,
    "drum-and-bass": "Electronic", "dub": "Reggae", "dubstep": "Electronic",
    "edm": "Electronic", "electro": "Electronic", "electronic": "Electronic",
    "emo": "Rock", "folk": "Country/Folk", "forro": "Latin", "french": None,
    "funk": "R&B/Soul", "garage": "Electronic", "german": None,
    "gospel": "R&B/Soul", "goth": "Rock", "grindcore": "Metal",
    "groove": "R&B/Soul", "grunge": "Rock", "guitar": None, "happy": None,
    "hard-rock": "Rock", "hardcore": "Metal", "hardstyle": "Electronic",
    "heavy-metal": "Metal", "hip-hop": "Hip-Hop", "honky-tonk": "Country/Folk",
    "house": "Electronic", "idm": "Electronic", "indian": "World",
    "indie": "Rock", "indie-pop": "Pop", "industrial": "Metal",
    "iranian": "World", "j-dance": "Electronic", "j-idol": "Pop",
    "j-pop": "Pop", "j-rock": "Rock", "jazz": "Jazz/Blues/Classical",
    "k-pop": "Pop", "kids": None, "latin": "Latin", "latino": "Latin",
    "malay": "World", "mandopop": "Pop", "metal": "Metal",
    "metalcore": "Metal", "minimal-techno": "Electronic", "mpb": "Latin",
    "new-age": None, "opera": "Jazz/Blues/Classical", "pagode": "Latin",
    "party": None, "piano": None, "pop": "Pop", "pop-film": "Pop",
    "power-pop": "Rock", "progressive-house": "Electronic", "psych-rock": "Rock",
    "punk": "Rock", "punk-rock": "Rock", "r-n-b": "R&B/Soul", "reggae": "Reggae",
    "reggaeton": "Latin", "rock": "Rock", "rock-n-roll": "Rock",
    "rockabilly": "Country/Folk", "romance": None, "sad": None,
    "salsa": "Latin", "samba": "Latin", "sertanejo": "Latin",
    "show-tunes": None, "singer-songwriter": "Country/Folk", "ska": "Reggae",
    "sleep": None, "songwriter": "Country/Folk", "soul": "R&B/Soul",
    "spanish": "Latin", "swedish": None, "synth-pop": "Pop", "tango": "Latin",
    "techno": "Electronic", "trance": "Electronic", "trip-hop": "Electronic",
    "turkish": "World", "world-music": "World",
}

random.seed(42)

rows = list(csv.DictReader(open("dataset.csv")))
print("raw rows:", len(rows))
print("unmapped genres:", set(r["track_genre"] for r in rows) - set(GENRE_MAP))

groups = defaultdict(set)
first_seen_title = {}
for row in rows:
    title = row["track_name"].strip()
    artist = row["artists"].strip()
    if not title:
        continue
    broad = GENRE_MAP.get(row["track_genre"])
    if broad is None:
        continue
    key = (title.lower(), artist.lower())
    groups[key].add(broad)
    first_seen_title.setdefault(key, title)

clean = []
for key, broads in groups.items():
    if len(broads) != 1:
        continue  # ambiguous across genres after collapsing -> drop
    (genre,) = broads
    clean.append({"title": first_seen_title[key], "genre": genre})

print("clean examples:", len(clean))
counts = Counter(c["genre"] for c in clean)
for g, n in counts.most_common():
    print(f"  {g:22s} {n}")

# Cap the largest classes so a few don't dominate training, then split.
CAP = 4500
by_genre = defaultdict(list)
for c in clean:
    by_genre[c["genre"]].append(c)
balanced = []
for g, items in by_genre.items():
    random.shuffle(items)
    balanced.extend(items[:CAP])
random.shuffle(balanced)

print("balanced total:", len(balanced))
counts2 = Counter(c["genre"] for c in balanced)
for g, n in counts2.most_common():
    print(f"  {g:22s} {n}")

n = len(balanced)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
test = balanced[:n_test]
val = balanced[n_test:n_test + n_val]
train = balanced[n_test + n_val:]

for name, split in [("train", train), ("val", val), ("test", test)]:
    with open(f"{name}.jsonl", "w") as f:
        for row in split:
            f.write(json.dumps(row) + "\n")
    print(name, len(split))
