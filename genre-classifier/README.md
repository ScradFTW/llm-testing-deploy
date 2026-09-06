# genre-classifier — a from-scratch trained model on bradjobe.dev

Live at: https://bradjobe.dev/genre-classifier/ (same credentials as `/llm-testing`)

Where `/llm-testing` shows *serving* an existing open-weights LLM, this one
shows the other half: collecting data, training a model from scratch,
evaluating it honestly, and shipping the version that actually earns its
place in production.

## The task

Predict a song's genre from its **title text alone** — no audio features,
no lyrics. This is a deliberately hard, low-signal task (most titles don't
encode genre at all); the point is the pipeline and the honesty of the
evaluation, not a high accuracy number.

## Data

[Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
(114k rows, 114 raw Spotify "genre" tags — many of which are moods/contexts
like `chill`, `study`, `sleep`, `party`, or nationality tags like `german`,
`swedish`, not real genres). `train/prepare_data.py`:

1. Maps the 114 raw tags to 11 coherent genres (Rock, Metal, Pop, Hip-Hop,
   Electronic, R&B/Soul, Country/Folk, Jazz/Blues/Classical, Latin, Reggae,
   World), dropping tags that aren't genres at all.
2. Deduplicates by (title, artist) — the same song often appears under
   several raw tags, which is a real label-noise source. Deduplication
   collapses over 100k raw rows into ~58k clean examples, and drops the
   ~13% of songs whose collapsed genre is still ambiguous.
3. Caps the largest classes at 4,500 examples so a few genres don't
   dominate training, leaving 38,281 examples (Hip-Hop stays a minority
   class at 448 examples — a real imbalance the eval accounts for).
4. Splits 80/10/10 train/val/test.

## Models compared (`train/`)

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Majority-class baseline | 12.1% | — |
| PyTorch embedding + MLP (`train.py`) | 36.1% | 0.334 |
| **TF-IDF (char n-grams) + Logistic Regression (`train_final.py`)** | **40.9%** | **0.376** |

The classical model won on accuracy, F1, *and* deployment cost — smaller
artifact (2.4MB vs. needing a full PyTorch runtime), sub-10ms inference on
a single CPU core, no GPU/torch dependency on the server at all. Shipped
that one. Full report: `train/eval_report.txt`.

This is the intended takeaway: a from-scratch neural net was tried first,
benchmarked fairly against a much simpler classical baseline, and the
simpler model won — so it's what's in production. Reaching for deep
learning by default, without checking whether a cheaper model gets you
further, is the mistake being avoided here.

## Serving (`serve/`, `systemd/`)

- `serve/app.py` — a small Flask app (`/health`, `/predict`) that loads the
  joblib-pickled scikit-learn pipeline once at startup.
- Served by `waitress` (a real WSGI server, not Flask's dev server) via
  `systemd/genre-classifier.service` — same pattern as `llama-server`:
  `Restart=always`, `MemoryMax` cap, sandboxed, loopback-only
  (`127.0.0.1:8083`).
- nginx (`../nginx/genre-classifier.conf`, and the `/genre-classifier*`
  blocks in `../nginx/bradjobe.dev-site.conf`) reverse-proxies, rate-limits,
  and gates it behind the same HTTP Basic Auth as `/llm-testing`.
- Static frontend served from `/var/www/genre-classifier/` — outside
  `/var/www/html` for the same reason as `/llm-testing` (see the top-level
  README's note on the portfolio's own deploy process).

## Reproducing

```
cd train
python3 -m venv venv && source venv/bin/activate
pip install torch scikit-learn joblib  # torch only needed for the comparison run
curl -sL -o dataset.csv "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
python3 prepare_data.py     # writes train/val/test.jsonl
python3 train.py            # PyTorch baseline (optional, for comparison)
python3 baseline_tfidf.py   # quick TF-IDF/LogReg check against the val split
python3 train_final.py      # trains the shipped model on train+val, evals on test
```

## What's next if this graduated beyond a demo

- Train the final model on more data / try char n-gram range tuning and
  per-class threshold calibration rather than argmax.
- Track predictions + (eventually) ground truth to build a real online
  eval loop, not just a static test-set number.
- Version the model artifact (e.g. a model registry) instead of a single
  `genre_pipeline.joblib` on disk.
