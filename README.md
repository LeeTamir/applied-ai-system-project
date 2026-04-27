# VibeFinder: AI-Enhanced Music Recommender

## Base Project

**Music Recommender Simulation (Project 3)** — VibeFinder 1.0 was built as a transparent, content-based music recommender. Given a user's preferred genre, mood, energy level, and acoustic preference, it scored every song in an 18-track catalog using a fixed weighted formula and returned the top-k matches with human-readable explanations. The goal was to understand how real-world recommenders translate user taste into ranked outputs, and to surface the biases that simple scoring rules can introduce.

This project (VibeFinder 2.0) extends that foundation by adding a **RAG-powered natural language interface**, a **hybrid scoring layer** that combines semantic similarity with CSV feature values, **genre detection**, a **Streamlit web UI**, and a **full unit test suite**.

---

## Demo Walkthrough

[Watch the demo on Loom](https://www.loom.com/share/e7b57cc13dc046159584e32ea0d59843)

---

## Title and Summary

**VibeFinder** is a music recommendation system that lets users describe what they want to hear in plain English — *"something calm and acoustic for studying"* — and returns the best matches from a 50-song catalog. It works in two modes:

- **Structured mode** (`src/recommender.py`): the original rules-based scorer from Project 3, still available via `src/main.py`
- **RAG + Hybrid mode** (`src/rag_retriever.py` + `app.py`): semantic retrieval combined with numeric feature re-ranking, exposed through a Streamlit web app

The system is fully transparent: every result shows which signals were detected and why each song was chosen.

---

## Architecture Overview

See [assets/system_diagram.md](assets/system_diagram.md) for the full Mermaid diagram.

```
Natural Language Query (e.g. "a happy rnb song")
        │
        ▼
  Text Encoder: multi-qa-MiniLM-L6-cos-v1
        │  encodes query into a dense vector
        ▼
  Cosine Similarity Search  ◄──── Pre-encoded Song Embeddings (songs.csv)
        │  RAG retrieval step
        ▼
  Feature Signal Detection
        │  keyword scan → maps query terms to CSV columns (energy, valence, etc.)
        │  genre detection → binary genre match signal
        ▼
  Hybrid Re-ranking
        │  hybrid_score = 0.5 × semantic_similarity + 0.5 × feature_score
        ▼
  Top-K Results  →  Streamlit UI (app.py)
```

**RAG component:** The 50-song catalog is the knowledge base. Every song is pre-encoded into a dense vector at startup using `song_to_text()`, which converts structured CSV data into natural language sentences. At query time, the user's input is encoded the same way and compared by cosine similarity — the system retrieves grounded catalog data rather than relying on model training knowledge.

**Hybrid scoring:** When feature keywords are detected in the query (e.g. *"energetic"* → `energy ↑`, *"acoustic"* → `acousticness ↑`, *"rnb"* → `genre:rnb`), the semantic score is blended with a numeric feature score computed directly from the CSV. When no signals are detected, the result falls back to pure semantic ranking.

**Structured scoring rules (base project, `recommender.py`):**

| Signal | Points |
|---|---|
| Genre match | +1.8 |
| Mood match | +1.4 |
| Energy closeness | up to +2.0 |
| Acoustic preference bonus | +0.6 |

---

## Getting Started

### Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac / Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Run the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Type a natural language description, choose how many results to return, and click **Find Songs**.

### Run the CLI (Structured Mode)

```bash
python -m src.main
```

Runs five structured profiles (normal, edge case, adversarial) through the rules-based scorer and prints scored results with explanations.

### Run Tests

```bash
pytest
```

94 tests across two files — all pass in under 5 seconds with no model download required.

---

## Feature Signal Detection

The system scans the query for keywords that map to CSV columns:

| Keyword examples | Feature boosted |
|---|---|
| *energetic, workout, pump, hype* | `energy ↑` |
| *calm, chill, study, rainy, sleep* | `energy ↓` |
| *acoustic, unplugged, folk, organic* | `acousticness ↑` |
| *electronic, synth, edm, digital* | `acousticness ↓` |
| *happy, uplifting, feel-good, bright* | `valence ↑` |
| *sad, dark, melancholic, heartbreak* | `valence ↓` |
| *dance, groove, party, bop, club* | `danceability ↑` |
| *high bpm, uptempo, fast tempo* | `tempo_bpm ↑` |
| *slow bpm, downtempo, low tempo* | `tempo_bpm ↓` |

Genre names are detected separately (pop, lofi, rock, jazz, rnb, hip hop, metal, edm, folk, country, reggae, synthwave, classical, indie pop, ambient). Longer phrases are checked before shorter ones — *"indie pop"* will not accidentally match as plain *"pop"*.

Tags displayed under each result reflect what was detected: if the query mentioned danceability, the song shows its `dance` value; if nothing was detected, it falls back to showing `energy`.

---

## Song Catalog

The catalog lives in `data/songs.csv` — 50 songs, 10 features each:

| Field | Type | Description |
|---|---|---|
| `id` | int | Unique identifier |
| `title` | str | Song title |
| `artist` | str | Artist name |
| `genre` | str | One of 15 genres |
| `mood` | str | One of 15 moods |
| `energy` | float [0–1] | Intensity level |
| `tempo_bpm` | float | Beats per minute |
| `valence` | float [0–1] | Emotional positivity |
| `danceability` | float [0–1] | Rhythm suitability |
| `acousticness` | float [0–1] | Acoustic vs. electronic |

Genre distribution: ambient ×3, classical ×2, country ×3, edm ×3, folk ×3, hip hop ×4, indie pop ×3, jazz ×3, lofi ×6, metal ×3, pop ×5, reggae ×2, rnb ×4, rock ×3, synthwave ×3.

---

## Sample Interactions (Streamlit)

### Query: "something calm and acoustic for studying on a rainy day"

**Detected signals:** `energy ↓`, `acousticness ↑`

| # | Song | Tags |
|---|---|---|
| 1 | Library Rain — Paper Lanterns | `lofi` · `chill` · `energy 0.35` · `acoustic 0.86` |
| 2 | Spacewalk Thoughts — Orbit Bloom | `ambient` · `chill` · `energy 0.28` · `acoustic 0.92` |
| 3 | Mountain Letters — Pine Harbor | `folk` · `reflective` · `energy 0.33` · `acoustic 0.88` |

### Query: "a happy rnb song"

**Detected signals:** `valence ↑`, `genre:rnb`

| # | Song | Tags |
|---|---|---|
| 1 | Velvet Hours — Luna Vale | `rnb` · `romantic` · `valence 0.73` |
| 2 | Afterglow — Luna Vale | `rnb` · `romantic` · `valence 0.78` |
| 3 | Honey & Smoke — Amber Vale | `rnb` · `romantic` · `valence 0.76` |

### Query: "energetic workout pump"

**Detected signals:** `energy ↑`, `danceability ↑`

| # | Song | Tags |
|---|---|---|
| 1 | Gym Hero — Max Pulse | `pop` · `intense` · `energy 0.93` · `dance 0.88` |
| 2 | Neon Pulse — Vector Drift | `edm` · `euphoric` · `energy 0.92` · `dance 0.91` |
| 3 | Shatter — Iron Lung | `metal` · `aggressive` · `energy 0.97` · `dance 0.54` |

---

## Project Structure

```
applied-ai-system-final/
├── app.py                     # Streamlit web UI
├── data/
│   └── songs.csv              # 50-song catalog (10 features each)
├── src/
│   ├── main.py                # CLI: structured profile runner
│   ├── recommender.py         # Base project: rules-based scorer + Recommender class
│   └── rag_retriever.py       # RAG retriever + hybrid scoring + genre/feature detection
├── tests/
│   ├── test_recommender.py    # 22 unit tests for recommender.py
│   └── test_rag_retriever.py  # 72 unit tests for rag_retriever.py
├── assets/
│   └── system_diagram.md      # Mermaid architecture diagram
├── model_card.md              # Limitations, bias, misuse, AI collaboration reflection
├── reflection.md              # Manual profile comparison notes (base project)
├── requirements.txt
├── pytest.ini
└── .streamlit/
    └── config.toml            # fileWatcherType=none (suppresses torchvision warnings)
```

---

## Design Decisions

**Why `multi-qa-MiniLM-L6-cos-v1` instead of `all-MiniLM-L6-v2`?**
`all-MiniLM-L6-v2` is a symmetric model trained to compare sentence pairs of similar length. Matching a short user query against longer song descriptions is an *asymmetric* retrieval task. `multi-qa-MiniLM-L6-cos-v1` is specifically trained for query-document retrieval, which produced meaningfully higher semantic scores (~60–70% peak vs ~35% peak) for the same catalog.

**Why hybrid scoring instead of pure semantic retrieval?**
The semantic model encodes meaning well but cannot reliably distinguish a song with `energy=0.28` from one with `energy=0.82` — those values live in CSV columns, not in the text description. Hybrid scoring feeds the actual numeric CSV values into the ranking formula, so feature-keyword queries (e.g. *"high bpm"*) produce results that reflect the real data.

**Why keyword-based feature detection instead of an LLM?**
An LLM-based feature extractor would require an API key and external network access. Keyword scanning is deterministic, auditable, and runs instantly. The trade-off is brittleness: negation phrases like "not energetic" will not correctly flip the signal. This is documented as a known limitation in `model_card.md`.

**Why sentence-transformers instead of a paid LLM API?**
`sentence-transformers` runs locally, downloads once (~22MB), and has no per-call cost. It is fully reproducible without accounts or credentials.

**Why expose dynamic feature tags in the UI?**
Showing the feature that was actually detected (e.g. `dance 0.91` when danceability was queried) closes the loop for the user — they can see that the system understood their query and which specific CSV value it acted on.

---

## Testing Summary

**94 automated tests** across two files — run with `pytest` in under 5 seconds:

| File | Tests | What is covered |
|---|---|---|
| `tests/test_recommender.py` | 22 | `score_song` signal accumulation, acoustic bonus conditions, energy closeness math, `recommend_songs` sort order and k-limits, `Recommender` class explain output |
| `tests/test_rag_retriever.py` | 72 | `song_to_text` energy labels and acoustic phrases; `_detect_genre` aliases and longest-match priority; `_detect_features` positive/negative keyword routing; `_compute_feature_score` direction scoring, genre binary match, BPM normalization and clamping; `retrieve` / `hybrid_retrieve` guard clauses, sort order, detected signals, semantic fallback, genre and energy boosting |

The RAG retriever tests use identity-matrix torch embeddings and a mock encoder — no model download required to run the suite.

**Manual testing** via the Streamlit UI confirmed:
- Genre detection correctly surfaces rnb songs for "a happy rnb song" and pop songs for "a happy pop song"
- Low-energy queries ("calm study rainy day") consistently return songs with `energy < 0.45`
- The phrase "low energy" correctly triggers `energy ↓` rather than matching the shorter positive keyword "energy"
- BPM queries ("high bpm", "low bpm") correctly use normalized tempo values from the CSV

---

## Reflection

Building VibeFinder clarified something that is easy to miss when using commercial recommenders: the model is not discovering taste, it is measuring a distance defined entirely by whoever chose the features and weights. Every design choice — which features to score, how much weight to assign, what counts as an acoustic song — encodes assumptions about what music listeners care about. Those assumptions can exclude whole categories of users whose preferences do not map cleanly onto the chosen feature space.

Adding the RAG layer reinforced a different lesson: retrieval and generation are more useful when they stay coupled to grounded data. Rather than letting a language model hallucinate a playlist, the system retrieves semantically plausible candidates first and then applies transparent scoring. The result is easier to audit, easier to debug, and easier to trust — qualities that matter as much in a student project as they do in a production system.
