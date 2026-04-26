# VibeFinder: AI-Enhanced Music Recommender

## Original Project

**Music Recommender Simulation (Modules 1–3)** — VibeFinder 1.0 was built as a transparent, content-based music recommender. Given a user's preferred genre, mood, energy level, and acoustic preference, it scored every song in an 18-track catalog using a fixed weighted formula and returned the top-k matches with human-readable explanations. The goal was to understand how real-world recommenders translate user taste into ranked outputs, and to surface the biases that simple scoring rules can introduce.

---

## Title and Summary

**VibeFinder** is a music recommendation system that combines a rules-based content scorer with semantic retrieval powered by `sentence-transformers`. Users can describe what they want to hear in plain English — *"something calm for studying"* — and the system finds songs whose meaning aligns with that description before ranking them by structured feature similarity.

This matters because most recommender research focuses on accuracy, but not on **explainability** or **accessibility**. VibeFinder is fully transparent: every recommendation includes a breakdown of exactly why a song was chosen. Adding a natural language front-end means users no longer need to understand internal parameters like `target_energy=0.35` to get useful results.

---

## Architecture Overview

The system has three layers. See [system_diagram.md](system_diagram.md) for the full Mermaid diagram.

```
Natural Language Input
        │
        ▼
  Text Encoder (sentence-transformers all-MiniLM-L6-v2)
        │  encodes query into a dense vector
        ▼
  Cosine Similarity Search  ◄──── Pre-encoded Song Embeddings (from songs.csv)
        │  retrieves semantically closest songs
        ▼
  Content-Based Scorer (recommender.py)
        │  applies weighted rules: genre, mood, energy, acousticness
        ▼
  Top-K Results with Explanations
        │
        ▼
  Human Review / pytest Unit Tests
```

**RAG component:** The song catalog is the knowledge base. When a user submits a query, the system retrieves the most semantically relevant song descriptions before scoring — the AI is not answering from training knowledge alone, it is retrieving grounded context first.

**Scoring rules (applied after retrieval):**

| Signal | Points |
|---|---|
| Genre match | +1.8 |
| Mood match | +1.4 |
| Energy closeness | up to +2.0 |
| Acoustic preference | +0.6 |

Tie-breaks: energy distance first, then danceability.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows
   ```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Interactions

### Example 1 — Chill Lofi Profile

**Input:**
```python
genre="lofi", mood="chill", energy=0.35, likes_acoustic=True
```

**Output:**
```
=== Top Recommendations ===
Profile Name: Chill Lofi
Profile: genre=lofi | mood=chill | energy=0.35 | likes_acoustic=True

1. Library Rain — Paper Lanterns
   Score: 5.80
   Reasons:
   - genre match (+1.8)
   - mood match (+1.4)
   - energy closeness (+2.00)
   - acoustic preference bonus (+0.6)

2. Midnight Coding — LoRoom
   Score: 5.61
   Reasons:
   - genre match (+1.8)
   - mood match (+1.4)
   - energy closeness (+1.81)
   - acoustic preference bonus (+0.6)

3. Focus Flow — LoRoom
   Score: 4.26
   Reasons:
   - genre match (+1.8)
   - energy closeness (+1.86)
   - acoustic preference bonus (+0.6)

4. Spacewalk Thoughts — Orbit Bloom
   Score: 3.81
   Reasons:
   - mood match (+1.4)
   - energy closeness (+1.81)
   - acoustic preference bonus (+0.6)

5. Coffee Shop Stories — Slow Stereo
   Score: 2.55
   Reasons:
   - energy closeness (+1.95)
   - acoustic preference bonus (+0.6)
```

**Why this makes sense:** Library Rain scores a perfect energy match (0.35 == 0.35), full genre and mood points, and an acoustic bonus — the maximum possible for this profile. Focus Flow earns genre points despite a mood mismatch because it is still a lofi track close to the energy target.

---

### Example 2 — High-Energy Pop Profile

**Input:**
```python
genre="pop", mood="happy", energy=0.88, likes_acoustic=False
```

**Output:**
```
=== Top Recommendations ===
Profile Name: High-Energy Pop
Profile: genre=pop | mood=happy | energy=0.88 | likes_acoustic=False

1. Sunrise City — Neon Echo
   Score: 5.04
   Reasons:
   - genre match (+1.8)
   - mood match (+1.4)
   - energy closeness (+1.84)

2. Gym Hero — Max Pulse
   Score: 3.66
   Reasons:
   - genre match (+1.8)
   - energy closeness (+1.86)

3. Rooftop Lights — Indigo Parade
   Score: 3.07
   Reasons:
   - mood match (+1.4)
   - energy closeness (+1.67)

4. Storm Runner — Voltline
   Score: 1.92
   Reasons:
   - energy closeness (+1.92)

5. Neon Pulse — Vector Drift
   Score: 1.89
   Reasons:
   - energy closeness (+1.89)
```

**Why this makes sense:** Sunrise City is the only pop/happy song in the catalog — it earns all three primary bonuses. Gym Hero stays in the top 2 because its genre and near-perfect energy outweigh its mood mismatch, a pattern documented in `reflection.md`.

---

### Example 3 — Adversarial: Unknown Genre + Very Low Energy

**Input:**
```python
genre="k-pop", mood="chill", energy=0.05, likes_acoustic=True
```

**Output:**
```
=== Top Recommendations ===
Profile Name: Adversarial: Unknown Genre + Chill + Very Low Energy
Profile: genre=k-pop | mood=chill | energy=0.05 | likes_acoustic=True

1. Spacewalk Thoughts — Orbit Bloom
   Score: 3.37
   Reasons:
   - mood match (+1.4)
   - energy closeness (+1.37)
   - acoustic preference bonus (+0.6)

2. Library Rain — Paper Lanterns
   Score: 3.18
   Reasons:
   - mood match (+1.4)
   - energy closeness (+1.18)
   - acoustic preference bonus (+0.6)

3. Midnight Coding — LoRoom
   Score: 2.99
   Reasons:
   - mood match (+1.4)
   - energy closeness (+0.99)
   - acoustic preference bonus (+0.6)

4. Winter Sonata — Aurora Quartet
   Score: 2.13
   Reasons:
   - energy closeness (+1.53)
   - acoustic preference bonus (+0.6)

5. Mountain Letters — Pine Harbor
   Score: 1.83
   Reasons:
   - energy closeness (+1.23)
   - acoustic preference bonus (+0.6)
```

**Why this makes sense:** Since "k-pop" does not exist in the catalog, no song earns genre points. The system falls back on mood and energy as primary signals. The result is a coherent ambient/acoustic playlist — not a crash or an error — which validates that the scorer degrades gracefully on out-of-distribution input.

---

## Design Decisions

**Why content-based scoring instead of collaborative filtering?**
Collaborative filtering requires user interaction history (plays, skips, likes). With 18 songs and no historical data, it would overfit to nothing. Content-based scoring is fully deterministic and explainable with zero training data.

**Why sentence-transformers for RAG instead of an LLM API?**
An LLM API (like Claude or GPT) would require a paid account and external network access, making the project hard to reproduce. `all-MiniLM-L6-v2` from `sentence-transformers` runs locally, downloads automatically on first use (~90MB), and produces high-quality semantic embeddings for short text. There is no API key and no per-call cost.

**Why expose per-song explanations instead of a single summary?**
Transparency is a core goal. A single summary hides which features drove a recommendation. Per-song breakdowns let users (and evaluators) verify that the scoring logic is behaving correctly — this is especially important for identifying bias.

**Trade-offs made:**
- The catalog is small (18 songs). Results are illustrative, not production-grade.
- Fixed weights (genre: 1.8, mood: 1.4, energy: 2.0) are tuned manually. A real system would learn these from feedback.
- The RAG retrieval step narrows the candidate set semantically, but the final ranking is still driven by the rule-based scorer — the two layers are not jointly optimized.

---

## Testing Summary

**Automated tests** (`tests/test_recommender.py`) verify that:
- `recommend_songs()` returns results sorted by score in descending order.
- The song with the best genre/mood/energy match ranks first.
- `explain_recommendation()` returns a non-empty string.

Run with `pytest` — all tests pass on the current implementation.

**Manual profile testing** (`src/main.py`) runs five structured profiles covering normal use, edge cases, and adversarial input:

| Profile | What it tests |
|---|---|
| High-Energy Pop | Best-case: full genre + mood + energy alignment |
| Chill Lofi | Acoustic bonus pathway; lofi genre specificity |
| Deep Intense Rock | High-energy non-pop genre; mood rarity |
| Conflicted High-Energy Sad | Mood mismatch with strong genre + energy signal |
| Unknown Genre + Chill + Very Low Energy | Out-of-distribution genre; graceful degradation |

**What worked:** The scorer is directionally valid — changing preferences consistently changes the output in the expected direction. Per-song explanations made it easy to audit results manually.

**What didn't:** The small catalog creates a ceiling effect. For some profiles, positions 3–5 are filled by songs with no genre or mood match, scored purely on energy. In a larger catalog this would not be visible.

**What was learned:** Small weight changes cause larger reorderings than expected. Raising the energy weight by 0.5 moved songs up or down by 2–3 positions. This sensitivity suggests that learned weights (from user feedback) would significantly outperform hand-tuned ones.

---

## Reflection

Building VibeFinder clarified something that is easy to miss when using commercial recommenders: the model is not discovering taste, it is measuring a distance defined entirely by whoever chose the features and weights. Every design choice — which features to score, how much weight to assign, what counts as an acoustic song — encodes assumptions about what music listeners care about. Those assumptions can exclude whole categories of users whose preferences do not map cleanly onto the chosen feature space.

Adding the RAG layer reinforced a different lesson: retrieval and generation are more useful when they stay coupled to grounded data. Rather than letting a language model hallucinate a playlist, the system retrieves semantically plausible candidates first and then applies transparent scoring. The result is easier to audit, easier to debug, and easier to trust — qualities that matter as much in a student project as they do in a production system.
