# Model Card: VibeFinder

**System:** VibeFinder — RAG-enhanced music recommender
**Model used:** `multi-qa-MiniLM-L6-cos-v1` (sentence-transformers)
**Catalog:** 50 songs, 10 features each (`data/songs.csv`)

---

## What Are the Limitations or Biases in Your System?

**Catalog scope.** The 50-song catalog was hand-authored, not drawn from real listening data. Genre and mood distributions reflect the author's choices, not actual music consumption patterns. Some genres have only 2–3 entries (reggae, classical), which means queries in those genres have very few candidates to compete against and will surface those songs even if the semantic match is weak.

**Western genre taxonomy.** The genre list (pop, rock, hip hop, edm, etc.) is centered on genres common in Western markets. Genres like K-pop, afrobeats, bossa nova, or cumbia have no representation. A query like "some afrobeats" returns nothing meaningful because neither the catalog nor the keyword detection knows those genres exist.

**Keyword detection is brittle.** Feature signal detection is a keyword scan, not natural language understanding. It does not handle negation — "not energetic" will fail to flip the `energy` direction because "energetic" is detected as a positive signal before "not" is considered. Similarly, sarcasm, slang, or indirect phrasing ("something that won't keep me awake") is invisible to the scanner.

**Fixed feature weights.** The structured scorer uses hand-tuned weights (genre: 1.8, mood: 1.4, energy: up to 2.0). These were set by intuition, not learned from user feedback. A different weighting could produce equally defensible rankings. Users whose taste is energy-dominant or mood-dominant may consistently get suboptimal results at the extremes.

**Acoustic threshold is a hard cut.** A song is labeled "acoustic" only if `acousticness >= 0.65`. A song at 0.64 gets no acoustic credit. This is an artifact of the encoding step, not a meaningful musical distinction.

**Semantic model language bias.** `multi-qa-MiniLM-L6-cos-v1` was trained primarily on English text. Queries in other languages or heavy dialect will produce degraded semantic scores.

**No personalization.** The system has no memory of prior queries. Every search starts from scratch. Two users with very different tastes get the same results for the same query.

---

## Could Your AI Be Misused, and How Would You Prevent That?

VibeFinder is a music discovery tool with a small, hand-authored catalog, so its direct misuse potential is low compared to systems that generate content or make high-stakes decisions. That said, some misuse vectors exist:

**Catalog manipulation.** Because the system ranks whatever is in `songs.csv`, an operator with write access to the CSV could plant songs designed to always surface first — for example, by setting `energy=1.0`, `valence=1.0`, and `danceability=1.0` simultaneously. In a production context this is a payola-style attack. Prevention: validate CSV data on load, cap feature values, and require human review before catalog updates.

**Query farming for profiling.** If query logs were retained without consent, patterns across users could reveal personal information (mood states, activity routines). Prevention: do not log raw queries; if analytics are needed, aggregate counts only.

**Misleading confidence signals.** The "Match %" displayed in the UI is a cosine similarity or hybrid score, not a true probability. A song showing 65% could mislead a user into thinking there is a 65% chance they will enjoy it. Prevention: relabel the display as "Relevance score" or add a tooltip explaining what the number represents.

The most important mitigation in VibeFinder is already in place: full transparency. Every result shows which signals were detected and how they influenced the ranking. Users can see exactly why a song appeared and evaluate the reasoning themselves rather than trusting an opaque score.

---

## What Surprised You While Testing Your AI's Reliability?

**Positive keyword matching "low energy."** The phrase "a song with low energy" was returning high-energy songs. The cause: `"energy"` was in the positive keyword list and was matched before checking the negative list, so `"low energy"` was classified as wanting *more* energy. The fix required two changes: adding multi-word negative phrases (`"low energy"`, `"low-energy"`, `"not energetic"`) and always checking the negative list before the positive list. The order of evaluation was not something I anticipated would matter at the design stage, but it turned out to be the most impactful correctness issue in the system.

**How low the initial semantic scores were.** Using `all-MiniLM-L6-v2`, the highest similarity score in the catalog was around 35%, with many results below 10%. This was surprising because the song descriptions felt semantically close to the queries. The root cause was using a symmetric sentence-similarity model for an asymmetric query-document retrieval task — the model was never trained on short query vs. long description matching. Switching to `multi-qa-MiniLM-L6-cos-v1` raised peak scores to 60–70% with no other changes.

**BPM queries returning the same song for both high and low BPM.** Before adding `tempo_bpm` to the feature signal dictionary, queries like "high bpm workout" and "slow bpm relaxing" both surfaced the same mid-tempo jazz song ("Coffee Shop Stories") because the semantic embedding for "bpm" had no numeric grounding. The fix — adding BPM as a detectable feature with a normalization range of 60–180 — immediately produced the expected split.

**Identity-matrix embeddings as a test fixture.** While writing unit tests, it was unclear how to test `retrieve()` and `hybrid_retrieve()` without loading the real model. Using a torch identity matrix as the embedding table (each song gets a unique orthogonal unit vector) turned out to be a clean solution: cosine similarity between song `i` and a query vector `e_i` is exactly 1.0, and all others are 0.0, making sort-order and boosting assertions deterministic. This was not planned in advance — it emerged while trying to avoid mocking the full sentence-transformers `util` module.

---

## Collaboration with AI During This Project

This project was built in direct collaboration with Claude (Anthropic). Claude served as a coding partner throughout: suggesting architecture decisions, writing implementation code, debugging errors, and reviewing test logic.

**One instance where the AI gave a helpful suggestion:**

When semantic similarity scores were unexpectedly low (~35% peak), Claude identified that the model choice was the cause. It explained the distinction between *symmetric* models (like `all-MiniLM-L6-v2`, trained to compare sentence pairs of similar length) and *asymmetric* models (like `multi-qa-MiniLM-L6-cos-v1`, trained specifically for short query → longer document retrieval). This was a non-obvious architectural distinction — I had assumed any sentence-transformers model would work equally well for this task. Switching models required only a one-line change but produced a substantial improvement in result quality. Claude's explanation was both technically accurate and actionable.

**One instance where the AI's suggestion was flawed:**

When writing unit tests for `hybrid_retrieve()`, Claude proposed patching `sentence_transformers.util` at the module level using `patch("src.rag_retriever.util")`. This failed with `AttributeError: module 'src.rag_retriever' does not have the attribute 'util'` because `util` is imported *inside* the `retrieve()` method body (`from sentence_transformers import util`) rather than at the module level, so it never becomes a named attribute of `src.rag_retriever`. The suggested patch target did not exist. The correct fix was to stop patching `util` altogether and let the real `sentence_transformers.util.cos_sim` run against the mock torch tensors — since the package is already installed, this worked correctly and produced simpler test code. Claude acknowledged the error and suggested the revised approach, but the initial suggestion required a debugging round to resolve.
