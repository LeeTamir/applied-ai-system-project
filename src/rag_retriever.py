"""
RAG retriever for VibeFinder.

Encodes the song catalog as dense embedding vectors using sentence-transformers
and retrieves semantically similar songs given a natural language query.
This is the Retrieval-Augmented Generation layer: the model looks up grounded
song data before producing recommendations, rather than relying on training
knowledge alone.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# Maps query keywords to CSV feature columns and the direction that is "better"
# direction=1 means higher CSV value is better, direction=-1 means lower is better
FEATURE_SIGNALS: Dict[str, Dict] = {
    "energy": {
        "positive": ["energetic", "energy", "pump", "workout", "intense", "powerful",
                     "fast", "hype", "loud", "upbeat", "driven", "fire", "high energy"],
        "negative": ["low energy", "low-energy", "no energy", "not energetic",
                     "calm", "chill", "relax", "relaxing", "soft", "quiet", "study",
                     "sleep", "gentle", "slow", "peaceful", "mellow", "ambient", "rainy"],
    },
    "acousticness": {
        "positive": ["acoustic", "unplugged", "natural", "organic", "raw", "stripped", "folk"],
        "negative": ["electronic", "synth", "edm", "digital", "electric"],
    },
    "valence": {
        "positive": ["happy", "positive", "uplifting", "joyful", "cheerful", "fun",
                     "bright", "feel good", "feel-good"],
        "negative": ["sad", "dark", "melancholic", "moody", "gloomy", "depressing",
                     "emotional", "heavy", "heartbreak"],
    },
    "danceability": {
        "positive": ["dance", "danceable", "groove", "beat", "club", "party", "bop"],
        "negative": [],
    },
    "tempo_bpm": {
        "positive": ["high bpm", "fast bpm", "high tempo", "fast tempo", "uptempo"],
        "negative": ["low bpm", "slow bpm", "low tempo", "slow tempo", "downtempo"],
        "normalize": (60, 180),  # clamp to [0, 1] before scoring
    },
}

# Maps query terms to genre values as they appear in songs.csv.
# Longer phrases are listed first so _detect_genre() checks them before
# shorter substrings (e.g. "indie pop" before "pop").
GENRE_KEYWORDS: Dict[str, str] = {
    "indie pop":      "indie pop",
    "heavy metal":    "metal",
    "hip hop":        "hip hop",
    "electronic dance music": "edm",
    "r&b":            "rnb",
    "r and b":        "rnb",
    "lo-fi":          "lofi",
    "lo fi":          "lofi",
    "synthwave":      "synthwave",
    "classical":      "classical",
    "ambient":        "ambient",
    "country":        "country",
    "reggae":         "reggae",
    "hiphop":         "hip hop",
    "lofi":           "lofi",
    "metal":          "metal",
    "folk":           "folk",
    "jazz":           "jazz",
    "rock":           "rock",
    "pop":            "pop",
    "edm":            "edm",
    "rnb":            "rnb",
    "soul":           "rnb",
    "rap":            "hip hop",
    "indie":          "indie pop",
}


_MOOD_PHRASES: Dict[str, str] = {
    "happy":      "upbeat and joyful",
    "chill":      "calm and relaxed",
    "intense":    "intense and powerful",
    "sad":        "emotional and melancholic",
    "focused":    "focused and steady",
    "melancholic":"melancholic and introspective",
    "moody":      "moody and atmospheric",
    "aggressive": "aggressive and heavy",
    "uplifting":  "uplifting and positive",
    "nostalgic":  "nostalgic and warm",
    "reflective": "reflective and thoughtful",
    "romantic":   "romantic and smooth",
    "euphoric":   "euphoric and exciting",
    "confident":  "confident and driven",
    "relaxed":    "relaxed and easygoing",
}

_GENRE_PHRASES: Dict[str, str] = {
    "pop":       "pop",
    "lofi":      "lo-fi",
    "rock":      "rock",
    "ambient":   "ambient",
    "jazz":      "jazz",
    "synthwave": "synthwave",
    "indie pop": "indie pop",
    "hip hop":   "hip hop",
    "classical": "classical",
    "reggae":    "reggae",
    "metal":     "metal",
    "country":   "country",
    "edm":       "electronic dance music",
    "rnb":       "R&B",
    "folk":      "folk",
}

_ENERGY_CONTEXT: Dict[str, str] = {
    "very high": "great for workouts, pump-up sessions, or high-energy activities",
    "high":      "good for active listening, dancing, or exercising",
    "medium":    "suitable for casual listening or background music",
    "low":       "ideal for studying, focusing, or winding down",
    "very low":  "perfect for deep relaxation, meditation, or rainy day listening",
}


def song_to_text(song: Dict) -> str:
    """
    Convert a song dict to a full natural language sentence for embedding.
    Richer descriptions produce better cosine similarity scores against
    natural language queries.
    """
    title  = song["title"]
    artist = song["artist"]
    genre  = _GENRE_PHRASES.get(song["genre"], song["genre"])
    mood   = _MOOD_PHRASES.get(song["mood"], song["mood"])
    energy = float(song["energy"])
    acoustic = float(song.get("acousticness", 0)) >= 0.65

    if energy >= 0.85:
        elabel, ekey = "very high energy", "very high"
    elif energy >= 0.70:
        elabel, ekey = "high energy", "high"
    elif energy >= 0.55:
        elabel, ekey = "moderate energy", "medium"
    elif energy >= 0.35:
        elabel, ekey = "low energy", "low"
    else:
        elabel, ekey = "very low energy", "very low"

    acoustic_phrase = " acoustic" if acoustic else ""
    context = _ENERGY_CONTEXT[ekey]

    return (
        f"{title} by {artist} is a {mood}{acoustic_phrase} {genre} track "
        f"with {elabel}. {context}."
    )


class RAGRetriever:
    """
    Retrieval-Augmented Generation component for VibeFinder.

    Encodes the song catalog into dense vectors on startup, then finds the
    most semantically similar songs to any natural language query using
    cosine similarity.
    """

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._SentenceTransformer = SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        logger.info("Loading sentence-transformers model: %s", MODEL_NAME)
        self._model = self._SentenceTransformer(MODEL_NAME)
        self._embeddings = None
        self._songs: List[Dict] = []
        logger.info("Model ready.")

    def index_songs(self, songs: List[Dict]) -> None:
        """Pre-encode all songs into embedding vectors. Must be called before retrieve()."""
        if not songs:
            logger.warning("index_songs() called with empty song list.")
            return

        self._songs = songs
        texts = [song_to_text(s) for s in songs]
        logger.info("Encoding %d songs into embeddings...", len(texts))
        self._embeddings = self._model.encode(texts, convert_to_tensor=True)
        logger.info("Embeddings built for %d songs.", len(texts))

    def retrieve(self, query: str, top_n: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find the top-n songs most semantically similar to the natural language query.

        Returns a list of (song_dict, similarity_score) tuples sorted by
        similarity descending. Similarity is cosine similarity in [0, 1].

        Raises RuntimeError if index_songs() has not been called first.
        Raises ValueError if query is empty or top_n < 1.
        """
        if self._embeddings is None:
            raise RuntimeError(
                "Song index is empty. Call index_songs() before retrieve()."
            )
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}.")

        logger.info("RAG retrieve | query='%s' top_n=%d", query.strip(), top_n)

        from sentence_transformers import util

        query_embedding = self._model.encode(query.strip(), convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self._embeddings)[0]
        top_indices = scores.argsort(descending=True)[:top_n]

        results: List[Tuple[Dict, float]] = []
        for idx in top_indices:
            song = self._songs[int(idx)]
            similarity = float(scores[idx])
            logger.debug("  match: %-30s similarity=%.4f", song["title"], similarity)
            results.append((song, similarity))

        return results

    def _detect_genre(self, query: str) -> Optional[str]:
        """
        Return the CSV genre string matched in the query, or None.
        Longer phrases are checked first to prevent 'pop' matching inside 'indie pop'.
        """
        query_lower = query.lower()
        for keyword in sorted(GENRE_KEYWORDS, key=len, reverse=True):
            if keyword in query_lower:
                return GENRE_KEYWORDS[keyword]
        return None

    def _detect_features(self, query: str) -> Dict[str, int]:
        """
        Scan the query for keywords that imply a specific CSV feature direction.
        Returns {feature_name: direction} where direction is +1 (higher is better)
        or -1 (lower is better).
        """
        query_lower = query.lower()
        detected: Dict[str, int] = {}
        for feature, signals in FEATURE_SIGNALS.items():
            # Check negative before positive so phrases like "low energy" don't
            # accidentally match the shorter positive keyword "energy".
            if any(kw in query_lower for kw in signals["negative"]):
                detected[feature] = -1
            elif any(kw in query_lower for kw in signals["positive"]):
                detected[feature] = 1
        return detected

    def _compute_feature_score(
        self,
        song: Dict,
        feature_directions: Dict[str, int],
        detected_genre: Optional[str] = None,
    ) -> float:
        """
        Compute a 0-1 score from the song's CSV values.

        Numeric features use the direction signal (+1/-1) with optional
        normalization. Genre is scored as a binary match (1.0 / 0.0).
        All active signals are averaged.
        """
        scores = []
        for feature, direction in feature_directions.items():
            value = float(song.get(feature, 0.5))
            norm = FEATURE_SIGNALS.get(feature, {}).get("normalize")
            if norm:
                lo, hi = norm
                value = max(0.0, min(1.0, (value - lo) / (hi - lo)))
            scores.append(value if direction > 0 else 1.0 - value)

        if detected_genre is not None:
            scores.append(1.0 if song.get("genre") == detected_genre else 0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def hybrid_retrieve(
        self, query: str, top_n: int = 5, semantic_weight: float = 0.5
    ) -> List[Tuple[Dict, float, Dict]]:
        """
        Hybrid retrieval: combines semantic similarity with numeric CSV feature scores.

        1. Ranks all songs by semantic similarity (RAG step).
        2. Detects which CSV features the query implies via keyword matching.
        3. Re-ranks using:
               hybrid_score = semantic_weight * similarity
                            + (1 - semantic_weight) * feature_score

        Falls back to pure semantic ranking when no feature signals are detected.

        Returns list of (song_dict, hybrid_score, details_dict) sorted descending.
        details_dict keys: "semantic", "feature", "hybrid", "detected".
        """
        if self._embeddings is None:
            raise RuntimeError(
                "Song index is empty. Call index_songs() before hybrid_retrieve()."
            )
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}.")

        logger.info("Hybrid retrieve | query='%s' top_n=%d", query.strip(), top_n)

        all_results = self.retrieve(query, top_n=len(self._songs))

        feature_directions = self._detect_features(query)
        detected_genre = self._detect_genre(query)
        has_signals = bool(feature_directions) or detected_genre is not None
        feature_weight = 1.0 - semantic_weight

        if has_signals:
            detected_labels = list(feature_directions.keys())
            if detected_genre:
                detected_labels.append(f"genre:{detected_genre}")
            logger.info("  Detected signals: %s", detected_labels)
        else:
            logger.info("  No signals detected — using pure semantic ranking.")

        re_ranked: List[Tuple[Dict, float, Dict]] = []
        for song, similarity in all_results:
            feature_score = self._compute_feature_score(
                song, feature_directions, detected_genre
            )

            hybrid_score = (
                semantic_weight * similarity + feature_weight * feature_score
                if has_signals
                else similarity
            )

            detected_labels = list(feature_directions.keys())
            if detected_genre:
                detected_labels.append(f"genre:{detected_genre}")
            details = {
                "semantic": round(similarity, 3),
                "feature": round(feature_score, 3),
                "hybrid": round(hybrid_score, 3),
                "detected": detected_labels,
            }
            logger.debug(
                "  %-30s semantic=%.3f feature=%.3f hybrid=%.3f",
                song["title"], similarity, feature_score, hybrid_score,
            )
            re_ranked.append((song, hybrid_score, details))

        re_ranked.sort(key=lambda x: x[1], reverse=True)
        return re_ranked[:top_n]
