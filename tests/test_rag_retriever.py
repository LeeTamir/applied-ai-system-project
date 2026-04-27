"""Unit tests for src/rag_retriever.py

Pure-logic tests (song_to_text, _detect_genre, _detect_features,
_compute_feature_score) run with no model loaded.

RAGRetriever.retrieve / hybrid_retrieve tests inject a mock model with
identity-matrix embeddings. sentence_transformers.util.cos_sim is called
for real (the package is installed), so no network or GPU is needed — only
the encoder is mocked.
"""

import pytest
import torch
from unittest.mock import MagicMock

from src.rag_retriever import (
    song_to_text,
    RAGRetriever,
    GENRE_KEYWORDS,
    FEATURE_SIGNALS,
)


# ---------------------------------------------------------------------------
# Shared song fixtures
# ---------------------------------------------------------------------------

SONG_HIGH_ENERGY = {
    "id": 1, "title": "Power Track", "artist": "Artist A",
    "genre": "pop", "mood": "intense",
    "energy": 0.92, "tempo_bpm": 140, "valence": 0.70,
    "danceability": 0.85, "acousticness": 0.05,
}
SONG_LOW_ENERGY = {
    "id": 2, "title": "Quiet Rain", "artist": "Artist B",
    "genre": "lofi", "mood": "chill",
    "energy": 0.28, "tempo_bpm": 72, "valence": 0.60,
    "danceability": 0.50, "acousticness": 0.88,
}
SONG_HAPPY_RNB = {
    "id": 3, "title": "Smooth Groove", "artist": "Artist C",
    "genre": "rnb", "mood": "happy",
    "energy": 0.60, "tempo_bpm": 95, "valence": 0.82,
    "danceability": 0.78, "acousticness": 0.40,
}
SONG_HAPPY_POP = {
    "id": 4, "title": "Sunshine Bop", "artist": "Artist D",
    "genre": "pop", "mood": "happy",
    "energy": 0.80, "tempo_bpm": 118, "valence": 0.88,
    "danceability": 0.82, "acousticness": 0.15,
}

ALL_SONGS = [SONG_HIGH_ENERGY, SONG_LOW_ENERGY, SONG_HAPPY_RNB, SONG_HAPPY_POP]


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_retriever() -> RAGRetriever:
    """Bare retriever with no model — safe for pure-logic method tests."""
    r = object.__new__(RAGRetriever)
    r._model = None
    r._embeddings = None
    r._songs = []
    return r


def _make_retriever_indexed(songs=None) -> RAGRetriever:
    """
    Retriever with identity-matrix embeddings and a mock model.
    Each song gets a unique orthogonal unit vector, so cosine similarity
    between song i's embedding and song j's embedding is 1 if i==j else 0.
    """
    if songs is None:
        songs = ALL_SONGS
    n = len(songs)
    embeddings = torch.eye(n)

    mock_model = MagicMock()
    mock_model.encode.return_value = embeddings[0]

    r = object.__new__(RAGRetriever)
    r._SentenceTransformer = None
    r._model = mock_model
    r._songs = list(songs)
    r._embeddings = embeddings
    return r


def _retrieve(r: RAGRetriever, query: str, top_n: int = 4, query_match_idx: int = 0):
    """Call r.retrieve() with the encoder mocked to return embeddings[query_match_idx].
    sentence_transformers.util.cos_sim is called for real against the torch tensors.
    """
    r._model.encode.return_value = r._embeddings[query_match_idx]
    return r.retrieve(query, top_n=top_n)


def _hybrid(r: RAGRetriever, query: str, top_n: int = 4, query_match_idx: int = 0):
    """Call r.hybrid_retrieve() with the encoder mocked to return embeddings[query_match_idx]."""
    r._model.encode.return_value = r._embeddings[query_match_idx]
    return r.hybrid_retrieve(query, top_n=top_n)


# ---------------------------------------------------------------------------
# song_to_text
# ---------------------------------------------------------------------------

class TestSongToText:
    def test_contains_title_and_artist(self):
        text = song_to_text(SONG_HIGH_ENERGY)
        assert "Power Track" in text
        assert "Artist A" in text

    def test_very_high_energy_label(self):
        assert "very high energy" in song_to_text({**SONG_HIGH_ENERGY, "energy": 0.90})

    def test_high_energy_label(self):
        assert "high energy" in song_to_text({**SONG_HIGH_ENERGY, "energy": 0.72})

    def test_moderate_energy_label(self):
        assert "moderate energy" in song_to_text({**SONG_HIGH_ENERGY, "energy": 0.60})

    def test_low_energy_label(self):
        assert "low energy" in song_to_text({**SONG_LOW_ENERGY, "energy": 0.40})

    def test_very_low_energy_label(self):
        assert "very low energy" in song_to_text({**SONG_LOW_ENERGY, "energy": 0.20})

    def test_acoustic_phrase_added_when_high_acousticness(self):
        text = song_to_text({**SONG_LOW_ENERGY, "acousticness": 0.80})
        assert " acoustic" in text

    def test_no_acoustic_phrase_when_low_acousticness(self):
        text = song_to_text({**SONG_HIGH_ENERGY, "acousticness": 0.10})
        assert " acoustic" not in text

    def test_genre_in_output(self):
        text = song_to_text(SONG_HIGH_ENERGY)
        assert "pop" in text

    def test_mood_in_output(self):
        text = song_to_text(SONG_HIGH_ENERGY)
        # mood="intense" → "intense and powerful"
        assert "intense" in text

    def test_boundary_energy_085_is_very_high(self):
        assert "very high energy" in song_to_text({**SONG_HIGH_ENERGY, "energy": 0.85})

    def test_boundary_energy_070_is_high(self):
        assert "high energy" in song_to_text({**SONG_HIGH_ENERGY, "energy": 0.70})


# ---------------------------------------------------------------------------
# _detect_genre
# ---------------------------------------------------------------------------

class TestDetectGenre:
    def setup_method(self):
        self.r = _make_retriever()

    def test_simple_genre(self):
        assert self.r._detect_genre("give me a jazz song") == "jazz"

    def test_longer_phrase_wins_over_shorter(self):
        assert self.r._detect_genre("I want indie pop music") == "indie pop"

    def test_lofi_alias_hyphen(self):
        assert self.r._detect_genre("a lo-fi track") == "lofi"

    def test_lofi_alias_space(self):
        assert self.r._detect_genre("lo fi beats") == "lofi"

    def test_rnb_alias_ampersand(self):
        assert self.r._detect_genre("some r&b vibes") == "rnb"

    def test_rnb_alias_words(self):
        assert self.r._detect_genre("r and b songs") == "rnb"

    def test_rap_maps_to_hip_hop(self):
        assert self.r._detect_genre("rap music") == "hip hop"

    def test_hip_hop_direct(self):
        assert self.r._detect_genre("hip hop beats") == "hip hop"

    def test_soul_maps_to_rnb(self):
        assert self.r._detect_genre("soul music") == "rnb"

    def test_heavy_metal_maps_to_metal(self):
        assert self.r._detect_genre("heavy metal riff") == "metal"

    def test_indie_without_pop_maps_to_indie_pop(self):
        assert self.r._detect_genre("indie track") == "indie pop"

    def test_no_match_returns_none(self):
        assert self.r._detect_genre("something with high energy") is None

    def test_case_insensitive(self):
        assert self.r._detect_genre("A great ROCK anthem") == "rock"

    def test_edm_direct(self):
        assert self.r._detect_genre("edm festival bangers") == "edm"

    def test_electronic_dance_music(self):
        assert self.r._detect_genre("electronic dance music only") == "edm"


# ---------------------------------------------------------------------------
# _detect_features
# ---------------------------------------------------------------------------

class TestDetectFeatures:
    def setup_method(self):
        self.r = _make_retriever()

    def test_positive_energy(self):
        assert self.r._detect_features("something energetic for the gym")["energy"] == 1

    def test_negative_energy_calm(self):
        assert self.r._detect_features("calm study music")["energy"] == -1

    def test_negative_energy_rainy(self):
        assert self.r._detect_features("rainy day background")["energy"] == -1

    def test_low_energy_phrase_triggers_negative_not_positive(self):
        # "energy" alone is a positive keyword, but "low energy" must win
        assert self.r._detect_features("a song with low energy")["energy"] == -1

    def test_positive_valence_happy(self):
        assert self.r._detect_features("happy uplifting songs")["valence"] == 1

    def test_negative_valence_sad(self):
        assert self.r._detect_features("sad melancholic music")["valence"] == -1

    def test_positive_acousticness(self):
        assert self.r._detect_features("acoustic folk guitar")["acousticness"] == 1

    def test_negative_acousticness_electronic(self):
        assert self.r._detect_features("electronic synth music")["acousticness"] == -1

    def test_positive_danceability(self):
        assert self.r._detect_features("songs to dance to at a party")["danceability"] == 1

    def test_positive_tempo_high(self):
        assert self.r._detect_features("high bpm workout")["tempo_bpm"] == 1

    def test_negative_tempo_low(self):
        assert self.r._detect_features("slow bpm downtempo")["tempo_bpm"] == -1

    def test_no_match_returns_empty(self):
        assert self.r._detect_features("I just want something nice") == {}

    def test_multiple_features_detected(self):
        result = self.r._detect_features("energetic dance music")
        assert result.get("energy") == 1
        assert result.get("danceability") == 1

    def test_genre_keywords_not_in_features(self):
        # Genre detection is separate — "jazz" should not appear in feature dict
        result = self.r._detect_features("a jazz song")
        assert "genre" not in result


# ---------------------------------------------------------------------------
# _compute_feature_score
# ---------------------------------------------------------------------------

class TestComputeFeatureScore:
    def setup_method(self):
        self.r = _make_retriever()

    def test_positive_direction_high_value_scores_high(self):
        score = self.r._compute_feature_score(SONG_HIGH_ENERGY, {"energy": 1})
        assert score > 0.85

    def test_negative_direction_high_value_scores_low(self):
        score = self.r._compute_feature_score(SONG_HIGH_ENERGY, {"energy": -1})
        assert score < 0.15

    def test_positive_direction_low_value_scores_low(self):
        score = self.r._compute_feature_score(SONG_LOW_ENERGY, {"energy": 1})
        assert score < 0.35

    def test_negative_direction_low_value_scores_high(self):
        score = self.r._compute_feature_score(SONG_LOW_ENERGY, {"energy": -1})
        assert score > 0.65

    def test_genre_match_gives_full_score(self):
        score = self.r._compute_feature_score(SONG_HIGH_ENERGY, {}, detected_genre="pop")
        assert score == pytest.approx(1.0)

    def test_genre_mismatch_gives_zero(self):
        score = self.r._compute_feature_score(SONG_HIGH_ENERGY, {}, detected_genre="jazz")
        assert score == pytest.approx(0.0)

    def test_tempo_bpm_normalized_correctly(self):
        # tempo_bpm=140, normalize=(60, 180) → (140-60)/(180-60) = 80/120
        song = {**SONG_HIGH_ENERGY, "tempo_bpm": 140}
        score = self.r._compute_feature_score(song, {"tempo_bpm": 1})
        assert score == pytest.approx(80 / 120, abs=1e-3)

    def test_tempo_bpm_clamped_at_upper_bound(self):
        song = {**SONG_HIGH_ENERGY, "tempo_bpm": 200}
        score = self.r._compute_feature_score(song, {"tempo_bpm": 1})
        assert score == pytest.approx(1.0)

    def test_tempo_bpm_clamped_at_lower_bound(self):
        song = {**SONG_HIGH_ENERGY, "tempo_bpm": 30}
        score = self.r._compute_feature_score(song, {"tempo_bpm": 1})
        assert score == pytest.approx(0.0)

    def test_no_signals_returns_zero(self):
        score = self.r._compute_feature_score(SONG_HIGH_ENERGY, {})
        assert score == 0.0

    def test_feature_and_genre_averaged(self):
        # energy=0.92 direction +1 → 0.92, genre pop matches → 1.0 → avg = 0.96
        score = self.r._compute_feature_score(
            SONG_HIGH_ENERGY, {"energy": 1}, detected_genre="pop"
        )
        assert score == pytest.approx((0.92 + 1.0) / 2, abs=1e-3)


# ---------------------------------------------------------------------------
# RAGRetriever.retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_raises_before_index(self):
        r = _make_retriever()
        with pytest.raises(RuntimeError, match="index_songs"):
            r.retrieve("any query")

    def test_raises_on_empty_query(self):
        r = _make_retriever_indexed()
        with pytest.raises(ValueError, match="non-empty"):
            r.retrieve("")

    def test_raises_on_whitespace_query(self):
        r = _make_retriever_indexed()
        with pytest.raises(ValueError):
            r.retrieve("   ")

    def test_raises_on_top_n_less_than_1(self):
        r = _make_retriever_indexed()
        with pytest.raises(ValueError):
            _retrieve(r, "happy music", top_n=0)

    def test_returns_top_n_results(self):
        r = _make_retriever_indexed()
        results = _retrieve(r, "test query", top_n=2)
        assert len(results) == 2

    def test_best_match_returned_first(self):
        r = _make_retriever_indexed()
        # query_match_idx=1 → SONG_LOW_ENERGY should score highest
        results = _retrieve(r, "test query", top_n=1, query_match_idx=1)
        assert results[0][0]["title"] == SONG_LOW_ENERGY["title"]

    def test_scores_are_descending(self):
        r = _make_retriever_indexed()
        results = _retrieve(r, "test query", top_n=len(ALL_SONGS), query_match_idx=0)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_tuples_are_song_and_float(self):
        r = _make_retriever_indexed()
        results = _retrieve(r, "test query", top_n=1)
        song, score = results[0]
        assert isinstance(song, dict)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# RAGRetriever.hybrid_retrieve
# ---------------------------------------------------------------------------

class TestHybridRetrieve:
    def test_raises_before_index(self):
        r = _make_retriever()
        with pytest.raises(RuntimeError):
            r.hybrid_retrieve("any query")

    def test_raises_on_empty_query(self):
        r = _make_retriever_indexed()
        with pytest.raises(ValueError):
            _hybrid(r, "")

    def test_returns_top_n(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "chill music", top_n=2)
        assert len(results) == 2

    def test_results_sorted_descending_by_hybrid_score(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "energetic pop workout", top_n=len(ALL_SONGS))
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_details_dict_has_required_keys(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "happy dance pop", top_n=2)
        for _, _, details in results:
            assert {"semantic", "feature", "hybrid", "detected"} <= details.keys()

    def test_genre_appears_in_detected(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "a happy pop song", top_n=2)
        detected = results[0][2]["detected"]
        assert "genre:pop" in detected

    def test_energy_signal_detected_positive(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "energetic workout music", top_n=2)
        detected = results[0][2]["detected"]
        assert "energy" in detected

    def test_energy_signal_detected_negative(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "calm study music", top_n=2)
        detected = results[0][2]["detected"]
        assert "energy" in detected

    def test_no_signals_hybrid_equals_semantic(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "music please", top_n=1, query_match_idx=0)
        details = results[0][2]
        # No feature or genre signals → hybrid_score should equal semantic_score
        assert details["hybrid"] == pytest.approx(details["semantic"], abs=1e-3)

    def test_result_tuples_have_three_elements(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "chill lofi", top_n=1)
        assert len(results[0]) == 3

    def test_high_energy_song_ranked_higher_for_energetic_query(self):
        r = _make_retriever_indexed()
        # For an "energetic" query with equal semantic scores (all 0 except match),
        # feature scoring should boost the high-energy song
        results = _hybrid(r, "energetic workout pump", top_n=len(ALL_SONGS), query_match_idx=2)
        titles = [song["title"] for song, _, _ in results]
        high_energy_pos = titles.index(SONG_HIGH_ENERGY["title"])
        low_energy_pos = titles.index(SONG_LOW_ENERGY["title"])
        assert high_energy_pos < low_energy_pos

    def test_rnb_genre_boosted_for_rnb_query(self):
        r = _make_retriever_indexed()
        results = _hybrid(r, "a happy rnb song", top_n=len(ALL_SONGS), query_match_idx=3)
        titles = [song["title"] for song, _, _ in results]
        rnb_pos = titles.index(SONG_HAPPY_RNB["title"])
        # SONG_HAPPY_RNB is the only rnb song — it should outrank pop songs
        # (semantic score for all non-matched is 0, so feature score decides)
        non_rnb_positions = [i for i, s in enumerate(titles) if s != SONG_HAPPY_RNB["title"]]
        assert rnb_pos < max(non_rnb_positions)
