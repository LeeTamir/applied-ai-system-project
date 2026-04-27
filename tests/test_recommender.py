from src.recommender import Song, UserProfile, Recommender, score_song, recommend_songs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _song(**overrides) -> Song:
    defaults = dict(
        id=1, title="Test Track", artist="Artist",
        genre="pop", mood="happy",
        energy=0.8, tempo_bpm=120, valence=0.85,
        danceability=0.80, acousticness=0.20,
    )
    defaults.update(overrides)
    return Song(**defaults)


def _user(**overrides) -> UserProfile:
    defaults = dict(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


def _make_recommender() -> Recommender:
    return Recommender([
        _song(id=1, title="Pop Hit",     genre="pop",  mood="happy",  energy=0.80, acousticness=0.15),
        _song(id=2, title="Chill Lofi",  genre="lofi", mood="chill",  energy=0.35, acousticness=0.85),
        _song(id=3, title="Rock Anthem", genre="rock", mood="intense",energy=0.92, acousticness=0.05),
    ])


# ---------------------------------------------------------------------------
# score_song
# ---------------------------------------------------------------------------

class TestScoreSong:
    def test_genre_match_adds_points(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": None, "target_energy": None, "likes_acoustic": False}
        score, reasons = score_song(prefs, {"genre": "pop", "mood": "chill", "energy": 0.5, "acousticness": 0.2})
        assert score == 1.8
        assert any("genre match" in r for r in reasons)

    def test_genre_mismatch_adds_no_points(self):
        prefs = {"favorite_genre": "jazz", "favorite_mood": None, "target_energy": None, "likes_acoustic": False}
        score, _ = score_song(prefs, {"genre": "pop", "mood": "chill", "energy": 0.5, "acousticness": 0.2})
        assert score < 1.8

    def test_mood_match_adds_points(self):
        prefs = {"favorite_genre": None, "favorite_mood": "happy", "target_energy": None, "likes_acoustic": False}
        score, reasons = score_song(prefs, {"genre": "pop", "mood": "happy", "energy": 0.5, "acousticness": 0.2})
        assert score == 1.4
        assert any("mood match" in r for r in reasons)

    def test_perfect_energy_match_gives_max_points(self):
        target = 0.8
        prefs = {"favorite_genre": None, "favorite_mood": None, "target_energy": target, "likes_acoustic": False}
        score, _ = score_song(prefs, {"genre": "pop", "mood": "happy", "energy": target, "acousticness": 0.2})
        assert score == 2.0

    def test_far_energy_mismatch_gives_near_zero(self):
        prefs = {"favorite_genre": None, "favorite_mood": None, "target_energy": 0.95, "likes_acoustic": False}
        score, _ = score_song(prefs, {"genre": "pop", "mood": "happy", "energy": 0.22, "acousticness": 0.2})
        assert score < 0.1

    def test_acoustic_bonus_when_song_is_acoustic_and_user_likes_it(self):
        prefs = {"favorite_genre": None, "favorite_mood": None, "target_energy": None, "likes_acoustic": True}
        score, reasons = score_song(prefs, {"genre": "folk", "mood": "chill", "energy": 0.3, "acousticness": 0.90})
        assert score == 0.6
        assert any("acoustic" in r for r in reasons)

    def test_no_acoustic_bonus_when_user_does_not_like_acoustic(self):
        prefs = {"favorite_genre": None, "favorite_mood": None, "target_energy": None, "likes_acoustic": False}
        score, _ = score_song(prefs, {"genre": "folk", "mood": "chill", "energy": 0.3, "acousticness": 0.90})
        assert score == 0.0

    def test_no_acoustic_bonus_when_song_not_acoustic(self):
        prefs = {"favorite_genre": None, "favorite_mood": None, "target_energy": None, "likes_acoustic": True}
        score, _ = score_song(prefs, {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.20})
        assert score == 0.0

    def test_all_signals_accumulate(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
        score, _ = score_song(prefs, {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.1})
        assert score == 1.8 + 1.4 + 2.0

    def test_no_match_returns_reason(self):
        prefs = {"favorite_genre": "jazz", "favorite_mood": "sad", "target_energy": 0.5, "likes_acoustic": False}
        _, reasons = score_song(prefs, {"genre": "metal", "mood": "intense", "energy": 0.5, "acousticness": 0.0})
        # energy is equal, so there will be an energy closeness reason — just ensure reasons list is non-empty
        assert len(reasons) > 0


# ---------------------------------------------------------------------------
# recommend_songs
# ---------------------------------------------------------------------------

class TestRecommendSongs:
    def _songs(self):
        return [
            {"id": 1, "title": "A", "genre": "pop",  "mood": "happy",  "energy": 0.80, "acousticness": 0.10},
            {"id": 2, "title": "B", "genre": "rock", "mood": "intense","energy": 0.90, "acousticness": 0.05},
            {"id": 3, "title": "C", "genre": "lofi", "mood": "chill",  "energy": 0.35, "acousticness": 0.85},
        ]

    def test_returns_top_k(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
        results = recommend_songs(prefs, self._songs(), k=2)
        assert len(results) == 2

    def test_k_zero_returns_empty(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
        results = recommend_songs(prefs, self._songs(), k=0)
        assert results == []

    def test_k_larger_than_catalog_returns_all(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
        results = recommend_songs(prefs, self._songs(), k=100)
        assert len(results) == 3

    def test_sorted_descending_by_score(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
        results = recommend_songs(prefs, self._songs(), k=3)
        scores = [score for _, score, _ in results]
        assert scores == sorted(scores, reverse=True)

    def test_best_match_is_first(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
        results = recommend_songs(prefs, self._songs(), k=3)
        assert results[0][0]["genre"] == "pop"

    def test_explanation_is_non_empty_string(self):
        prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
        results = recommend_songs(prefs, self._songs(), k=1)
        assert isinstance(results[0][2], str)
        assert results[0][2].strip() != ""


# ---------------------------------------------------------------------------
# Recommender class
# ---------------------------------------------------------------------------

class TestRecommender:
    def test_recommend_returns_songs_sorted_by_score(self):
        user = _user()
        rec = _make_recommender()
        results = rec.recommend(user, k=3)
        assert len(results) == 3
        assert results[0].genre == "pop"
        assert results[0].mood == "happy"

    def test_recommend_k_limit(self):
        user = _user()
        rec = _make_recommender()
        results = rec.recommend(user, k=1)
        assert len(results) == 1

    def test_recommend_prefers_matching_genre(self):
        user = _user(favorite_genre="rock", favorite_mood="intense", target_energy=0.92)
        rec = _make_recommender()
        results = rec.recommend(user, k=1)
        assert results[0].genre == "rock"

    def test_explain_recommendation_returns_non_empty_string(self):
        user = _user()
        rec = _make_recommender()
        explanation = rec.explain_recommendation(user, rec.songs[0])
        assert isinstance(explanation, str)
        assert explanation.strip() != ""

    def test_explain_recommendation_mentions_genre_match(self):
        user = _user(favorite_genre="pop")
        rec = _make_recommender()
        pop_song = next(s for s in rec.songs if s.genre == "pop")
        explanation = rec.explain_recommendation(user, pop_song)
        assert "genre match" in explanation

    def test_explain_recommendation_mentions_mood_match(self):
        user = _user(favorite_mood="happy")
        rec = _make_recommender()
        happy_song = next(s for s in rec.songs if s.mood == "happy")
        explanation = rec.explain_recommendation(user, happy_song)
        assert "mood match" in explanation
