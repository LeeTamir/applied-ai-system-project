from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of dictionaries."""
    print(f"Loading songs from {csv_path}...")
    songs: List[Dict] = []

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            song = {
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            }
            songs.append(song)

    print(f"Loaded songs: {len(songs)}")
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences and return score plus reasons."""
    score = 0.0
    reasons: List[str] = []

    genre_pref = user_prefs.get("favorite_genre", user_prefs.get("genre"))
    mood_pref = user_prefs.get("favorite_mood", user_prefs.get("mood"))
    target_energy = user_prefs.get("target_energy", user_prefs.get("energy"))
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))

    genre_points = 1.8
    mood_points = 1.4
    max_energy_points = 2.0
    acoustic_bonus_points = 0.6

    if genre_pref is not None and song.get("genre") == genre_pref:
        score += genre_points
        reasons.append(f"genre match (+{genre_points:.1f})")

    if mood_pref is not None and song.get("mood") == mood_pref:
        score += mood_points
        reasons.append(f"mood match (+{mood_points:.1f})")

    if target_energy is not None and song.get("energy") is not None:
        energy_range = 0.95 - 0.22
        distance = abs(float(song["energy"]) - float(target_energy))
        energy_similarity = 1.0 - (distance / energy_range)
        energy_points = max(0.0, min(max_energy_points, max_energy_points * energy_similarity))
        score += energy_points
        reasons.append(f"energy closeness (+{energy_points:.2f})")

    if likes_acoustic and float(song.get("acousticness", 0.0)) >= 0.65:
        score += acoustic_bonus_points
        reasons.append(f"acoustic preference bonus (+{acoustic_bonus_points:.1f})")

    if not reasons:
        reasons.append("no strong feature match (+0.0)")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank all songs by score and return the top-k recommendations."""
    if k <= 0:
        return []

    scored_songs: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored_songs.append((song, score, explanation))

    ranked_songs = sorted(scored_songs, key=lambda item: item[1], reverse=True)
    return ranked_songs[:k]
