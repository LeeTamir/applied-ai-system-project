"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n=== Top Recommendations ===")
    print(f"Profile: genre={user_prefs['genre']} | mood={user_prefs['mood']} | energy={user_prefs['energy']:.2f}\n")

    for index, rec in enumerate(recommendations, start=1):
        song, score, explanation = rec
        reason_items = [part.strip() for part in explanation.split(";") if part.strip()]

        print(f"{index}. {song['title']} — {song['artist']}")
        print(f"   Score: {score:.2f}")
        print("   Reasons:")
        for reason in reason_items:
            print(f"   - {reason}")
        print()


if __name__ == "__main__":
    main()
