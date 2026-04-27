"""
Command line runner for VibeFinder.

Demonstrates two recommendation modes:
  1. Structured mode  — explicit genre/mood/energy preferences → weighted scoring
  2. RAG mode         — natural language query → semantic retrieval via sentence-transformers
"""

import logging
from .recommender import load_songs, recommend_songs
from .rag_retriever import RAGRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def run_structured_profiles(songs: list) -> None:
    profiles = [
        (
            "High-Energy Pop",
            {"genre": "pop", "mood": "happy", "energy": 0.88, "likes_acoustic": False},
        ),
        (
            "Chill Lofi",
            {"genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True},
        ),
        (
            "Deep Intense Rock",
            {"genre": "rock", "mood": "intense", "energy": 0.92, "likes_acoustic": False},
        ),
        (
            "Edge Case: Conflicted High-Energy Sad",
            {"genre": "pop", "mood": "sad", "energy": 0.90, "likes_acoustic": False},
        ),
        (
            "Adversarial: Unknown Genre + Chill + Very Low Energy",
            {"genre": "k-pop", "mood": "chill", "energy": 0.05, "likes_acoustic": True},
        ),
    ]

    print(f"\n{SEPARATOR}")
    print("STRUCTURED SCORING MODE")
    print(SEPARATOR)

    for profile_name, user_prefs in profiles:
        logger.info("Running structured profile: %s", profile_name)
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print(f"\n=== Top Recommendations ===")
        print(f"Profile Name: {profile_name}")
        print(
            f"Profile: genre={user_prefs['genre']} | mood={user_prefs['mood']} | "
            f"energy={user_prefs['energy']:.2f} | likes_acoustic={user_prefs['likes_acoustic']}\n"
        )

        for index, rec in enumerate(recommendations, start=1):
            song, score, explanation = rec
            reason_items = [part.strip() for part in explanation.split(";") if part.strip()]
            print(f"{index}. {song['title']} — {song['artist']}")
            print(f"   Score: {score:.2f}")
            print("   Reasons:")
            for reason in reason_items:
                print(f"   - {reason}")
            print()


def run_rag_queries(retriever: RAGRetriever) -> None:
    queries = [
        "something calm and acoustic for studying on a rainy day",
        "high energy pump-up music for working out",
        "dark moody atmospheric late night drive",
    ]

    print(f"\n{SEPARATOR}")
    print("RAG NATURAL LANGUAGE MODE")
    print(SEPARATOR)

    for query in queries:
        logger.info("RAG query submitted: '%s'", query)
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)

        try:
            results = retriever.retrieve(query, top_n=3)
        except Exception as exc:
            logger.error("RAG retrieval failed for query '%s': %s", query, exc)
            print(f"  [Error] Could not retrieve results: {exc}")
            continue

        for rank, (song, similarity) in enumerate(results, 1):
            print(f"{rank}. {song['title']} — {song['artist']}")
            print(f"   [{song['genre']} | {song['mood']} | energy={song['energy']}]")
            print(f"   Semantic similarity: {similarity:.3f}")
        print()


def main() -> None:
    logger.info("VibeFinder starting up.")

    songs = load_songs("data/songs.csv")

    run_structured_profiles(songs)

    retriever = RAGRetriever()
    retriever.index_songs(songs)
    run_rag_queries(retriever)

    logger.info("VibeFinder run complete.")


if __name__ == "__main__":
    main()
