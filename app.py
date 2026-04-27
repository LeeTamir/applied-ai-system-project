import streamlit as st
from src.recommender import load_songs
from src.rag_retriever import RAGRetriever

st.set_page_config(page_title="VibeFinder", page_icon="🎵", layout="centered")

FEATURE_LABELS = {
    "energy": "⚡ energy",
    "acousticness": "🎸 acoustic",
    "valence": "😊 mood",
    "danceability": "🕺 danceability",
}

# How to render each feature as a song tag
FEATURE_TAG_FN = {
    "energy":       lambda s: f"energy {s['energy']}",
    "danceability": lambda s: f"dance {s['danceability']}",
    "acousticness": lambda s: f"acoustic {s['acousticness']}",
    "valence":      lambda s: f"valence {s['valence']}",
    "tempo_bpm":    lambda s: f"{s['tempo_bpm']} BPM",
}


@st.cache_resource(show_spinner="Loading song catalog and AI model...")
def load_retriever():
    songs = load_songs("data/songs.csv")
    retriever = RAGRetriever()
    retriever.index_songs(songs)
    return retriever, songs


retriever, songs = load_retriever()

st.title("🎵 VibeFinder")
st.caption("Describe the vibe you're looking for and the AI will find the closest matches from the catalog.")

st.divider()

query = st.text_input(
    "What are you in the mood for?",
    placeholder="e.g. something calm and acoustic for studying on a rainy day",
)
top_k = st.slider("Number of results", min_value=1, max_value=len(songs), value=5)

search = st.button("Find Songs", type="primary", use_container_width=True)

if search:
    if not query.strip():
        st.warning("Please enter a description before searching.")
    else:
        with st.spinner("Finding matches..."):
            try:
                results = retriever.hybrid_retrieve(query.strip(), top_n=top_k)
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()

        st.subheader(f"Top {len(results)} matches for: _{query}_")

        if results:
            detected = results[0][2]["detected"]
            if detected:
                labels = " · ".join(FEATURE_LABELS.get(f, f) for f in detected)
                st.caption(f"Feature signals detected from your query: {labels} — used to re-rank results using CSV values.")
            else:
                st.caption("No specific feature signals detected — ranked by semantic similarity only.")

        st.divider()

        for rank, (song, score, details) in enumerate(results, 1):
            col_info, col_score = st.columns([4, 1])

            with col_info:
                st.markdown(f"**{rank}. {song['title']}** &nbsp; *{song['artist']}*")

                tags = [song["genre"], song["mood"]]
                detected_features = [f for f in details["detected"] if not f.startswith("genre:")]
                if detected_features:
                    for feat in detected_features:
                        if feat in FEATURE_TAG_FN:
                            tags.append(FEATURE_TAG_FN[feat](song))
                else:
                    tags.append(f"energy {song['energy']}")
                    if float(song.get("acousticness", 0)) >= 0.65:
                        tags.append("acoustic")
                st.markdown(" &nbsp;·&nbsp; ".join(f"`{t}`" for t in tags))

            with col_score:
                st.metric("Match", f"{score:.0%}")

            st.progress(float(score))
            st.divider()

st.divider()
with st.expander("How does this work?"):
    st.markdown(
        """
        VibeFinder uses **hybrid scoring** that combines two signals.

        **Step 1 — Semantic retrieval (RAG)**
        Every song is converted to a text description and encoded into a dense vector
        by `all-MiniLM-L6-v2`. Your query is encoded the same way. Cosine similarity
        finds the songs whose *meaning* is closest to your query.

        **Step 2 — Feature signal detection**
        The system scans your query for keywords that map to numeric columns in `songs.csv`:

        | Keyword examples | CSV feature boosted |
        |---|---|
        | *energetic, workout, pump* | `energy` ↑ |
        | *calm, chill, study, rainy* | `energy` ↓ |
        | *acoustic, unplugged, folk* | `acousticness` ↑ |
        | *happy, uplifting, feel-good* | `valence` ↑ |
        | *sad, dark, melancholic* | `valence` ↓ |
        | *dance, groove, party* | `danceability` ↑ |

        **Step 3 — Hybrid re-ranking**
        ```
        hybrid_score = 0.5 × semantic_similarity + 0.5 × feature_score
        ```
        The actual numeric values from `songs.csv` are used directly here — not text.
        If no feature signals are detected, the result falls back to pure semantic ranking.
        """
    )
