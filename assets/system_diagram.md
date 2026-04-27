# System Diagram: VibeFinder with RAG

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        A["👤 User\nNatural language query\ne.g. 'something calm for studying'"]
        A2["👤 User\nStructured preferences\ne.g. genre=lofi, energy=0.35"]
    end

    subgraph RAG["RAG Mode  ·  src/rag_retriever.py"]
        B["song_to_text()\nConverts each song to a\nnatural language description"]
        C["RAGRetriever.index_songs()\nEncodes all song descriptions\ninto dense vectors\n(all-MiniLM-L6-v2)"]
        D["RAGRetriever.retrieve()\nEncodes query · cosine similarity\nreturns top-N matches with scores"]
        R["RAG Output\nTitle · Artist · Genre · Mood · Energy\nSemantic similarity score"]
    end

    subgraph SCORING["Structured Mode  ·  src/recommender.py"]
        E["score_song()\nGenre match +1.8\nMood match +1.4\nEnergy closeness up to +2.0\nAcoustic bonus +0.6"]
        F["recommend_songs()\nRanks all songs by score descending\nReturns top-K with explanations"]
        S["Structured Output\nTitle · Artist · Score\nPer-signal reason breakdown"]
    end

    subgraph DATA["Data Layer  ·  data/songs.csv"]
        G[("18 songs\ngenre · mood · energy\ntempo · valence\ndanceability · acousticness")]
    end

    subgraph TESTING["Evaluation Layer"]
        T1["pytest\ntests/test_recommender.py\nUnit tests: sort order, explanations"]
        T2["👤 Human Review\nreflection.md · model_card.md\nDirectional validity checks"]
        T3["5 Structured Profiles\nin main.py: normal · edge · adversarial"]
        T4["3 RAG Queries\nin main.py: calm · energetic · moody"]
    end

    A  -->|"natural language"| D
    A2 -->|"structured prefs"| E

    G  -->|"load songs"| B
    G  -->|"load songs"| E

    B  --> C
    C  -->|"embedding index"| D
    D  --> R

    E  --> F
    F  --> S

    R  -->|"RAG results"| T4
    S  -->|"ranked output"| T3
    E  -->|"score_song + Recommender class"| T1
    T1 -->|"pass / fail"| T2
    T3 -->|"observations"| T2
    T4 -->|"observations"| T2
```

## Component Descriptions

| Component | File | Role |
|---|---|---|
| **song_to_text()** | `src/rag_retriever.py` | Converts a song dict to a text description for embedding (e.g. `"Library Rain by Paper Lanterns - lofi chill very low energy acoustic"`) |
| **RAGRetriever.index_songs()** | `src/rag_retriever.py` | Pre-encodes all 18 songs into dense vectors using `all-MiniLM-L6-v2` at startup |
| **RAGRetriever.retrieve()** | `src/rag_retriever.py` | Encodes the natural language query, runs cosine similarity against song embeddings, returns top-N matches |
| **score_song()** | `src/recommender.py` | Rules-based scorer — awards weighted points for genre, mood, energy closeness, and acoustic preference |
| **recommend_songs()** | `src/recommender.py` | Scores every song in the catalog, sorts descending, returns top-K with per-signal explanations |
| **Recommender class** | `src/recommender.py` | OOP wrapper used by unit tests — delegates to `score_song()` internally |
| **data/songs.csv** | `data/` | Knowledge base — 18 songs, 10 features each — shared by both modes |
| **pytest** | `tests/` | Automated unit tests for sort order and explanation correctness |
| **Human Review** | `reflection.md`, `model_card.md` | Manual evaluation of directional validity across profiles |
