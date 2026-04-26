# System Diagram: VibeFinder 1.0 with RAG

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        A["👤 User\nNatural language vibe description\ne.g. 'something chill for studying'"]
        A2["👤 User\nStructured preferences\ne.g. genre=lofi, energy=0.35"]
    end

    subgraph RAG["RAG Retrieval Layer\n(sentence-transformers)"]
        B["Text Encoder\nEncodes user query\ninto embedding vector"]
        C["Song Embeddings\nPre-encoded song descriptions\nstored as vectors"]
        D["Cosine Similarity Search\nFinds semantically closest songs\nfrom catalog"]
    end

    subgraph SCORING["Content-Based Scoring Layer\n(recommender.py)"]
        E["score_song()\nGenre match +1.8\nMood match +1.4\nEnergy closeness up to +2.0\nAcoustic bonus +0.6"]
        F["recommend_songs()\nRanks all songs by score\nTie-breaks by energy distance\nthen danceability"]
    end

    subgraph DATA["Data Layer"]
        G[("data/songs.csv\n18 songs\ngenre · mood · energy\ntempo · valence\ndanceability · acousticness")]
    end

    subgraph OUTPUT["Output Layer"]
        H["Top-K Recommendations\nTitle · Artist · Score\nExplanation per song"]
    end

    subgraph TESTING["Human & Automated Evaluation"]
        T1["pytest\ntests/test_recommender.py\nUnit tests for scoring logic"]
        T2["👤 Human Review\nProfile comparison analysis\nreflection.md · model_card.md"]
        T3["Manual Profile Tests\n5 test profiles in main.py\nDirectional validity checks"]
    end

    A -->|"natural language query"| B
    A2 -->|"structured prefs"| E
    B --> D
    C --> D
    G -->|"load songs"| C
    G -->|"load songs"| E
    D -->|"candidate songs"| E
    E --> F
    F --> H
    H -->|"results"| T2
    E -->|"scoring function"| T1
    F -->|"ranked output"| T3
    T1 -->|"pass/fail"| T2
    T3 -->|"observations"| T2
```

## Component Descriptions

| Component | Role |
|---|---|
| **Text Encoder** | Converts natural language input into a dense vector using `sentence-transformers` (`all-MiniLM-L6-v2`) |
| **Song Embeddings** | Pre-encoded text representations of each song's features, enabling semantic search |
| **Cosine Similarity Search** | RAG retrieval step — finds songs whose meaning is closest to the user's query |
| **score_song()** | Rules-based scorer applying weighted feature matching against a structured UserProfile |
| **recommend_songs()** | Aggregates scores, ranks, and returns top-k results with explanations |
| **data/songs.csv** | Knowledge base — 18 songs with 10 features each |
| **pytest** | Automated unit tests for scoring correctness |
| **Human Review** | Manual profile comparison and model card evaluation by a person |
| **Manual Profile Tests** | 5 hardcoded test profiles in main.py used to check directional validity |
