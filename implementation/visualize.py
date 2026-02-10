import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from pathlib import Path

# Configuration
DB_PATH = str(Path(__file__).parent.parent / "preprocessed_db")
COLLECTION_NAME = "docs"


def _choose_tsne_perplexity(n_samples: int, *, preferred: int = 30) -> int:
    """
    Pick a t-SNE perplexity that is always valid for the given sample size.

    Rules:
      - must satisfy perplexity < n_samples (scikit-learn requirement)
      - for small datasets, a common heuristic is perplexity <= (n_samples - 1) / 3
    """
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples for t-SNE, got {n_samples}")

    # Heuristic for stability on small datasets; also never exceed the user's preferred value.
    perplexity = min(preferred, (n_samples - 1) // 3)

    # Keep within a sensible minimum, while preserving the strict constraint perplexity < n_samples.
    perplexity = max(2, perplexity)
    perplexity = min(perplexity, n_samples - 1)

    return int(perplexity)


def visualize_chroma():
    # 1. Connect to ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # 2. Get all data (embeddings, documents, metadatas)
    results = collection.get(include=["embeddings", "documents", "metadatas"])

    embeddings = np.asarray(results["embeddings"], dtype=np.float32)
    documents = results["documents"]
    metadatas = results["metadatas"]
    ids = results["ids"]

    if len(embeddings) == 0:
        print("No embeddings found in the collection.")
        return

    # Create a DataFrame for easier handling
    df = pd.DataFrame(
        {
            "id": ids,
            "document": [
                (doc[:100] + "...") if len(doc) > 100 else doc for doc in documents
            ],
            "source": [m.get("source", "unknown") for m in metadatas],
            "type": [m.get("type", "unknown") for m in metadatas],
        }
    )

    # 3. Dimensionality Reduction (t-SNE)
    n_samples = embeddings.shape[0]
    perplexity = _choose_tsne_perplexity(n_samples)

    print(f"Running t-SNE for 2D... (n_samples={n_samples}, perplexity={perplexity})")
    tsne_2d = TSNE(
        n_components=2,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    embeddings_2d = tsne_2d.fit_transform(embeddings)
    df["x_2d"] = embeddings_2d[:, 0]
    df["y_2d"] = embeddings_2d[:, 1]

    print(f"Running t-SNE for 3D... (n_samples={n_samples}, perplexity={perplexity})")
    tsne_3d = TSNE(
        n_components=3,
        random_state=42,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    embeddings_3d = tsne_3d.fit_transform(embeddings)
    df["x_3d"] = embeddings_3d[:, 0]
    df["y_3d"] = embeddings_3d[:, 1]
    df["z_3d"] = embeddings_3d[:, 2]

    # 4. 2D Visualization
    fig_2d = px.scatter(
        df,
        x="x_2d",
        y="y_2d",
        color="type",
        hover_data=["id", "source", "document"],
        title="ChromaDB 2D Visualization (t-SNE)",
        labels={"x_2d": "Dimension 1", "y_2d": "Dimension 2"},
    )
    fig_2d.show()

    # 5. 3D Visualization
    fig_3d = px.scatter_3d(
        df,
        x="x_3d",
        y="y_3d",
        z="z_3d",
        color="type",
        hover_data=["id", "source", "document"],
        title="ChromaDB 3D Visualization (t-SNE)",
        labels={"x_3d": "Dimension 1", "y_3d": "Dimension 2", "z_3d": "Dimension 3"},
    )
    fig_3d.show()


if __name__ == "__main__":
    visualize_chroma()