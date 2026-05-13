#!/usr/bin/env python3
"""
UMAP embedding of lead-lag latency profiles (78 channels).

Each channel is a 78-d vector (its row in the latency matrix). Reduces to 2D, 3D,
or 4D for visualization; nearby points have similar lead-lag profiles.
- 2D: scatter
- 3D: 3D scatter (saved as PNG from fixed view)
- 4D: 3D scatter with 4th dimension as color
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio

DEFAULT_DATA_DIR = Path(
    "/vol/brains/bd3/pesaranlab/Archie_RecStim_vSUBNETS220_2nd/matlab/mfiles/Chen/session134/trial001_bipolar"
)


def load_latency_and_labels(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load latency matrix and channel names."""
    npy_path = data_dir / "lead_lag_latency_ms.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Latency matrix not found: {npy_path}")
    latency = np.load(npy_path)

    mat_path = data_dir / "lead_lag_latency_ms.mat"
    channel_names = None
    if mat_path.exists():
        try:
            data = sio.loadmat(str(mat_path))
            ch = data.get("channels")
            if ch is not None:
                ch = np.squeeze(ch)
                if ch.ndim == 0:
                    channel_names = [str(ch)]
                else:
                    channel_names = [str(np.squeeze(c)) for c in ch.flat]
        except Exception:
            pass
    if channel_names is None or len(channel_names) != latency.shape[0]:
        n_ch = latency.shape[0]
        channel_names = [f"b{i:03d}" for i in range(1, n_ch + 1)]
    return latency, channel_names


def _run_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """PCA via SVD; return (n_samples, n_components)."""
    from scipy import linalg
    Xc = X - X.mean(axis=0)
    U, s, Vt = linalg.svd(Xc, full_matrices=False)
    return U[:, :n_components] * s[:n_components]


def run_embedding(
    X: np.ndarray,
    n_components: int = 2,
    method: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Run PCA, UMAP, or TSNE; return (n_samples, n_components). n_components in {2,3,4}. method: pca, umap, tsne."""
    n_components = int(n_components)
    if n_components not in (2, 3, 4):
        n_components = 2
    method = (method or "umap").lower().strip()

    if method == "pca":
        print("Using PCA.")
        return _run_pca(X, n_components)

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                metric="euclidean",
            )
            return reducer.fit_transform(X)
        except ImportError:
            print("umap-learn not found; falling back to TSNE (2D) or PCA (3D/4D).")

    if method == "tsne" or (method == "umap" and n_components == 2):
        try:
            from sklearn.manifold import TSNE
            nc = min(n_components, 2)
            if n_components > 2:
                print("Using PCA for 3D/4D (TSNE only supports 2D).")
                return _run_pca(X, n_components)
            if method == "umap":
                print("Using TSNE (umap-learn not found).")
            reducer = TSNE(n_components=nc, random_state=random_state, perplexity=min(30, X.shape[0] - 1))
            return reducer.fit_transform(X)
        except ImportError:
            print("sklearn not found; using PCA.")
            return _run_pca(X, n_components)

    return _run_pca(X, n_components)


def _plot_2d(embedding: np.ndarray, channel_names: list, out_path: Path, figsize: tuple, dpi: int, method: str = "") -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=30, c="steelblue", edgecolors="white", linewidths=0.5)
    for i, name in enumerate(channel_names):
        ax.annotate(name, (embedding[i, 0], embedding[i, 1]), fontsize=4, alpha=0.8, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    title = "Lead-lag profile 2D embedding (78 channels)"
    if method:
        title = f"{method.upper()} " + title
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_3d(embedding: np.ndarray, channel_names: list, out_path: Path, figsize: tuple, dpi: int, method: str = "") -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.7, s=40, c="steelblue", edgecolors="white", linewidths=0.5)
    for i, name in enumerate(channel_names):
        ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2], name, fontsize=3, alpha=0.9)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    title = "Lead-lag profile 3D embedding (78 channels)"
    if method:
        title = f"{method.upper()} " + title
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_4d(embedding: np.ndarray, channel_names: list, out_path: Path, figsize: tuple, dpi: int, method: str = "") -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    c = embedding[:, 3]
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=c, cmap="viridis", alpha=0.8, s=40, edgecolors="white", linewidths=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, label="Component 4")
    for i, name in enumerate(channel_names):
        ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2], name, fontsize=3, alpha=0.9)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    title = "Lead-lag profile 4D embedding (78 channels; color = 4th dim)"
    if method:
        title = f"{method.upper()} " + title
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="2D/3D/4D embedding of lead-lag latency (78 channels)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory with lead_lag_latency_ms.npy")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG (default: data-dir/<method>_latency_<n>d.png)")
    parser.add_argument("--method", type=str, default="umap", choices=["pca", "umap", "tsne"], help="Embedding method: pca, umap, or tsne")
    parser.add_argument("--n-components", type=int, default=2, choices=[2, 3, 4], help="Embedding dimension (2=scatter, 3=3D, 4=3D+color)")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--seed", type=int, default=42, help="Random state")
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 8], metavar=("W", "H"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--save-embedding", action="store_true", help="Save embedding as .npy and .csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    n_comp = args.n_components
    method = args.method.lower()
    out_path = Path(args.output) if args.output is not None else data_dir / f"{method}_latency_{n_comp}d.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    latency, channel_names = load_latency_and_labels(data_dir)
    X = np.asarray(latency, dtype=np.float64)
    np.fill_diagonal(X, 0.0)
    print(f"Running {method.upper()} on {X.shape[0]} channels x {X.shape[1]} features -> {n_comp}D...")
    embedding = run_embedding(
        X,
        n_components=n_comp,
        method=method,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
    )
    print(f"Embedding shape: {embedding.shape}")

    figsize = tuple(args.figsize)
    if n_comp == 2:
        _plot_2d(embedding, channel_names, out_path, figsize, args.dpi, method)
    elif n_comp == 3:
        _plot_3d(embedding, channel_names, out_path, figsize, args.dpi, method)
    else:
        _plot_4d(embedding, channel_names, out_path, figsize, args.dpi, method)
    print(f"Saved: {out_path}")

    if args.save_embedding:
        np.save(data_dir / f"{method}_latency_embedding_{n_comp}d.npy", embedding)
        csv_path = data_dir / f"{method}_latency_embedding_{n_comp}d.csv"
        cols = ["channel"] + [f"dim{d+1}" for d in range(n_comp)]
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            for i, name in enumerate(channel_names):
                row = [name] + [f"{embedding[i,d]:.6f}" for d in range(n_comp)]
                f.write(",".join(row) + "\n")
        print(f"Saved embedding: {data_dir / f'{method}_latency_embedding_{n_comp}d.npy'}, {csv_path}")


if __name__ == "__main__":
    main()
