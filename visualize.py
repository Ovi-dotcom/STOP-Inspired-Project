import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from config import FEATURES_DIR, FRAMES_DIR, CLASSES, RESULTS_DIR
from aggregation import compute_frame_scores


def plot_accuracy_bar(results, save_path=None):
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    names  = list(results.keys())
    accs   = [results[k] * 100 for k in names]
    colors = ["#6c8ebf", "#d6b656", "#82b366"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=colors, width=0.45,
                  edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title(
        "Zero-Shot Video Classification\n"
        "Pooling Strategy Comparison (CLIP ViT-B/32, 10 HMDB51 Classes)",
        fontsize=12
    )
    ax.set_ylim(0, 100)
    ax.axhline(y=10, color="red", linestyle="--", linewidth=0.8,
               label="Random Chance (10%)")
    ax.legend(fontsize=9)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    plt.xticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_per_class_accuracy(per_class_results, save_path=None):
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "per_class_accuracy.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    strategies = list(per_class_results.keys())
    n_classes  = len(CLASSES)
    n_strats   = len(strategies)

    x      = np.arange(n_classes)
    width  = 0.25
    colors = ["#6c8ebf", "#d6b656", "#82b366"]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (name, color) in enumerate(zip(strategies, colors)):
        accs = [per_class_results[name][cls] * 100 for cls in CLASSES]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, accs, width, label=name,
                      color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Action Class", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title(
        "Per-Class Accuracy Breakdown\n"
        "Pooling Strategy Comparison (CLIP ViT-B/32, 10 HMDB51 Classes)",
        fontsize=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.axhline(y=10, color="red", linestyle="--",
               linewidth=0.8, label="Random Chance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_frame_weights(cls="dive", num_examples=2, save_path=None):
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, f"frame_weights_{cls}.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    cls_feat_dir  = os.path.join(FEATURES_DIR, cls)
    cls_frame_dir = os.path.join(FRAMES_DIR, cls)

    if not os.path.isdir(cls_feat_dir) or not os.path.isdir(cls_frame_dir):
        print(f"  Skipping frame weight plot for {cls} — data not found.")
        return

    feat_files = sorted([
        f for f in os.listdir(cls_feat_dir) if f.endswith(".npy")
    ])[:num_examples]

    if not feat_files:
        print(f"  Skipping frame weight plot for {cls} — no feature files.")
        return

    n_cols = 9
    fig, axes = plt.subplots(
        num_examples, n_cols,
        figsize=(n_cols * 2, num_examples * 2.5)
    )
    if num_examples == 1:
        axes = [axes]

    cmap = cm.RdYlGn

    for row, feat_file in enumerate(feat_files):
        emb    = np.load(os.path.join(cls_feat_dir, feat_file))
        frames = np.load(os.path.join(cls_frame_dir, feat_file))
        N      = len(emb)

        scores  = compute_frame_scores(emb)
        weights = scores / (scores.sum() + 1e-8)
        max_w   = weights.max() if weights.max() > 0 else 1.0

        for col in range(N):
            ax = axes[row][col]
            ax.imshow(frames[col])
            color = cmap(weights[col] / max_w)
            ax.set_title(f"w={weights[col]:.3f}", fontsize=7, color=color)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused frame subplots
        for col in range(N, 8):
            axes[row][col].axis("off")

        ax = axes[row][8]
        bar_colors = [cmap(weights[i] / max_w) for i in range(N)]
        ax.barh(range(N), weights, color=bar_colors)
        ax.set_yticks(range(N))
        ax.set_yticklabels([f"F{i}" for i in range(N)], fontsize=7)
        ax.set_title("Weight", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Score", fontsize=7)

        video_name = os.path.splitext(feat_file)[0]
        axes[row][0].set_ylabel(video_name[:20], fontsize=7,
                                rotation=0, labelpad=60, va="center")

    plt.suptitle(
        f"Temporal Variation Weights per Frame — class: '{cls}'\n"
        f"(Green = high motion change, Red = low change)",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    dummy_overall = {
        "Uniform Pooling (Baseline)": 0.50,
        "TV-Weighted Pooling (Ours)": 0.62,
        "Top-K Pooling (Ablation)":   0.56,
    }
    plot_accuracy_bar(dummy_overall)