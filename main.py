import download_data
import extract_frames
import features
import evaluate
from visualize import plot_accuracy_bar, plot_per_class_accuracy, plot_frame_weights


def main():
    print("=" * 55)
    print(" STOP-Inspired: Temporal Variance-Weighted Aggregation")
    print("=" * 55)

    print("\n[1/5] Downloading HMDB51 subset...")
    download_data.main()

    print("\n[2/5] Extracting frames from videos...")
    extract_frames.main()

    print("\n[3/5] Extracting CLIP features...")
    features.main()

    print("\n[4/5] Evaluating pooling strategies...")
    results, per_class_results = evaluate.main()

    print("\n[5/5] Generating visualizations...")
    plot_accuracy_bar(results)
    plot_per_class_accuracy(per_class_results)
    plot_frame_weights(cls="dive",       num_examples=2)
    plot_frame_weights(cls="run",        num_examples=2)
    plot_frame_weights(cls="shoot_ball", num_examples=2)
    plot_frame_weights(cls="jump",       num_examples=2)

    print("\n" + "=" * 55)
    print("All done! Check the 'results/' folder for plots.")
    print("=" * 55)


if __name__ == "__main__":
    main()
