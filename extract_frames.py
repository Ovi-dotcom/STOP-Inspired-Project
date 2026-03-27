import os
import cv2
import numpy as np
from tqdm import tqdm
from config import VIDEOS_DIR, FRAMES_DIR, CLASSES, NUM_FRAMES, MAX_VIDEOS_PER_CLASS


def extract_frames_from_video(video_path, num_frames=NUM_FRAMES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

    cap.release()
    return frames


def main():
    for cls in CLASSES:
        cls_video_dir = os.path.join(VIDEOS_DIR, cls, cls)
        cls_frame_dir = os.path.join(FRAMES_DIR, cls)
        os.makedirs(cls_frame_dir, exist_ok=True)

        videos = sorted([
            f for f in os.listdir(cls_video_dir)
            if f.lower().endswith(".avi")
        ])[:MAX_VIDEOS_PER_CLASS]

        print(f"\nExtracting frames — {cls} ({len(videos)} videos)")

        for vid in tqdm(videos, desc=cls):
            vid_name   = os.path.splitext(vid)[0]
            out_path   = os.path.join(cls_frame_dir, f"{vid_name}.npy")

            if os.path.exists(out_path):
                continue

            vid_path = os.path.join(cls_video_dir, vid)
            frames   = extract_frames_from_video(vid_path)

            if frames:
                np.save(out_path, np.array(frames, dtype=np.uint8))

    print("\nFrame extraction complete.")


if __name__ == "__main__":
    main()