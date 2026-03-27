import os
import shutil
import requests
import rarfile
from tqdm import tqdm
from config import (HMDB51_URL, MAIN_RAR, VIDEOS_DIR,
                    CLASSES, DATA_DIR, WINRAR_UNRAR)

rarfile.UNRAR_TOOL = WINRAR_UNRAR

MIN_RAR_SIZE_BYTES = 500 * 1024 * 1024  # 500MB minimum


def download_file(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, timeout=120, headers=headers)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=os.path.basename(dest)
    ) as bar:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))


def validate_rar(path):
    size = os.path.getsize(path)
    if size < MIN_RAR_SIZE_BYTES:
        print(f"  ERROR: File is only {size / 1024 / 1024:.1f}MB — likely corrupted.")
        return False
    try:
        with rarfile.RarFile(path) as rf:
            rf.namelist()
        return True
    except Exception as e:
        print(f"  ERROR: RAR validation failed — {e}")
        return False


def extract_class_rars(main_rar_path, target_classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Opening main archive (this may take a moment)...")

    with rarfile.RarFile(main_rar_path) as rf:
        all_entries = rf.namelist()

        for cls in target_classes:
            cls_rar_name = f"{cls}.rar"
            match = next(
                (e for e in all_entries
                 if os.path.basename(e) == cls_rar_name), None
            )
            if match is None:
                print(f"  WARNING: {cls_rar_name} not found in archive. Skipping.")
                continue

            cls_rar_dest = os.path.join(out_dir, cls_rar_name)
            if not os.path.exists(cls_rar_dest):
                print(f"  Extracting {cls_rar_name} from main archive...")
                rf.extract(match, out_dir)
                extracted_path = os.path.join(out_dir, match)
                if extracted_path != cls_rar_dest and os.path.exists(extracted_path):
                    shutil.move(extracted_path, cls_rar_dest)
            else:
                print(f"  {cls_rar_name} already extracted, skipping.")

            cls_video_dir = os.path.join(out_dir, cls)
            if not os.path.isdir(cls_video_dir):
                os.makedirs(cls_video_dir, exist_ok=True)
                print(f"  Extracting {cls} videos...")
                with rarfile.RarFile(cls_rar_dest) as cls_rf:
                    cls_rf.extractall(cls_video_dir)
                print(f"  Done: {cls}")
            else:
                print(f"  {cls} videos already exist, skipping.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(MAIN_RAR):
        print("Found existing download. Validating...")
        if not validate_rar(MAIN_RAR):
            print("  Deleting corrupted file and re-downloading...")
            os.remove(MAIN_RAR)
        else:
            print("  File is valid.")

    if not os.path.exists(MAIN_RAR):
        print("Downloading HMDB51 (~2GB). This will take a while...\n")
        download_file(HMDB51_URL, MAIN_RAR)
        print("\nValidating downloaded file...")
        if not validate_rar(MAIN_RAR):
            print("\nDownload failed validation. Please try downloading manually:")
            print("  URL: http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar")
            print(f"  Save to: {os.path.abspath(MAIN_RAR)}")
            return

    extract_class_rars(MAIN_RAR, CLASSES, VIDEOS_DIR)

    print("\nDownload complete. Summary:")
    for cls in CLASSES:
        cls_dir = os.path.join(VIDEOS_DIR, cls, cls)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir)
                         if f.lower().endswith(".avi")])
            print(f"  {cls}: {count} videos")
        else:
            print(f"  {cls}: NOT FOUND")


if __name__ == "__main__":
    main()
