import os

DATA_DIR        = "data"
VIDEOS_DIR      = os.path.join(DATA_DIR, "videos")
FRAMES_DIR      = os.path.join(DATA_DIR, "frames")
FEATURES_DIR    = os.path.join(DATA_DIR, "features")
RESULTS_DIR     = "results"

CLASSES             = ["clap", "dive", "run", "shoot_ball", "swing_baseball"]
NUM_FRAMES          = 8
MAX_VIDEOS_PER_CLASS = 25
TOPK                = 4

HMDB51_URL  = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
MAIN_RAR    = os.path.join(DATA_DIR, "hmdb51_org.rar")

# WinRAR path — change this if WinRAR is installed elsewhere
WINRAR_UNRAR = r"C:\Program Files\WinRAR\UnRAR.exe"