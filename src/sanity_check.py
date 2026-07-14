import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import run_pipeline

DATA_DIR = r"C:\Users\chere\Downloads\archive (8)\casting_data\casting_data\train"

def check(class_dir, label, n=25):
    files = sorted(os.listdir(class_dir))[:n]
    void_counts = []
    void_areas = []
    for f in files:
        r = run_pipeline(os.path.join(class_dir, f))
        void_counts.append(r["blob_stats"]["num_candidate_voids"])
        void_areas.append(r["blob_stats"]["total_void_area_px"])
    print(f"{label}: n={len(files)}")
    print(f"  num_candidate_voids -> mean={sum(void_counts)/len(void_counts):.2f}, "
          f"min={min(void_counts)}, max={max(void_counts)}")
    print(f"  total_void_area_px  -> mean={sum(void_areas)/len(void_areas):.1f}, "
          f"min={min(void_areas)}, max={max(void_areas)}")
    return void_counts, void_areas

if __name__ == "__main__":
    check(os.path.join(DATA_DIR, "def_front"), "DEFECTIVE")
    check(os.path.join(DATA_DIR, "ok_front"), "NORMAL")
