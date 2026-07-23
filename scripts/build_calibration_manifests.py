"""Phase 0 deliverable (plan-v2 section 11, day 5): build calibration/eval
text pairs for all four tasks, check calib/eval disjointness, and persist
manifests (hashes + overlap stats) plus the raw text so later runs are
reproducible without re-downloading/re-shuffling.
"""
import json
import os

from selfquant.data.calibration import TASKS, build_manifest

CONFIG_DIR = "configs/calibration"
DATA_DIR = "results/calibration_data"

N_CALIB = 150
N_EVAL = 100
SEED = 0


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    for task in TASKS:
        print(f"building manifest for task={task} ...")
        manifest, calib, eval_texts = build_manifest(task, N_CALIB, N_EVAL, seed=SEED)
        print(
            f"  n_calib={manifest.n_calib} n_eval={manifest.n_eval} "
            f"ngram_overlap_ratio={manifest.ngram_overlap_ratio:.4f}"
        )
        if manifest.ngram_overlap_ratio > 0.05:
            print(f"  WARNING: overlap ratio above 5% for task {task} — inspect for leakage")

        with open(os.path.join(CONFIG_DIR, f"{task}.json"), "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        with open(os.path.join(DATA_DIR, f"{task}_calib.jsonl"), "w") as f:
            for t in calib:
                f.write(json.dumps({"text": t}) + "\n")
        with open(os.path.join(DATA_DIR, f"{task}_eval.jsonl"), "w") as f:
            for t in eval_texts:
                f.write(json.dumps({"text": t}) + "\n")

    print("\nall manifests built.")


if __name__ == "__main__":
    main()
