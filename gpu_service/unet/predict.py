"""Out-of-fold predictions: every scan is scored by the fold model that never saw its patient,
through the same TTA code the service runs, so the analysis measures what the gate would see.

In plain words: After training, each B-scan is segmented by the one fold model that never saw its
patient ("out-of-fold"). Those masks and confidences are what the analysis judges, so every number
in the report comes from unseen patients.

Writes outputs/<experiment>/oof_predictions.csv, one row per scan with the image, the three
grader masks and the predicted mask linked (image and grader paths are relative to the
manifest, as in the manifest; pred_mask is relative to the CSV), plus the predicted masks
themselves in outputs/<experiment>/oof_masks/.

    python -m gpu_service.unet.predict --config gpu_service/unet/config.yaml
Next step: python -m analysis.gate_analysis --experiment outputs/unet_w32_cv5_ez_border
"""
import argparse                          # command-line options
import csv                               # the output table
from pathlib import Path                 # file paths

import torch                             # GPU check
from PIL import Image                    # saving the masks

from gpu_service.unet.data import assign_folds, load_scans         # the same data and folds as training
from gpu_service.unet.train import load_config                     # the same YAML
from gpu_service.unet.unet import load_checkpoint, preprocess      # the same checkpoint format and preprocessing
from gpu_service.app import measurements  # the exact function the service answers with
from gpu_service.tta import predict_tta   # the exact TTA the service runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gpu_service/unet/config.yaml")
    cfg = load_config(parser.parse_args().config)
    out = Path("outputs") / cfg["experiment_name"]
    (out / "oof_masks").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows, images, _ = load_scans(cfg["data"]["manifest"])   # graders' masks not needed here
    # Same seed -> the same patient-to-fold assignment training used.
    folds = assign_folds(rows, cfg["data"]["n_folds"], cfg["seed"])
    results = []
    for fold in range(cfg["data"]["n_folds"]):
        model, name = load_checkpoint(out / f"fold{fold}.pt", device)  # the model that never saw this fold
        for i, r in enumerate(rows):
            if folds[r["patient"]] != fold:  # only this fold's test patients
                continue
            mask, confidence = predict_tta(model, preprocess(images[i]).to(device))  # 6 passes -> mask + score
            mask = mask.cpu().numpy()
            pred_path = Path("oof_masks") / f"{r['scan_id']}.png"
            Image.fromarray(mask).save(out / pred_path)
            results.append({"scan_id": r["scan_id"], "patient": r["patient"], "cohort": r["cohort"],
                            "fold": fold, "model": name, "confidence": confidence, **measurements(mask),
                            "image": r["image"], **{f"mask_g{g}": r[f"mask_g{g}"] for g in (1, 2, 3)},
                            "pred_mask": pred_path.as_posix()})
        print(f"fold {fold}: {sum(x['fold'] == fold for x in results)} scans predicted", flush=True)

    with open(out / "oof_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(results[0]))  # columns = the first row's keys
        w.writeheader()
        w.writerows(sorted(results, key=lambda x: (x["patient"], x["scan_id"])))  # stable order for diffs
    print(f"wrote {len(results)} rows -> {out / 'oof_predictions.csv'}")


if __name__ == "__main__":
    main()
