"""Re-predict the validation fold of a trained checkpoint with another TTA / scale setting.

    python scripts/predict_oof.py --exp v1_effv2s --fold 0 --tta d4 --scales 1.0 --out v1_effv2s_d4
Writes runs/<out>/oof_fold<f>.npy (+ stems json) so that sweep_pp / cand_build can use it like an experiment.
"""
import sys, json, argparse, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch
from fil import data, paths, infer
from fil.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--tta", default="d4"); ap.add_argument("--scales", type=float, nargs="+", default=[1.0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stems = json.load(open(paths.CACHE / "stems.json")); idx = {s: i for i, s in enumerate(stems)}
    vst = json.load(open(paths.RUNS / a.exp / f"oof_fold{a.fold}_stems.json"))
    ck = torch.load(paths.RUNS / a.exp / f"fold{a.fold}.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["args"]["arch"], ck["args"]["encoder"], out_ch=ck.get("out_ch", 2), pretrained=False); m.load_state_dict(ck["model"])
    m = m.cuda().eval().to(memory_format=torch.channels_last)
    img = np.load(paths.CACHE / "img.npy", mmap_mode="r")
    out = paths.RUNS / a.out; out.mkdir(exist_ok=True, parents=True)
    dst = np.lib.format.open_memmap(out / f"oof_fold{a.fold}.npy", "w+", np.uint8, (len(vst), ck.get("out_ch", 2), paths.H, paths.W))
    json.dump(vst, open(out / f"oof_fold{a.fold}_stems.json", "w"))
    t = time.time()
    for j, s in enumerate(vst):
        p = infer.predict_full(m, np.ascontiguousarray(img[idx[s]]), tta=a.tta, scales=tuple(a.scales))
        dst[j] = np.clip(np.round(p * 255), 0, 255).astype(np.uint8)
    dst.flush(); print(f"{a.out} fold {a.fold}: {len(vst)} images in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
