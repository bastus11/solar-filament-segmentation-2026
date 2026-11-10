"""Average the probability maps of several checkpoints on the test images.

    python scripts/predict_test.py --ckpts runs/v1_effv2s/fold0.pt runs/v1_effv2s/fold1.pt --out v1 --tta flip4
Writes runs/test/<out>.npy  (N_test, 2, H, W) uint8.
"""
import sys, argparse, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, torch
from fil import paths, infer
from fil.model import build_model


def load_models(ckpts):
    models = []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        m = build_model(ck["args"]["arch"], ck["args"]["encoder"], out_ch=ck.get("out_ch", 2), pretrained=False)
        m.load_state_dict(ck["model"])
        models.append(m.cuda().eval().to(memory_format=torch.channels_last))
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tta", default="flip4")
    a = ap.parse_args()
    timg = np.load(paths.CACHE / "test_img.npy", mmap_mode="r")
    models = load_models(a.ckpts)
    out = paths.RUNS / "test"; out.mkdir(exist_ok=True)
    dst = np.lib.format.open_memmap(out / f"{a.out}.npy", "w+", np.uint8, (len(timg), 2, paths.H, paths.W))
    t = time.time()
    for i in range(len(timg)):
        img = np.ascontiguousarray(timg[i])
        p = sum(infer.predict_full(m, img, tta=a.tta) for m in models) / len(models)
        dst[i] = np.clip(np.round(p * 255), 0, 255).astype(np.uint8)
    dst.flush()
    print(f"saved {out / (a.out + '.npy')} ({len(models)} models, {time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
