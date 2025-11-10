#!/usr/bin/env python3
# yolo_train.py — Ultralytics YOLOv8/11 Segmentation
import os, argparse, tempfile, sys, yaml, shutil
from ultralytics import YOLO

def set_cpu_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def preflight_data_yaml(data_path, override_train=None):
    d = read_yaml(data_path)
    for k in ["train", "val"]:
        if k not in d:
            raise ValueError(f"data.yaml missing '{k}' key")
    if override_train:
        d["train"] = override_train
    for k in ["train", "val"]:
        p = d[k]
        if not os.path.exists(p):
            p2 = os.path.join(os.path.dirname(data_path), p)
            if os.path.exists(p2):
                d[k] = p2
            else:
                print(f"[warn] Path not found: {p}", file=sys.stderr)
    if "names" in d:
        d["nc"] = len(d["names"])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    try:
        write_yaml(d, tmp.name)
    finally:
        tmp.close()
    return tmp.name, d

def make_best_callback():
    state = {"best": -1.0}
    # включаем масочные ключи для сегментации, а затем боксовые — на всякий случай
    keys = ["metrics/mAP50-95(M)", "seg/mAP50-95", "metrics/mAP50-95(B)", "box/mAP50-95", "metrics/mAP50(M)", "metrics/mAP50(B)"]
    def on_fit_epoch_end(tr):
        m = getattr(tr.validator, "metrics", {}) or {}
        for k in keys:
            if k in m:
                cur = float(m[k])
                if cur > state["best"]:
                    state["best"] = cur
                    try:
                        wdir = os.path.join(str(tr.save_dir), "weights")
                        src = os.path.join(wdir, "last.pt")
                        dst = os.path.join(wdir, "best.pt")
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"[warn] best-copy failed: {e}", file=sys.stderr)
                break
    return on_fit_epoch_end

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11l-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=-1)  # авто-батч
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--train_list", default=None)
    return ap.parse_args()

def main():
    set_cpu_env()
    args = parse_args()
    data_yaml_path, data_obj = preflight_data_yaml(args.data, override_train=args.train_list)

    print("[info] Effective data.yaml:")
    print(yaml.safe_dump(data_obj, allow_unicode=True, sort_keys=False))

    print("[info] Model:", args.model)
    model = YOLO(args.model)

    # регистрируем колбэк правильно
    model.add_callback("on_fit_epoch_end", make_best_callback())

    # безопасная обработка torch и выбора девайса
    try:
        import torch
        dev = 0 if torch.cuda.is_available() else "cpu"
    except Exception as e:
        print(f"[warn] torch import failed: {e}", file=sys.stderr)
        dev = "cpu"

    model.train(
        data=data_yaml_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        optimizer="auto",
        amp=True,
        cos_lr=True,
        seed=42,
        plots=True,
        save=True,
        save_period=1,
        device=dev
    )
    print("[done] Training finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())