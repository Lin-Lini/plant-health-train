# YOLO11-L Seg — Determined AI experiment (plants / defects)

## Dataset mount on agent
- Plants: `/mnt/datasets/plants_seg`
- Defects: `/mnt/datasets/defects_seg`

Each folder must contain: `images/{train,valid,test}`, `labels/{train,valid,test}`, and `data.yaml` (Ultralytics Seg format).

## Run
```bash
export DET_MASTER=http://<master>:8080
det user login <you>

# Plants
cd yolo11-plants-det
det experiment create experiment.yaml .

# Defects
cd ../yolo11-defects-det
det experiment create experiment.yaml .
```
