# 🌿 Plant Health — Training

Репозиторий для обучения моделей проекта **Plant Health**, предшественник [plant-health](https://github.com/Lin-Lini/plant-health).

Здесь готовятся веса для прод-сервиса:  
- **YOLO11-Seg** — сегментация растений;  
- **YOLO11-Seg** — сегментация дефектов;  
- **EfficientNet-B0 (TorchScript)** — классификация видов.

---

## 📁 Структура проекта

```

.
├── yolo11-plants-seg-determined/    # обучение сегментации растений
├── yolo11-defects-seg-determined/   # обучение сегментации дефектов
└── train_species.ipynb              # обучение классификатора видов

````

---

## ⚙️ Требования

- Python 3.10+
- PyTorch (CUDA — опционально)
- Ultralytics (YOLOv8 / YOLO11)
- timm (для EfficientNet)
- pandas, matplotlib, scikit-learn

Установка (CPU-вариант):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics timm pandas matplotlib scikit-learn
````

---

## 🧩 Данные

### Сегментация (YOLO-формат)

`data.yaml`:

```yaml
path: /abs/path/to/dataset
train: /abs/path/to/train/images
val: /abs/path/to/val/images
names: [tree, shrub]
```

Структура:

```
dataset/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt
```

### Классификация видов

Формат `species_dataset/{train,val}/{class}/*.jpg`
или CSV-индекс, указанный в `train_species.ipynb`.

---

## 🚀 Обучение

### 1. Сегментация растений

```bash
cd yolo11-plants-seg-determined
python yolo_train.py \
  --data /data/plants_seg/data.yaml \
  --model yolo11l-seg.pt \
  --imgsz 1280 \
  --epochs 200 \
  --batch -1 \
  --workers 8 \
  --project runs/plants \
  --name exp_plants \
  --patience 25
```

### 2. Сегментация дефектов

```bash
cd yolo11-defects-seg-determined
python yolo_train.py \
  --data /data/defects_seg/data.yaml \
  --model yolo11l-seg.pt \
  --imgsz 1280 \
  --epochs 200 \
  --batch -1 \
  --workers 8 \
  --project runs/defects \
  --name exp_defects \
  --patience 25
```

> 💡 `--batch -1` включает авто-батчинг.
> Для сегментации используйте веса `*-seg.pt`.
> В `yolo_train.py` реализовано копирование `last.pt → best.pt` при улучшении mAP.

### 3. Классификация видов

Откройте `train_species.ipynb`, настройте пути к данным и запустите обучение.
В конце экспортируйте TorchScript:

```python
script = torch.jit.script(model.eval().cpu())
script.save("model_ts.pt")
```

---

## 📦 Экспорт в прод-сервис

После обучения скопируйте лучшие веса в структуру,
ожидаемую [plant-health](https://github.com/Lin-Lini/plant-health):

```
plant-health/
├── weights/
│   ├── plant/plant_seg.pt
│   └── defect/defect_seg.pt
└── models/species/
    ├── model_ts.pt
    ├── species_classes.json
    └── species_ru_map.json
```

Переменные окружения:

```
PLANT_SEG_WEIGHTS=/srv/app/weights/plant/plant_seg.pt
DEFECT_SEG_WEIGHTS=/srv/app/weights/defect/defect_seg.pt
SPECIES_TS=/srv/app/models/species/model_ts.pt
SPECIES_CLASSES=/srv/app/models/species/species_classes.json
SPECIES_RU_MAP=/srv/app/models/species/species_ru_map.json
```

---

## 📊 Рекомендации

* Целевая метрика: `mAP50-95` для масок.
* Разделяй `train` и `val`, без утечек.
* Для дефектов усиливай аугментации, для растений — мягче.
* Для классификатора — баланс классов, `MixUp` и `CutMix` по необходимости.

---

## 🔁 Репродуцируемость

* Фиксируй `seed` (например, `42`).
* Логируй гиперпараметры и версии пакетов.
* Сохраняй `runs/**/results.csv` и примеры предсказаний.

---

## 👥 Авторы

**Проект «Зелёный Контроль»**
Полина Чудинова и команда.
(см. основной репозиторий — [plant-health](https://github.com/Lin-Lini/plant-health))
