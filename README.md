# Rebel Locate

## How it works
Rebel Locate runs in two stages:

1. **Building (KNN), `knn.py`.** Reads the latitude/longitude from each photo's EXIF metadata and trains a K-Nearest Neighbors classifier on building names. K is chosen with 10-fold cross validation.
2. **Room (CNN), `cnn.py`.** Takes only the predicted building's photos and fine-tunes the last layer of a ResNet-18 pre-trained on [Places365](https://github.com/CSAILVision/places365), a scene-recognition dataset. The output layer has one node per room in that building (hallway, lounge, stairwell, …).

`rebel_locate.py` runs both stages and prints the top 3 buildings and rooms with their probabilities.

Photos taken within ~1 meter of each other (such as burst shots) always stay on the same side of every training/testing split. Otherwise near-duplicate photos would inflate the reported accuracy.

## Dataset
The photos aren't stored in this repo. Download the [Rebel Locate Dataset](https://drive.google.com/drive/u/4/folders/1_shwU9ab9lqalvdD6KOODgHzUmMU253G) and arrange it as one folder per building and one sub-folder per room:

```
Ethan/
├── CEB/
│   ├── hallway/
│   │   ├── IMG_0001.JPG
│   │   └── ...
│   └── lounge/
└── TBE/
    └── ...
```

Every photo needs GPS metadata. Photos without it are skipped and listed when the program runs.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
The Places365 weights (~45 MB) download automatically on the first run.

## Usage
```bash
python rebel_locate.py --data-dir Ethan --image path/to/photo.jpg
```

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `./Ethan` | Dataset folder |
| `--image` | `./IMG_8948.JPG` | Photo to locate (must have GPS metadata) |
| `--weights` | `./resnet18_places365.pth.tar` | Where the Places365 weights are cached |
| `--epochs` | `10` | CNN training epochs |
| `--max-k` | `50` | Largest K tried for KNN |

Example output:
```
Building Prediction (KNN):
  100.00% -> CEB

Room Prediction (CNN):
  45.79% -> hallway
  28.21% -> stairwell
  26.00% -> lounge
```
