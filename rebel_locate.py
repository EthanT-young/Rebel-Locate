"""RebelLocate: predict which UNLV building and room a photo was taken in.

Stage 1 (knn.py): KNN on the photo's EXIF GPS coordinates predicts the building.
Stage 2 (cnn.py): A ResNet18 pre-trained on Places365 is fine-tuned on that
                  building's photos and predicts the room.

Expected dataset layout (one folder per building, one sub-folder per room):

    <data-dir>/<BUILDING>/<room>/*.jpg      e.g. Ethan/CEB/hallway/IMG_0001.JPG

Usage:
    python rebel_locate.py --data-dir Ethan --image IMG_8948.JPG
"""
# The text above is also shown by `python rebel_locate.py --help`.
import argparse
import sys
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder

import cnn
import knn

# Directory this file is in, used for default paths.
SCRIPT_DIR = Path(__file__).resolve().parent


# Read settings from the command line, using defaults when not given.
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "Ethan",
                        help="dataset folder laid out as <BUILDING>/<room>/*.jpg (default: ./Ethan)")
    parser.add_argument("--image", type=Path, default=SCRIPT_DIR / "IMG_8948.JPG",
                        help="photo to locate (default: ./IMG_8948.JPG)")
    parser.add_argument("--weights", type=Path, default=SCRIPT_DIR / "resnet18_places365.pth.tar",
                        help="where to cache the Places365 weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-k", type=int, default=50, help="largest k to try for KNN")
    return parser.parse_args()


def main():
    args = parse_args()

    # Make sure the dataset and image exist before doing any work.
    if not args.data_dir.is_dir():
        sys.exit(f"Dataset folder not found: {args.data_dir}")
    if not args.image.is_file():
        sys.exit(f"Image not found: {args.image}")

    # Extract coordinates from this image.
    target_gps = knn.read_gps(args.image)
    if target_gps is None:
        sys.exit(f"{args.image} has no GPS metadata, so its building can't be predicted.")

    # ---- Stage 1: Use KNN to predict what building you are in.
    print("Extracting metadata from dataset...")
    paths, buildings, coords = knn.load_gps_dataset(args.data_dir)
    if len(paths) < 2:
        sys.exit("Need at least 2 photos with GPS to train the KNN.")
    groups = [knn.location_group(lat, lon) for lat, lon in coords]

    # Encode all building names into numerical labels.
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(buildings)

    print("\n--- Optimizing KNN Model ---")
    best_k, best_score = knn.tune_knn(coords, labels, groups, args.max_k)
    print(f"Optimal K found: {best_k} with an expected building accuracy of {best_score * 100:.2f}%")

    # Train the final KNN on every image using the best K.
    building_knn = knn.make_knn(best_k).fit(coords, labels)
    building_ranking = knn.rank_buildings(building_knn, label_encoder, target_gps)
    building_name = building_ranking[0][0]

    # ---- Stage 2: Use CNN to predict what room you are in, training only on the predicted building.
    print(f"\nTraining CNN Model for {building_name}")
    dataset = cnn.load_room_dataset(args.data_dir / building_name)
    if len(dataset.classes) > 1:
        # Use the GPU if available.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cnn.train_room_cnn(dataset, paths, groups, args.weights, args.epochs, device)
        room_ranking = cnn.rank_rooms(model, dataset.classes, args.image, device)
    else:
        # Account for a building with a single room; nothing to train.
        print("Only one room in this building; skipping CNN training.")
        room_ranking = [(dataset.classes[0], 1.0)]

    # Print final results.
    print("\nFinal Predictions")
    print(f"Location GPS: {target_gps[0]:.6f}, {target_gps[1]:.6f}")

    # Top 3 buildings.
    print("\nBuilding Prediction (KNN):")
    for name, prob in building_ranking[:3]:
        if prob > 0:
            print(f"  {prob * 100:.2f}% -> {name}")

    # Top 3 rooms.
    print("\nRoom Prediction (CNN):")
    for name, prob in room_ranking[:3]:
        print(f"  {prob * 100:.2f}% -> {name}")


if __name__ == "__main__":
    main()
