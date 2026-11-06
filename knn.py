# Stage 1: Predict the building from an image's EXIF GPS coordinates using KNN.
import os
from pathlib import Path

import numpy as np
from exif import Image as ExifImage
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

# Images must end with .jpg/.jpeg.
IMAGE_EXTENSIONS = (".jpg", ".jpeg")

# Coordinates rounded to 5 decimals (~1.1 meters) are treated as the same spot.
# Images from the same spot always stay together in training or testing,
# so near-duplicate burst shots can't inflate accuracy.
GROUP_DECIMALS = 5


# Convert coordinates from D/M/S notation into decimal.
def dms_to_decimal(dms, ref):
    value = dms[0] + dms[1] / 60 + dms[2] / 3600

    # South and West coordinates are negative.
    return -value if ref in ("S", "W") else value


# Extract (latitude, longitude) from an image, or None if it has no GPS metadata.
def read_gps(path):
    # Open each image.
    with open(path, "rb") as image_file:
        image = ExifImage(image_file)

    # Image didn't have coordinate metadata.
    try:
        lat = dms_to_decimal(image.gps_latitude, image.gps_latitude_ref)
        lon = dms_to_decimal(image.gps_longitude, image.gps_longitude_ref)
    except (AttributeError, KeyError):
        return None
    return lat, lon


# Label each image with its rounded location, used to group near-duplicates.
def location_group(lat, lon):
    return f"{round(lat, GROUP_DECIMALS)},{round(lon, GROUP_DECIMALS)}"


# Loop through every building/room directory and extract each image's coordinates.
# Returns image paths, building names, and coordinates.
def load_gps_dataset(data_dir):
    # Temporary 1D arrays to store info during each loop.
    paths, buildings, coords = [], [], []
    skipped = []

    # Loop through every building and room label.
    for subdir, dirs, files in os.walk(data_dir):
        dirs.sort() # Sort so the order is the same every run.
        relative = Path(os.path.relpath(subdir, data_dir))

        # Loop through every image.
        for file in sorted(files):
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(subdir, file)

            # Skip images without coordinate metadata.
            gps = read_gps(path)
            if gps is None:
                skipped.append(path)
                continue

            # Append information to temp 1D arrays.
            paths.append(path)
            buildings.append(relative.parts[0] if relative.parts else ".")
            coords.append(gps)

    # Report how many images were used and skipped.
    print(f"Loaded {len(paths)} photos with GPS; skipped {len(skipped)} without GPS.")
    for path in skipped[:20]:
        print(f"  no GPS: {path}")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more")
    return paths, np.array(buildings), np.array(coords, dtype=float)


# Create a KNN classifier where closer neighbors count more.
def make_knn(k):
    return neighbors.KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")


# Use K-Fold Cross Validation to find the most accurate K value.
# Returns the best K and its average accuracy.
def tune_knn(coords, labels, groups, max_k):
    # Use 10 folds, or fewer if the smallest building has less than 10 images.
    smallest_class = np.bincount(labels).min()
    n_splits = max(2, min(10, smallest_class))
    if n_splits < 10:
        print(f"Note: smallest building has {smallest_class} photos; using {n_splits}-fold CV.")

    # Stratified folds keep each building's proportion; grouped folds keep each spot together.
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    folds = list(cv.split(coords, labels, groups))

    # K can't be larger than the smallest training set.
    max_k = min(max_k, min(len(train_idx) for train_idx, _ in folds))

    # Variables that will be constantly updated to determine best K.
    best_k, best_score = 1, -1.0

    # Test every K value.
    for k in range(1, max_k + 1):
        fold_scores = []

        # Train and attain accuracy for each fold.
        for train_idx, test_idx in folds:
            knn = make_knn(k).fit(coords[train_idx], labels[train_idx])
            fold_scores.append(accuracy_score(labels[test_idx], knn.predict(coords[test_idx])))

        # If new leader, update the variables.
        score = float(np.mean(fold_scores))
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


# Predict building probabilities for an image's coordinates.
# Returns [(building, probability), ...] sorted from most to least likely.
def rank_buildings(knn, label_encoder, gps):
    probs = knn.predict_proba([gps])[0]

    # Translate numerical labels back into viewable building names.
    names = label_encoder.inverse_transform(knn.classes_)
    order = np.argsort(probs)[::-1]
    return [(str(names[i]), float(probs[i])) for i in order]
