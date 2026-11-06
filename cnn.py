# Stage 2: Predict the room within a building using a ResNet18 CNN with Places365 weights.
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image as PILImage
from PIL import ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

# Places365 pretrained weights, designed for scene/room recognition.
PLACES365_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"

# Image transformations that will be applied to all.
TRANSFORM = Compose([
    Resize((224, 224)), # Required size for Places365.
    ToImage(), # Convert every image into a tensor.
    ToDtype(torch.float32, scale=True), # Scale pixel values to 0-1.
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Known from Places365 pretraining.
])


# Open an image and rotate it upright using its EXIF orientation.
# Phones often save images sideways along with a tag saying how to rotate them,
# and PIL ignores that tag unless asked.
def load_upright(path):
    with PILImage.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


# Load every image in a building, labelled by its room directory.
def load_room_dataset(building_dir):
    return ImageFolder(building_dir, transform=TRANSFORM, loader=load_upright)


# Load ResNet18 and update with Places365 weights.
def load_places365_resnet18(weights_path, device):
    # Download pretrained weights if not already there.
    if not os.path.exists(weights_path):
        print("Downloading Places365 weights...")
        partial = f"{weights_path}.part"
        urllib.request.urlretrieve(PLACES365_URL, partial)
        os.replace(partial, weights_path) # Rename only once finished, so a cut-off download isn't reused.

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = models.resnet18(num_classes=365)

    # Remove 'module.' prefix from multi-GPU training.
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}

    # Import the weights from Places365.
    model.load_state_dict(state_dict)
    return model


# Normalize a path so the same file always matches.
def _path_key(path):
    return os.path.normcase(os.path.abspath(path))


# Use 80/20 training/testing split, keeping images from the same spot together.
# Returns training and testing indices; testing is empty if there's too little data.
def grouped_split(dataset, paths, groups, test_fraction=0.2):
    # Look up each image's location group from the KNN stage.
    path_to_group = {_path_key(p): g for p, g in zip(paths, groups)}
    sample_groups = [path_to_group.get(_path_key(path), path) for path, _ in dataset.samples]
    targets = np.array(dataset.targets)

    # Need enough distinct spots to split; otherwise train on everything.
    n_splits = round(1 / test_fraction)
    if len(set(sample_groups)) < n_splits:
        return np.arange(len(dataset)), np.array([], dtype=int)

    # Use the first of 5 stratified, grouped folds as the testing set.
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)
    return next(cv.split(np.zeros(len(targets)), targets, sample_groups))


# Evaluate accuracy on a set of images.
def accuracy_on(model, loader, device):
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Chooses the highest probable room and counts correct predictions.
            correct += (model(inputs).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# Train the last layer of the model on one building's rooms, then report testing accuracy.
def train_room_cnn(dataset, paths, groups, weights_path, epochs, device):
    # Split into training/testing sets.
    train_idx, test_idx = grouped_split(dataset, paths, groups)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)

    # Freeze all layers.
    model = load_places365_resnet18(weights_path, device)
    for param in model.parameters():
        param.requires_grad = False

    # Adjust last layer to the number of rooms in this building; it's the only layer trained.
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    print(f"Training on {len(train_idx)} photos, testing on {len(test_idx)}.")
    for epoch in range(epochs):
        # eval() keeps the frozen layers' BatchNorm statistics fixed.
        # The last layer has no dropout/BatchNorm, so it still trains normally.
        model.eval()
        running_loss = 0.0

        # Go through every image in the training set.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear previous gradients and run a forward pass.
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)

            # Backpropagate.
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss:.4f}")

    # Evaluate accuracy on the testing set.
    if len(test_idx) > 0:
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False)
        print(f"Room test accuracy: {accuracy_on(model, test_loader, device):.2f}%")
    else:
        print("Too few distinct photo locations to hold out a test set; room accuracy not measured.")
    return model


# Use CNN to predict what type of room the image is in.
# Returns [(room, probability), ...] sorted from most to least likely.
def rank_rooms(model, room_names, image_path, device):
    # Load and transform image.
    input_tensor = TRANSFORM(load_upright(image_path)).unsqueeze(0).to(device)

    # Use a softmax to gauge predictions with multiple labels.
    with torch.no_grad():
        probs = F.softmax(model(input_tensor), dim=1).squeeze(0).cpu()
    top_probs, top_idx = probs.sort(descending=True)
    return [(room_names[i], p.item()) for p, i in zip(top_probs, top_idx)]
