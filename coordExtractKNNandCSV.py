from exif import Image
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.preprocessing import LabelEncoder
import csv as csv
# CHANGE THIS DIRECTORY WHEN RUNNING PROGRAM
parent_directory = "/home/ethan325/projects/cs422/ProjRebelLocate/CS422-RebelLocate/Ethan"

# TEMP 1D ARRAYS
imgname = []
buildingname = []
latitudes = []
longitudes = []

# Extract all subdirectories.
for subdir, _, files in os.walk(parent_directory):
    # Extract images from subdirectory.
    for file in files:

        # Images must end with .jpg/.jpeg.
        filePath = os.path.join(subdir, file)
        if not file.lower().endswith(('.jpg', '.jpeg')):
            continue

        # Open each image.
        with open(filePath, 'rb') as image_file:
            my_image = Image(image_file)

        # Extract coordinates from each image.
        try:

            # Convert coordinates from D/M/S notation into decimal.
            convertedLat = (my_image.gps_latitude[0] + (my_image.gps_latitude[1] / 60) + (my_image.gps_latitude[2] / 3600))
            if my_image.gps_latitude_ref == 'S':
                convertedLat = -convertedLat
            convertedLon = (my_image.gps_longitude[0] + (my_image.gps_longitude[1] / 60) + (my_image.gps_longitude[2] / 3600))
            if my_image.gps_longitude_ref == 'W':
                convertedLon = -convertedLon

            # Append information to temp 1D arrays.
            imgname.append(str(file))
            buildingname.append(str(subdir).replace(parent_directory, ""))
            latitudes.append(convertedLat)
            longitudes.append(convertedLon)

        # Images didn't have coordinate metadata.
        except AttributeError:
            print(f"ERROR: {filePath} doesn't have coordinate metadata.")

# !REMINDER! FORMAT TO THE 7th decimal place to match google earth
# Creates a CSV file that contains columns for the image and building name, latitude, longitude, and label 
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image name", "Building name", "Latitude", "Longitude", "Label"])
    for i in range(len(imgname)):
        # Adds a row containing the attributes of a single photo
        # Latitude and longitude rounded to the 7th decimal to remain consistent with google earth coordinates
        row = [imgname[i], buildingname[i], round(latitudes[i], 7), round(longitudes[i], 7), ' ']
        writer.writerow(row)


# Make 2D matrix holding all data.
df = np.column_stack((imgname, buildingname, latitudes, longitudes))

# SCATTER PLOT OF ALL POINTS.
plt.figure(figsize=(8, 6))
plt.scatter(longitudes, latitudes,  marker='o', c = LabelEncoder().fit_transform(df[:, 1:2].ravel()), alpha=0.6)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Scatter Plot of Image GPS Coordinates")
plt.grid(True)
plt.show()

# Randomize to create distributed labels. (More accurate training).
np.random.shuffle(df)

# Amount of rows in df.
dfSize = df.shape[0]

# Sizes of 75/25 dataset partitions.
trSize = round(df.shape[0] * 0.75)
testSize = df.shape[0] - trSize

# Split datasets for building names. (1D enumerated label array [y])
bNamesTr = df[0:trSize, 1:2].ravel()
bNamesTrEnc = LabelEncoder().fit_transform(bNamesTr)  # One-Hot Encode
bNamesTest = df[trSize:dfSize, 1:2].ravel()
bNamesTestEnc = LabelEncoder().fit_transform(bNamesTest)  # One-Hot Encode

# Split datasets for coordinates. (2D numerical array [X])
coordsTrain = df[0:trSize, 2:4].astype(float)
coordsTest = df[trSize:dfSize, 2:4].astype(float)

# Train KNN model.
knn = neighbors.KNeighborsClassifier(n_neighbors=round(math.sqrt(dfSize)), weights = 'distance', metric='euclidean')
knn.fit(coordsTrain, bNamesTrEnc)
pred = knn.predict(coordsTest)

# Determine accuracy score.
totalMatches = 0
for j in range(testSize):
    if bNamesTestEnc[j] == pred[j]:
        totalMatches += 1

print(f"Total Matches: {totalMatches}")
print(f"Accuracy: {totalMatches / testSize * 100:.2f}%")
print(f"CSV file created following the format: building name, image name, latitude, longitude, and label.")
