import json
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Erstelle leere Arrays, um die Daten zu speichern
train_images = []
train_labels = []
test_images = []
test_labels = []

# Durchlaufe die Trainingsdaten
train_data_dir = "storm_train_data"
train_label_dir = "storm_train_labels"

# Durchlaufe die Testdaten
test_data_dir = "storm_test_data"
test_label_dir = "storm_test_labels"


def load_data(image_path, label_path):
    # Lade das Bild und konvertiere es in ein Numpy-Array
    img = load_img(image_path, target_size=(366, 366))
    img = img_to_array(img) / 255.0

    # Lade die Labels aus der JSON-Datei
    with open(label_path) as f:
        labels = json.load(f)

    # Extrahiere die Windgeschwindigkeit aus den Labels
    wind_speed = labels["wind_speed"]

    return img, wind_speed

i=0

for storm_folder in os.listdir(test_data_dir):
    i += 1
    img_path = os.path.join(test_data_dir, storm_folder, "image.jpg")
    label_path = os.path.join(test_label_dir, storm_folder, "labels.json")
    img, wind_speed = load_data(img_path, label_path)
    test_images.append(img)
    test_labels.append(wind_speed)
    if i==5000: 
        break

test_images = np.array(test_images)
test_labels = np.array(test_labels)     

for storm_folder in os.listdir(train_data_dir):
    i += 1
    img_path = os.path.join(train_data_dir, storm_folder, "image.jpg")
    label_path = os.path.join(train_label_dir, storm_folder, "labels.json")
    img, wind_speed = load_data(img_path, label_path)
    train_images.append(img)
    train_labels.append(wind_speed)
    if i==15000: 
        break

train_images = np.array(train_images)
train_labels = np.array(train_labels)