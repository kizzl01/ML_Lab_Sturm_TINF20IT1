import os
import json
import numpy as np
import pandas as pd
from PIL import Image

# Ordnerpfade definieren
test_data_path = 'storm_test_data'
test_labels_path = 'storm_test_labels'
train_data_path = 'storm_train_data'
train_labels_path = 'storm_train_labels'

# Funktion zum Laden der Bilder und Labels
def load_data(data_path, labels_path,max):
    # Leeres DataFrame erstellen
    data_df = pd.DataFrame(columns=['storm_id', 'relative_time', 'ocean', 'image_path', 'wind_speed'])
    index = 0
    # Alle Sturmordner durchlaufen
    for storm_folder in os.listdir(data_path):
        # Pfad zum Sturmordner erstellen
        storm_path = os.path.join(data_path, storm_folder)
        index +=1
        # Alle Bildordner des Sturms durchlaufen
        
            # Pfad zum Bildordner erstellen
        image_path = os.path.join(storm_path, 'image.jpg')
            
            # Features aus der JSON-Datei laden
        with open(os.path.join(storm_path, 'features.json')) as f:
            features = json.load(f)
            
            # Bild laden
        with Image.open(image_path) as img:
            image = np.array(img)
            
            # Labels aus der JSON-Datei laden
        with open(os.path.join(labels_path, storm_folder, 'labels.json')) as f:
            labels = json.load(f)
            
            # Daten zum DataFrame hinzufügen
        data_df = data_df.append({
            'storm_id': features['storm_id'],
            'relative_time': features['relative_time'],
            'ocean': features['ocean'],
            'image_path': image_path,
            'wind_speed': labels['wind_speed']
        }, ignore_index=True)
        if index==max: 
            break
    return data_df

# Trainings- und Testdaten laden
train_data = load_data(train_data_path, train_labels_path,10000)
test_data = load_data(test_data_path, test_labels_path,5000)