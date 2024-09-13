import os
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Pfade zur CSV und zum Ordner mit den NIfTI-Dateien
ages_file = "../bildbearbeitung_final/ages.csv"
images_folder = "../bildbearbeitung_final/scaled_images_interlinear"

# 1. CSV-Datei laden
data = pd.read_csv(ages_file)

# 2. Verzeichnis mit NIfTI-Dateien einlesen
nifti_files = [f for f in os.listdir(images_folder) if f.endswith('.nii.gz')]

# 3. Verknüpfung der Dateinamen mit Altersdaten
file_age_map = {row['Dateiname']: row['Alter'] for _, row in data.iterrows()}

# Listen für Pfade und Altersangaben
file_paths = []
ages = []

for nifti_file in nifti_files:
    if nifti_file in file_age_map:
        file_paths.append(os.path.join(images_folder, nifti_file))
        ages.append(file_age_map[nifti_file])

# Konvertiere in NumPy-Arrays
file_paths = np.array(file_paths)
ages = np.array(ages)

# 4. Split in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(file_paths, ages, test_size=0.5, random_state=42)


# 5. Funktion zum Laden der NIfTI-Dateien und Extrahieren der Schichten
def load_nifti_file(file_path):
    nifti_image = nib.load(file_path)
    image_data = nifti_image.get_fdata()
    # Rückgabe von 100 Schichten des NIfTI-Bildes
    return image_data


# Funktion zum Vorbereiten der Daten
# Funktion zum Vorbereiten der Daten
def prepare_data(file_paths, ages):
    images = []
    target_ages = []

    for idx, file_path in enumerate(file_paths):
        image_data = load_nifti_file(file_path)
        # Iteriere durch jede Schicht des 3D-Bildes
        for i in range(image_data.shape[2]):  # assuming 3rd dimension is the slice dimension
            image_slice = image_data[:, :, i]
            images.append(image_slice)
            # Füge das Alter für jede Schicht hinzu (dupliziere es für jede Schicht)
            target_ages.append(ages[idx])

    return np.array(images), np.array(target_ages)


# 6. Trainings- und Testdaten vorbereiten
X_train_data, y_train_data = prepare_data(X_train, y_train)
X_test_data, y_test_data = prepare_data(X_test, y_test)

# Reshape für das CNN (füge eine Kanal-Dimension hinzu)
X_train_data = np.expand_dims(X_train_data, axis=-1)
X_test_data = np.expand_dims(X_test_data, axis=-1)

# Optional: Normalisierung der Daten
X_train_data = X_train_data / np.max(X_train_data)
X_test_data = X_test_data / np.max(X_test_data)



# 7. CNN Modell definieren
def create_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # Regression für Altersvorhersage
    ])
    return model


# Eingabeform des Modells
input_shape = (X_train_data.shape[1], X_train_data.shape[2], 1)  # (Height, Width, Channels)
model = create_cnn_model(input_shape)

model.summary()

# Modell kompilieren
# Modell kompilieren (Regression für Altersvorhersage)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


# 8. Modell trainieren
history = model.fit(X_train_data, y_train, epochs=10, batch_size=16, validation_data=(X_test_data, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# 9. Evaluierung des Modells auf Testdaten
test_loss, test_acc = model.evaluate(X_test_data, y_test)
print(f"Test Mean Absolute Error: {test_acc}")
