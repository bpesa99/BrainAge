import os
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Pfade zur CSV und zum Ordner mit den NIfTI-Dateien
ages_file = "../bildbearbeitung_final/ages_neu2.csv"
images_folder = "../bildbearbeitung_final/scaled_Gefiltert_2/scaled_MRT_Gesund_neu"

# 1. CSV-Datei laden
data = pd.read_csv(ages_file)

# 2. Verzeichnis mit NIfTI-Dateien einlesen
nifti_files = [f for f in os.listdir(images_folder) if f.endswith('.nii')]

# 3. Verknüpfung der Dateinamen mit Altersdaten
file_age_map = {os.path.basename(row['Dateiname']): row['Alter'] for _, row in data.iterrows()}


# Listen für Pfade und Altersangaben
file_paths = []
ages = []

for nifti_file in nifti_files:
    file_paths.append(os.path.join(images_folder, nifti_file))
    ages.append(file_age_map[nifti_file])

# Konvertiere in NumPy-Arrays
file_paths = np.array(file_paths)
ages = np.array(ages)

# 4. Split in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(file_paths, ages, test_size=0.2, random_state=42)


# 5. Funktion zum Laden der NIfTI-Dateien und Extrahieren der Schichten
def load_nifti_file(file_path):
    nifti_image = nib.load(file_path)
    image_data = nifti_image.get_fdata()
    # Rückgabe von 100 Schichten des NIfTI-Bildes
    return image_data


# Funktion zum Vorbereiten der Daten
def prepare_data(file_paths, ages, slice_range=(0, 100)):
    images = []
    target_ages = []

    for idx, file_path in enumerate(file_paths):
        image_data = load_nifti_file(file_path)
        # Iteriere durch jede Schicht des 3D-Bildes
        for i in range(slice_range[0], slice_range[1]):  # assuming 3rd dimension is the slice dimension
            image_slice = image_data[:, :, i]
            images.append(image_slice)
            # Füge das Alter für jede Schicht hinzu (dupliziere es für jede Schicht)
            target_ages.append(ages[idx])

    return np.array(images), np.array(target_ages)


X_train_data, y_train_data = prepare_data(X_train, y_train)
X_test_data, y_test_data = prepare_data(X_test, y_test)

# Statistische Kennzahlen
print("Trainingsdaten Altersstatistik:")
print("Min:", np.min(y_train_data), "Max:", np.max(y_train_data), "Mean:", np.mean(y_train_data), "Std:", np.std(y_train_data))

print("\nTestdaten Altersstatistik:")
print("Min:", np.min(y_test_data), "Max:", np.max(y_test_data), "Mean:", np.mean(y_test_data), "Std:", np.std(y_test_data))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(y_train_data, bins=20, alpha=0.7, label='Trainingsdaten', color='blue')
plt.title('Altersverteilung in Trainingsdaten')
plt.xlabel('Alter')
plt.ylabel('Anzahl')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(y_test_data, bins=20, alpha=0.7, label='Testdaten', color='orange')
plt.title('Altersverteilung in Testdaten')
plt.xlabel('Alter')
plt.ylabel('Anzahl')
plt.legend()

plt.tight_layout()
plt.show()

# Reshape für das CNN (füge eine Kanal-Dimension hinzu)
X_train_data = np.expand_dims(X_train_data, axis=-1)
X_test_data = np.expand_dims(X_test_data, axis=-1)

# Optional: Normalisierung der Daten
X_train_data = X_train_data / np.max(X_train_data) if np.max(X_train_data) != 0 else X_train_data
X_test_data = X_test_data / np.max(X_test_data) if np.max(X_test_data) != 0 else X_test_data


# Laden des Modells
model = tf.keras.models.load_model('BrainAge.keras')


# 9. Evaluierung des Modells auf Testdaten
test_loss, test_mae = model.evaluate(X_test_data, y_test_data)
print(f"Test Mean Absolute Error: {test_mae}")


predicted_train_ages = model.predict(X_train_data)

# Vergleich für die Trainingsdaten erstellen und in einem DataFrame speichern
train_comparison_df = pd.DataFrame({
    'True Age': y_train_data,
    'Predicted Age': predicted_train_ages.flatten(),
    'Absolute Error': np.abs(y_train_data - predicted_train_ages.flatten())
})

# 11. Vorhersagen für die Testdaten generieren
predicted_test_ages = model.predict(X_test_data)

# Vergleich für die Testdaten erstellen und an das DataFrame anhängen
test_comparison_df = pd.DataFrame({
    'True Age': y_test_data,
    'Predicted Age': predicted_test_ages.flatten(),
    'Absolute Error': np.abs(y_test_data - predicted_test_ages.flatten())
})

# Zusammenfügen von Trainings- und Test-Datenframes
full_comparison_df = pd.concat([train_comparison_df, test_comparison_df], ignore_index=True)

# Speichern des gesamten Vergleichs als CSV
full_comparison_df.to_csv('age_comparison_neu.csv', index=False)

print("Die Ergebnisse für Trainings- und Testdaten wurden in 'age_comparison_neu.csv' gespeichert.")

