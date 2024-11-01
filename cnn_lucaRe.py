import os
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Pfade zur CSV und zum Ordner mit den NIfTI-Dateien
#ages_file = 'agesneu.csv'#"../bildbearbeitung_final/ages.csv"
images_folder = 'C:/Users/bruno/OneDrive/Bilder/scaled_Gefiltert_2/scaled_MRT_Gesund' #'./Daten_Bearbeitet/scaled_images_interlinear_Neu' #"../bildbearbeitung_final/scaled_images_interlinear"
# 1. CSV-Datei laden
data = pd.read_csv('ages_neu_bruno.csv')#, sep='\t')
# 2. Verzeichnis mit NIfTI-Dateien einlesen
nifti_files = [f for f in os.listdir(images_folder) if f.endswith('.gz')]

# 3. Verknüpfung der Dateinamen mit Altersdaten
file_age_map = {row['Dateiname']: row['Alter'] for _, row in data.iterrows()}

# Listen für Pfade und Altersangaben
file_paths = []
ages = []
for nifti_file in nifti_files:
    #if nifti_file in file_age_map:
    file_paths.append(os.path.join(images_folder, nifti_file))
    ages.append(file_age_map[nifti_file])

# Konvertiere in NumPy-Arrays
file_paths = np.array(file_paths)
ages = np.array(ages)


# 4. Split in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(file_paths, ages, test_size=0.8, random_state=42)

# 5. Funktion zum Laden der NIfTI-Dateien und Extrahieren der Schichten
def load_nifti_file(file_path):
    nifti_image = nib.load(file_path)
    image_data = nifti_image.get_fdata()
    # Rückgabe von 100 Schichten des NIfTI-Bildes
    return image_data

# Funktion zum Vorbereiten der Daten
def prepare_data(file_paths, ages,slice_range=(0, 100)):#slice_range=(30, 50)
    images = []
    target_ages = []

    for idx, file_path in enumerate(file_paths):
        image_data = load_nifti_file(file_path)
        # Iteriere durch jede Schicht des 3D-Bildes
        for i in range(slice_range[0],slice_range[1]):  # slice_range[0],slice_range[1]
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

def create_callbacks():
    # Learning Rate Scheduler
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-5 * 10 ** (epoch / 20)
    )

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    return [lr_scheduler, early_stopping]

# 7. CNN Modell definieren
def create_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        #tf.keras.layers.Dense(256, activation='relu'),
        #tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        #tf.keras.layers.Dropout(0.5), #Dropout zur Vermeidung von Überanpassung
        tf.keras.layers.Dense(1, activation='linear')  # Regression für Altersvorhersage
    ])
    return model


# Eingabeform des Modells
input_shape = (X_train_data.shape[1], X_train_data.shape[2], 1)  # (Height, Width, Channels)
model = create_cnn_model(input_shape)

model.summary()

# Modell kompilieren
# Modell kompilieren (Regression für Altersvorhersage)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])

# 8. Modell trainieren
#X_train_data = np.clip(X_train_data, -1, 1)
#X_test_data = np.clip(X_test_data, -1, 1)
#X_train_data = X_train_data / np.max(X_train_data, axis=(0,1,2), keepdims=True)
#X_test_data = X_test_data / np.max(X_test_data, axis=(0,1,2), keepdims=True)

history = model.fit(X_train_data, y_train_data, epochs=200, batch_size=128, validation_data=(X_test_data, y_test_data)) #, callbacks=create_callbacks())
model.save('BrainAge.keras')
#print(history.history)
plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
plt.plot(history.history['val_mean_absolute_error'], label='val_mean_absolute_error')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')
# 9. Evaluierung des Modells auf Testdaten
test_loss, test_acc = model.evaluate(X_test_data, y_test_data)