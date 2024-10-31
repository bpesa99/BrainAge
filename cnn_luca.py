import os
import pandas as pd
import nibabel as nib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Definiere die Callbacks
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


# Pfade zur CSV und zum Ordner mit den NIfTI-Dateien
ages_file = "./bildbearbeitung_final/ages2.csv"
images_folder = "../bildbearbeitung_final/scaled_images_interlinear2"

# 1. CSV-Datei laden
data = pd.read_csv(ages_file)

# 2. Verzeichnis mit NIfTI-Dateien einlesen
nifti_files = [f for f in os.listdir(images_folder) if f.endswith('.nii')]

# 3. Verknüpfung der Dateinamen mit Altersdaten
file_age_map = {row['Dateiname']: row['Alter'] for _, row in data.iterrows()}

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
X_train, X_test, y_train, y_test = train_test_split(file_paths, ages, test_size=0.5, random_state=42)


# 5. Funktion zum Laden der NIfTI-Dateien und Extrahieren der Schichten
def load_nifti_file(file_path):
    nifti_image = nib.load(file_path)
    image_data = nifti_image.get_fdata()
    # Rückgabe von 100 Schichten des NIfTI-Bildes
    return image_data


# Funktion zum Vorbereiten der Daten
# Funktion zum Vorbereiten der Daten
def prepare_data(file_paths, ages, slice_range=(20, 80)):
    images = []
    target_ages = []

    for idx, file_path in enumerate(file_paths):
        image_data = load_nifti_file(file_path)
        # Iteriere durch jede Schicht des 3D-Bildes
        for i in range(slice_range[0],slice_range[1]):  # assuming 3rd dimension is the slice dimension
            image_slice = image_data[:, :, i]
            images.append(image_slice)
            # Füge das Alter für jede Schicht hinzu (dupliziere es für jede Schicht)
            target_ages.append(ages[idx])

    return np.array(images), np.array(target_ages)


# Nach dem Training des Modells
def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))

    # Verlustkurven
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # MAE-Kurven
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Mean Absolute Error over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 6. Trainings- und Testdaten vorbereiten
X_train_data, y_train_data = prepare_data(X_train, y_train)
X_test_data, y_test_data = prepare_data(X_test, y_test)


# Reshape für das CNN (füge eine Kanal-Dimension hinzu)
X_train_data = np.expand_dims(X_train_data, axis=-1)
X_test_data = np.expand_dims(X_test_data, axis=-1)

# Optional: Normalisierung der Daten
# Normalisierung auf Basis des Maximalwerts
X_train_data = X_train_data / np.max(X_train_data) if np.max(X_train_data) != 0 else X_train_data
X_test_data = X_test_data / np.max(X_test_data) if np.max(X_test_data) != 0 else X_test_data


# 7. CNN Modell definieren
def create_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        #tf.keras.layers.MaxPooling2D((2, 2)),
        #tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        #tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(128, activation='relu'),
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
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)  # Lernrate reduzieren
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])

# 8. Modell trainieren
callbacks = create_callbacks()
history = model.fit(X_train_data, y_train_data, epochs=10, batch_size=16,
                    validation_data=(X_test_data, y_test_data),
                    callbacks=callbacks)

plot_learning_curves(history)

# 9. Evaluierung des Modells auf Testdaten
test_loss, test_acc = model.evaluate(X_test_data, y_test_data)
print(f"Test Mean Absolute Error: {test_acc}")
