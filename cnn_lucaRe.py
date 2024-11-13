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
    file_paths.append(os.path.join(images_folder, nifti_file))
    ages.append(file_age_map[nifti_file])

# Konvertiere in NumPy-Arrays
file_paths = np.array(file_paths)
ages = np.array(ages)


# 4. Split in Trainings- und Testsets
X_train, X_test, y_train, y_test = train_test_split(file_paths, ages, test_size=0.2, random_state=42)
# random_state: Steuert die Zufälligkeit der Datenaufteilung.
# Ohne random_state sind die Trainings- und Testdaten immer unterschiedlich,
# wodurch die Reproduzierbarkeit erschwert wird. random_state ∈ ℕ_0, wobei man
# aber bei einem Wert bleiben sollte für die Reproduzierbarkeit.

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

    for idx, file_path in enumerate(file_paths): #enumerate: iteriert den Indize als auch file_paths
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

# Reshape für das CNN (füge eine Kanal-Dimension hinzu), sonst können die Daten nicht verarbeitet werden
X_train_data = np.expand_dims(X_train_data, axis=-1) #Hinzufügen einer Dimension mit dem Wert 1, da wir nur einen Farbkanal(Graustufen) haben
X_test_data = np.expand_dims(X_test_data, axis=-1) # -1 fügt ans Ende eine Dimension hinzu. 0 würde am Anfang eine hinzufügen

# Optional: Normalisierung der Daten
X_train_data = X_train_data / np.max(X_train_data)
X_test_data = X_test_data / np.max(X_test_data)

# 7. CNN Modell definieren
def create_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        # Je tiefer die Schichten der Convolutional Layer, umso größer ist i.d.R. die Anzahl an Filtern (z.B. 32 bis 512 Filter)
        # Am Anfang werden meistens einfache Merkmale(Kanten, Ecken, Texturen) erkannt, weshalb auch weniger Filter ausreichen.
        # In den tieferen Schichten versucht das Modell komplexere Merkmale(z.B. Kombinationen aus Kanten und Texturen)
        # zu erkennen, weshalb man hier auch mehr Filter braucht.
        # Wenn die Anzahl an Convolutional Layern zu groß ist, dann tritt ein negative dimension error auf
        tf.keras.layers.BatchNormalization(),
        # Die Batch-Normalisierung normalisiert die Ausgaben eines Layers auf Mini-Batch-Ebene.
        # Für jeden Mini-Batch(Teil der Daten, der gleichzeitig durch das Netzwerk geht) werden die Verteilungen der Werte
        # so angepasst, dass jede Ausgabe nach der Normalisierung einen Mittelwert von 0 und eine Standardabweichung von 1 hat,
        # was das Training vereinfachen un die Konvergenz verbessern soll.
        tf.keras.layers.MaxPooling2D((2, 2)),
        # Max-Pooling reduziert die räumliche Dimension der Daten, indem es die wichtigsten Merkmale extrahiert und die Anzahl der zu bearbeitenden Informatioonen
        # reduziert, was die Berechnungen vereinfacht und die Effizienz erhöht.
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        #Die Funktion Flatten() wandelt mehrdimensionale Daten in einen eindimensionalen Vektor um,
        #damit die Hidden Layer die Daten der Convolutioonal Layer bearbeiten können.
        tf.keras.layers.Dense(128, activation='relu'),
        # Die Hidden Layer berechnen eine gewichtete Summe, um nichtlineare Beziehungen abzubilden.
        # Für komplexe Aufgaben können es zwischen 1024 und 4096 Neuronen oder mehr sein.
        # Jedes Neuron in einer solchen Layer lernt eine bestimmte Transformation der Daten, damit das Netzwerk koomplexe Muster erkennen kann.
        # Umso mehr Hidden Layer man hat, umso leichter kann das Netzwerk komplexe Abhängigkeiten in den Daten erfassen.
        # Die units bestimmen die Anzahl an Neuronen/Knoten in einer Hidden Layer. Mehr Neuronen können dem Layer ermöglichen,
        # mehr und komplexere Merkmale zu lernen, was dann aber auch mehr Rechenzeit benötigt.
        # Zu viele Neuronen können jedoch zu Overfitting führen.
        # Overfitting:  - Das Modell ist zu komplex, z.B. lernt es feine Details, die nur in den Trainingsdaten vorkommen
        #               - Das Modell wurde zu lange trainiert und hat sich zu sehr den Trainingsdaten angepasst
        # Die Aktivierungsfunktion entscheidet, wie die Ausgabe jedes Neurons transformiert wird und ist notwendig, um Nichtlinearitäten abzubilden.
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
        # Wir wollen nur das geschätzte Alter wissen, deshalb gibt es in der letzten Layer nur ein Neuron.
        # Die Aktievierungsfunktion linear bzw. None gibt das Alter ohne Transformation aus.
    ])
    return model

# Eingabeform des Modells
input_shape = (X_train_data.shape[1], X_train_data.shape[2], 1)  # (Height, Width, Channels)
model = create_cnn_model(input_shape)

model.summary()

# Modell kompilieren (Regression für Altersvorhersage)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Der Adam-Optimierer wird oft für tiefe neuronale Netzwerke eingesetzt, da er effizient und robust gegenüber Hyperparametereinstellungen ist.
# Die learing_rate legt fest, wie groß die Schritte sind, die das Netzwerk bei jeder Aktualisierung der Gewichte macht, um den Verlust zu minimieren.
# Ist die learing_rate zu groß kann es sein, dass das Modell divergiert statt zu konvergieren.
# Und ist die learing_rate zu klein dauert es sehr lange bi sdas Modell konvergiert.
# Für Adam wäre ein typischer Wert für die learning_rate 0.001
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_absolute_error'])

history = model.fit(X_train_data, y_train_data, epochs=5, batch_size=512, validation_data=(X_test_data, y_test_data))
# In jeder Epoche wird das Modell auf den Trainingsdaten trainiert und die Gewichte werden angepasst.
# Die batch_size legt fest, wie viele Daten gleichzeitig bearbeitet werden.
# Ist die batch_size zu groß kann es zu Speichermangel führen und ist sie zu klein verlangsamt dies das Training.
# Die validation_data dient dazu das Modell auf ungesehenen Daten zu evaluieren, um zu überwachen ob es z.B. zu Overfitting kommt.
model.save('BrainAge.keras')
#print(history.history)
#plt.plot(history.history['mean_absolute_error'], label='mean_absolute_error')
#plt.plot(history.history['val_mean_absolute_error'], label='val_mean_absolute_error')
#plt.xlabel('Epoch')
#plt.ylabel('Mean Absolute Error')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')

# 9. Evaluierung des Modells auf Testdaten
test_loss, test_acc = model.evaluate(X_test_data, y_test_data)
#Hier wird das bereits trainierte Modell bewertet.