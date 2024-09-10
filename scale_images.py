import os
import nibabel as nib
import numpy as np
import cv2  # OpenCV-Bibliothek zum Skalieren der Bilder

# Pfade zu den Ordnern
source_folder = './cropped_images2'  # Ordner mit den zurechtgeschnittenen NIfTI-Dateien
destination_folder = './scaled_images_interarea'  # Ordner zum Speichern der skalierten NIfTI-Dateien

# Erstelle den Zielordner, falls er nicht existiert
os.makedirs(destination_folder, exist_ok=True)

# Zielgröße
target_height = 80
target_width = 65  # Gerundet von 64.5

# Durchlaufe alle Dateien im Quellordner
for filename in os.listdir(source_folder):
    if filename.endswith('.nii.gz'):
        # Vollständiger Pfad zur Datei
        file_path = os.path.join(source_folder, filename)

        # Lade die NIfTI-Datei
        img = nib.load(file_path)
        img_data = img.get_fdata()

        # Originalgröße der Schichten
        original_height, original_width, num_slices = img_data.shape

        # Leeres Array für die skalierten Schichten erstellen
        scaled_img_data = np.zeros((target_width, target_height, num_slices))

        # Jede Schicht skalieren
        for i in range(num_slices):
            slice_ = img_data[:, :, i]
            # Skaliere die Schicht mit OpenCV
            scaled_slice = cv2.resize(slice_, (target_height, target_width), interpolation=cv2.INTER_AREA)
            scaled_img_data[:, :, i] = scaled_slice

        # Erstelle ein neues NIfTI-Bild mit den skalierten Daten
        scaled_img = nib.Nifti1Image(scaled_img_data, img.affine)

        # Speichere die skalierte NIfTI-Datei
        scaled_file_path = os.path.join(destination_folder, filename)
        nib.save(scaled_img, scaled_file_path)

        print(f"Die Datei {filename} wurde erfolgreich skaliert und in {scaled_file_path} gespeichert.")

print("Alle Dateien wurden skaliert und gespeichert.")
