import os
import nibabel as nib
import numpy as np
import pandas as pd

# Pfad zum Ordner, der die .nii.gz Dateien enthält
source_folder = './bearbeitet_nifti'  # Ersetze dies mit deinem Ordnerpfad

# Pfad zum Zielordner für die zugeschnittenen Bilder
cropped_folder = './cropped_images2'
os.makedirs(cropped_folder, exist_ok=True)

# Lade die Bounding-Box-Koordinaten aus der CSV-Datei
bounding_boxes_df = pd.read_csv('bounding_boxes2.csv')


# Durchlaufe jede Datei im Quellordner
for filename in os.listdir(source_folder):
    if filename.endswith('.nii.gz'):
        # Prüfe, ob die Datei in der CSV-Datei enthalten ist
        if filename in bounding_boxes_df['filename'].values:
            # Lade die entsprechenden Bounding-Box-Koordinaten
            box = bounding_boxes_df[bounding_boxes_df['filename'] == filename].iloc[0]
            min_row, min_col, max_row, max_col = int(box['min_row']), int(box['min_col']), int(box['max_row']), int(box['max_col'])

            # Vollständiger Pfad zur Datei
            file_path = os.path.join(source_folder, filename)

            # Lade die NIfTI-Datei
            img = nib.load(file_path)
            img_data = img.get_fdata()

            # Erstelle ein Array für die zugeschnittenen Schichten
            cropped_slices = []

            # Iteriere über die Schichten 100 bis 200
            for i in range(100, min(200, img_data.shape[2])):  # Min, um sicherzustellen, dass wir nicht über die maximale Schichtzahl hinausgehen
                # Schneide die Schicht anhand der Bounding Box zu
                cropped_slice = img_data[min_col:max_col, min_row:max_row, i]
                cropped_slices.append(cropped_slice)

            # Konvertiere die Liste der zugeschnittenen Schichten zurück in ein 3D-Numpy-Array
            cropped_slices = np.stack(cropped_slices, axis=-1)

            # Erstelle eine neue NIfTI-Bilddatei für das zugeschnittene Bild
            cropped_img = nib.Nifti1Image(cropped_slices, img.affine, img.header)

            # Speichere das zugeschnittene Bild
            cropped_file_path = os.path.join(cropped_folder, f"cropped_{filename}")
            nib.save(cropped_img, cropped_file_path)

            print(f"Gespeichert: {cropped_file_path}")


print("Zuschneiden und Speichern der Dateien abgeschlossen.")
