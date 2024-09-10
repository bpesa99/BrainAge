import os
import nibabel as nib
import numpy as np
import pandas as pd


def find_largest_bounding_box(image):
    """Findet die größte Bounding Box in einem 3D-Numpy-Array."""
    bounding_boxes = []

    # Iteriere durch jede Schicht (Z-Achse)
    for i in range(image.shape[2]):
        # Extrahiere die Schicht
        slice_ = image[:, :, i]

        # Finde die Positionen der nicht-null Elemente (annimmt, dass Gehirn Pixel != 0 sind)
        non_zero_coords = np.argwhere(slice_ > 0)

        if non_zero_coords.size == 0:
            continue

        # Bestimme die Bounding Box für diese Schicht
        min_col, min_row = np.min(non_zero_coords, axis=0)
        max_col, max_row = np.max(non_zero_coords, axis=0)

        # Speichere die Bounding Box
        bounding_boxes.append((i, min_row, min_col, max_row, max_col))

    # Finde die größte Bounding Box basierend auf der Fläche
    if bounding_boxes:
        largest_box = max(bounding_boxes, key=lambda x: (x[3] - x[1]) * (x[4] - x[2]))
        return largest_box
    else:
        return None


# Pfad zum Ordner, der die .nii.gz Dateien enthält
source_folder = './bearbeitet_nifti'  # Ersetze dies mit deinem Ordnerpfad

# Liste zum Speichern der Ergebnisse
results_list = []

# Hol dir die Liste der Dateien und sortiere sie alphabetisch
file_list = sorted([f for f in os.listdir(source_folder) if f.endswith('.nii.gz')])

# Durchlaufe die alphabetisch sortierten Dateien im Quellordner
for filename in file_list:
    # Vollständiger Pfad zur Datei
    file_path = os.path.join(source_folder, filename)

    # Lade die NIfTI-Datei
    img = nib.load(file_path)
    img_data = img.get_fdata()

    # Finde die größte Bounding Box
    largest_box = find_largest_bounding_box(img_data)

    if largest_box:
        slice_, min_row, min_col, max_row, max_col = largest_box
        results_list.append({
            "filename": filename,
            "slice": slice_,
            "min_row": min_row,
            "min_col": min_col,
            "max_row": max_row,
            "max_col": max_col
        })


# Konvertiere die Ergebnisse in einen DataFrame
results = pd.DataFrame(results_list)

# Speichere die Ergebnisse in einer CSV-Datei
results.to_csv("bounding_boxes2.csv", index=False)
print(f"Die Koordinaten der größten Bounding Boxen der alphabetisch sortierten Dateien wurden in bounding_boxes.csv gespeichert.")
