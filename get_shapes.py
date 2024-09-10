import os
import nibabel as nib

# Pfad zum Ordner, der die .nii.gz Dateien enthält
source_folder = './cropped_images'  # Ersetze dies mit deinem Ordnerpfad

# Durchlaufe alle Dateien im Quellordner
for filename in os.listdir(source_folder):
    if filename.endswith('.nii.gz'):
        # Vollständiger Pfad zur Datei
        file_path = os.path.join(source_folder, filename)

        # Lade die NIfTI-Datei
        img = nib.load(file_path)
        img_data = img.get_fdata()

        # Gib die Shape der NIfTI-Datei aus
        print(f"Die Shape der Datei {filename} ist: {img_data.shape}")
