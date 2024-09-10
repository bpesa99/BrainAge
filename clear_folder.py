import os
import shutil

# Pfad zum Ordner, der die .nii.gz Dateien enthält
source_folder = './bearbeitet_nifti'  # Ersetze dies mit deinem Quellordnerpfad
# Pfad zum Ordner, in den die Dateien verschoben werden sollen
destination_folder = './Auslagerungsordner'  # Ersetze dies mit deinem Zielordnerpfad

# Erstelle den Zielordner, falls er noch nicht existiert
os.makedirs(destination_folder, exist_ok=True)

# Durchlaufe alle Dateien im Quellordner
for filename in os.listdir(source_folder):
    # Prüfe, ob die Datei mit ".nii.gz" endet und "mask" im Dateinamen enthält
    if filename.endswith('.nii.gz') and 'mask' in filename:
        # Vollständiger Pfad zur Datei
        source_path = os.path.join(source_folder, filename)
        # Vollständiger Pfad zum Ziel
        destination_path = os.path.join(destination_folder, filename)

        # Datei verschieben
        shutil.move(source_path, destination_path)
        print(f'{filename} wurde nach {destination_folder} verschoben.')

print('Verschiebung abgeschlossen.')
