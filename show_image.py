import os
import nibabel as nib
import matplotlib.pyplot as plt
import time

# Pfad zum Ordner, der die .nii.gz Dateien enthält
source_folder = './scaled_images_interarea'  # Ersetze dies mit deinem Ordnerpfad

# Suche nach der ersten .nii.gz Datei im Ordner
for filename in os.listdir(source_folder):
    if filename.endswith('.nii.gz'):
        # Vollständiger Pfad zur Datei
        file_path = os.path.join(source_folder, filename)

        # Lade die NIfTI-Datei
        img = nib.load(file_path)
        img_data = img.get_fdata()

        # Gib die Shape der NIfTI-Datei aus
        print(f"Die Shape der Datei {filename} ist: {img_data.shape}")

        # Iteriere durch jede Schicht und zeige sie an
        for i in range(img_data.shape[2]):
            # Extrahiere die aktuelle Schicht
            current_slice = img_data[:, :, i]

            # Plotten der aktuellen Schicht
            plt.imshow(current_slice.T, cmap='gray', origin='lower')
            plt.title(f"Schicht {i} der Datei {filename}")
            plt.axis('off')  # Schalte die Achsen aus
            plt.draw()  # Zeichne das Bild
            plt.pause(0.5)  # Pausiere für eine Sekunde

        # Nachdem alle Schichten angezeigt wurden, schließe die Plots
        plt.close()

        break
else:
    print("Keine .nii.gz Datei im Ordner gefunden.")
