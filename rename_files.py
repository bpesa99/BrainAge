import os

# Pfad zum Ordner
ordner_pfad = "scaled_images_interarea"

# Gehe durch alle Dateien im Ordner
for datei_name in os.listdir(ordner_pfad):
    # Überprüfen, ob der Dateiname mit "cropped" beginnt
    if datei_name.startswith("cropped"):
        # Neuen Dateinamen erstellen, indem "cropped" durch "scaled" ersetzt wird
        neuer_datei_name = datei_name.replace("cropped", "scaled2", 1)

        # Alte und neue Pfade für die Umbenennung erstellen
        alter_pfad = os.path.join(ordner_pfad, datei_name)
        neuer_pfad = os.path.join(ordner_pfad, neuer_datei_name)

        # Datei umbenennen
        os.rename(alter_pfad, neuer_pfad)
        print(f"Umbenannt: {datei_name} -> {neuer_datei_name}")

print("Alle relevanten Dateien wurden umbenannt.")
