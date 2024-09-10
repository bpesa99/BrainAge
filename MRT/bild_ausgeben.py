import nibabel as nib
import numpy as np
import cv2

# NIfTI-Datei laden
nifti_image = nib.load('sub10002_s0.nii')

# Bilddaten als numpy Array extrahieren
image_data = nifti_image.get_fdata()

# Normalisieren und Konvertieren zu uint8
normalized_image_data = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX)
image_data_uint8 = normalized_image_data.astype(np.uint8)

# Optional: Wählen Sie eine bestimmte Schicht (z.B. die mittlere Schicht)
slice_index = 150
gray_image_slice = image_data_uint8[:, :, slice_index]

# Schwellenwertbildung anwenden
_, thresh = cv2.threshold(gray_image_slice, 1, 255, cv2.THRESH_BINARY)

# Konturen finden
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Größte Kontur finden (Annahme: das größte Objekt ist das Interessante)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Berechnung der Bounding Box (rechtwinklig)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Originalbild in Farbe konvertieren, um die Bounding Box zu zeichnen
    image_color = cv2.cvtColor(gray_image_slice, cv2.COLOR_GRAY2BGR)

    # Zeichnen der Bounding Box auf das Bild
    cv2.rectangle(image_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Bild anzeigen
    cv2.imshow('Bounding Box', image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Keine Konturen gefunden")
