import pandas as pd

# Lade die Bounding-Box-Koordinaten aus der CSV-Datei
bounding_boxes_df = pd.read_csv('bounding_boxes2.csv')

# Berechne die Höhe und Breite für jede Bounding Box
bounding_boxes_df['height'] = bounding_boxes_df['max_row'] - bounding_boxes_df['min_row']
bounding_boxes_df['width'] = bounding_boxes_df['max_col'] - bounding_boxes_df['min_col']

# Berechne den Mittelwert der Höhen und Breiten
mean_height = bounding_boxes_df['height'].mean()
mean_width = bounding_boxes_df['width'].mean()

# Ausgabe der Ergebnisse
print(f"Der Mittelwert der Höhen aller Bounding Boxen ist: {mean_height:.2f}")
print(f"Der Mittelwert der Breiten aller Bounding Boxen ist: {mean_width:.2f}")
print(f"Verhältnis der Kantenlängen: {(mean_height/mean_width):.2f}")
print(f"Verhältnis von 80 zu 65 ist: {(80/65):.2f}. Dies ist ungefähr das gleiche Verhältnis der mittleren Bounding Box.")