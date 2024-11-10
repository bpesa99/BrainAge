import pandas as pd

# 1. age_comparison.csv laden
comparison_df = pd.read_csv('age_comparison_depression.csv')

# 2. Gruppenspalte erstellen, die alle 80 Zeilen eine neue Gruppe startet
comparison_df['Group'] = (comparison_df.index // 100)

# 3. Mittelwert der Predicted Ages für jede Gruppe berechnen
compromised_df = comparison_df.groupby('Group').agg({
    'True Age': 'first',  # Nimmt den ersten True Age-Wert in der Gruppe
    'Predicted Age': 'mean'  # Berechnet den Mittelwert der Predicted Age in der Gruppe
}).reset_index(drop=True)

# 4. Spalte umbenennen für Klarheit
compromised_df.rename(columns={'Predicted Age': 'Mean Predicted Age'}, inplace=True)

# 5. Absoluten Fehler berechnen
compromised_df['Absolute Error'] = abs(compromised_df['True Age'] - compromised_df['Mean Predicted Age'])

# 6. Speichern in eine neue CSV-Datei
compromised_df.to_csv('compromised_age_comparison_depression.csv', index=False)

print("Die kompromittierten Altersvergleiche wurden in 'compromised_age_comparison_depression.csv' gespeichert.")
