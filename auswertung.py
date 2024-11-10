import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV-Datei laden
df = pd.read_csv('compromised_age_comparison_depression.csv')

# Differenz und Mittelwert berechnen
df['Differenz'] = df['Mean Predicted Age'] - df['True Age']
df['Mittelwert'] = (df['Mean Predicted Age'] + df['True Age']) / 2

coefficients = np.polyfit(df['Mittelwert'], df['Differenz'], 1)
trendlinie = np.poly1d(coefficients)

steigung = coefficients[0]
achsenabschnitt = coefficients[1]
x_schnittpunkte = -achsenabschnitt/steigung
winkel = np.degrees(np.arctan(steigung))

print(f"Die Trendlinie schneidet die x-Achse bei: {x_schnittpunkte:.4f}")
print(f"Der Winkel der Trendlinie gegen die x-Achse beträgt: {winkel:.2f}°")

x_line = np.linspace(df['Mittelwert'].min(), df['Mittelwert'].max(), 100)
y_line = trendlinie(x_line)

# Zeile mit dem größten absoluten Fehler finden
max_error_row = df.loc[df['Absolute Error'].idxmax()]
true_age_max_error = max_error_row['True Age']
predicted_age_max_error = max_error_row['Mean Predicted Age']
absolute_error_max = max_error_row['Absolute Error']

# Ausgabe des größten absoluten Fehlers
print(f"Größter absoluter Fehler: {absolute_error_max}")
print(f"True Age: {true_age_max_error}")
print(f"Predicted Age: {predicted_age_max_error}")

# Erster Plot: Differenz vs. Mittelwert
plt.figure(figsize=(10, 10))
plt.scatter(df['Mittelwert'], df['Differenz'], alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.plot(x_line, y_line, color='red', label='Trendlinie')
plt.xlabel('Mittelwert von True Age und Predicted Age')
plt.ylabel('Differenz (Predicted Age - True Age)')
plt.title('Differenz über Mittelwert von True Age und Predicted Age')
plt.grid(True)
plt.show()

# Zweiter Plot: True Age vs. Predicted Age
plt.figure(figsize=(10, 10))
plt.scatter(df['True Age'], df['Mean Predicted Age'], alpha=0.6)
plt.plot([df['True Age'].min(), df['True Age'].max()], [df['True Age'].min(), df['True Age'].max()], color='red', linestyle='--', linewidth=1)
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('True Age vs. Predicted Age')
plt.grid(True)
plt.show()
