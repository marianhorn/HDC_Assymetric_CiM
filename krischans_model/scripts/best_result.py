import csv
import sys

# Pfad zur CSV-Datei aus Argument oder Default
path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"

best_row = None
best_acc = float("-inf")

with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            acc = float(row["Accuracy"])
        except (KeyError, ValueError):
            continue  # falls kaputte Zeile
        if acc > best_acc:
            best_acc = acc
            best_row = row

if best_row is not None:
    # Ausgabe im Stil "D,M,Accuracy"
    print(f'{best_row["D"]},{best_row["M"]},{best_row["Accuracy"]}')
else:
    print("Keine gültigen Daten in", path)
