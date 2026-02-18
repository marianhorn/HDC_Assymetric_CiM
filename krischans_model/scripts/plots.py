import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")

# Accuracy vs M
plt.figure()
for D in sorted(df["D"].unique()):
    sub = df[df["D"] == D]
    plt.plot(sub["M"], sub["Accuracy"], marker='o', label=f"D={D}")

plt.title("Accuracy vs CIM Levels (M)")
plt.xlabel("M")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.legend()
plt.savefig("results/plot_accuracy_vs_M.png", dpi=180)

# Accuracy vs D
plt.figure()
for M in sorted(df["M"].unique()):
    sub = df[df["M"] == M]
    plt.plot(sub["D"], sub["Accuracy"], marker='o', label=f"M={M}")

plt.title("Accuracy vs Vector Length (D)")
plt.xlabel("D")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.legend()
plt.savefig("results/plot_accuracy_vs_D.png", dpi=180)

print("Plots saved.")
