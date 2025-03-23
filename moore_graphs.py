#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Column names, assuming both files have the same column structure:
    col_names = [
        "p_script",      # Column 0
        "M_script",      # Column 1
        "nproc",         # Column 2
        "N",             # Column 3
        "p_code",        # Column 4
        "M_code",        # Column 5
        "avgSteps",      # Column 6
        "avgTime",       # Column 7,
        "bottomFraction" # Column 8
    ]

    # 1) Read the two files
    # moore_data.txt => Moore neighborhood
    # convergence_data.txt => Von Neumann neighborhood
    df_moore = pd.read_csv("moore_data.txt", sep=" ", header=None, names=col_names)
    df_von = pd.read_csv("convergence_data.txt", sep=" ", header=None, names=col_names)

    # 2) Plot: Average Steps vs. M for each p, combining both methods on one plot
    plt.figure(figsize=(7,5))

    # -- Moore data --
    for p_val, grp in df_moore.groupby("p_script"):
        grp_sorted = grp.sort_values(by="M_script")
        plt.plot(
            grp_sorted["M_script"],
            grp_sorted["avgSteps"],
            marker='o',
            label=f"Moore, p={p_val}"
        )

    # -- Von Neumann data --
    for p_val, grp in df_von.groupby("p_script"):
        grp_sorted = grp.sort_values(by="M_script")
        plt.plot(
            grp_sorted["M_script"],
            grp_sorted["avgSteps"],
            marker='^',
            linestyle='--',
            label=f"Von Neumann, p={p_val}"
        )

    plt.xlabel("Number of Repeats (M)")
    plt.ylabel("Average Steps")
    plt.title("Forest Fire Convergence (Steps vs. M)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("combined_convergence_steps.png", dpi=150)
    plt.show()

    print("Plot saved to: combined_convergence_steps.png")

if __name__ == "__main__":
    main()