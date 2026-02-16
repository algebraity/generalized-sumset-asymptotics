import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_int_selection(raw_value, valid_values):
    raw_value = raw_value.strip().lower()
    if raw_value in {"", "all"}:
        return set(valid_values)

    pieces = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    selected = set()
    for piece in pieces:
        value = int(piece)
        if value not in valid_values:
            raise ValueError(f"{value} is not a valid option")
        selected.add(value)
    return selected


def fit_cd_parameters(n_values, y_values):
    x_log = np.log(n_values) / n_values
    x_inv = 1.0 / n_values
    design = np.column_stack((x_log, -x_inv))
    target = 0.5 - y_values
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    c_fit = float(coefficients[0])
    d_fit = float(coefficients[1])
    return c_fit, d_fit


def model(n_values, c_value, d_value):
    n_values = np.array(n_values, dtype=float)
    return 0.5 - c_value * np.log(n_values) / n_values + d_value / n_values


def r_squared(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


csv_path = Path("data/sets_data.csv")
if not csv_path.exists():
    raise FileNotFoundError("Could not find data/sets_data.csv")

rows = []
with csv_path.open("r", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        d_val = int(row["d"])
        r_val = int(row["r"])
        n_val = int(row["n"])
        a_size = float(row["|A|"])
        aa_size = float(row["|A+A|"])
        y_data = aa_size / (a_size ** 2)
        rows.append((d_val, r_val, n_val, y_data))

if not rows:
    raise ValueError("CSV file is empty")

available_d = sorted({row[0] for row in rows})
available_r = sorted({row[1] for row in rows})

print("Available d values:", ", ".join(str(value) for value in available_d))
print("Available r values:", ", ".join(str(value) for value in available_r))
print("Enter comma-separated values (e.g. 2,4,6) or 'all'.")

selected_d = parse_int_selection(input("Select d values: "), set(available_d))
selected_r = parse_int_selection(input("Select r values: "), set(available_r))

grouped = defaultdict(list)
for d_val, r_val, n_val, y_data in rows:
    if d_val in selected_d and r_val in selected_r:
        grouped[(d_val, r_val)].append((n_val, y_data))

if not grouped:
    raise ValueError("No rows found for the selected (d, r) combinations")

plt.figure(figsize=(12, 7))

for (d_val, r_val), points in sorted(grouped.items()):
    points.sort(key=lambda item: item[0])
    n_values = np.array([item[0] for item in points], dtype=float)
    y_values = np.array([item[1] for item in points], dtype=float)
    c_fit, d_fit = fit_cd_parameters(n_values, y_values)
    fit_values = model(n_values, c_fit, d_fit)
    fit_r2 = r_squared(y_values, fit_values)

    line = plt.plot(
        n_values,
        fit_values,
        linewidth=2,
        label=(
            f"d={d_val}, r={r_val}: "
            f"y=1/2-{c_fit:.8f}log(n)/n+{d_fit:.8f}/n, R^2={fit_r2:.8f}"
        ),
    )[0]
    plt.scatter(
        n_values,
        y_values,
        s=20,
        alpha=0.6,
        color=line.get_color(),
    )

plt.xlabel("n")
plt.ylabel("y")
plt.title("Best-fit curves: y_{d,r} = 1/2 - Clog(n)/n + D/n")
plt.axhline(y=0.5, color="black", linestyle="--", linewidth=1.5, label="y=1/2")
plt.legend(loc="best", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_file = Path("data/plots") / "sets_data_best_fit.png"
output_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_file, dpi=150)
print(f"Plot saved to {output_file}")

plt.show()
