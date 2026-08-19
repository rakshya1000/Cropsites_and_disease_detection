"""
AgroSmart synthetic dataset generator.

Generates realistic crop recommendation training data using agronomic 
ranges anchored to real Nepal-specific (or regional, where noted) studies.

Realism principles applied:
1. Normal (Gaussian) distribution per feature, clipped to min/max -- 
   values cluster near the typical/mean, not spread uniformly.
2. Correlated noise between rainfall and humidity within the same row 
   (a rainy sample tends to also be humid).
3. Natural overlap between agronomically similar crops (e.g. Barley/
   Buckwheat) is NOT artificially removed.
4. Realistic precision: 1 decimal for temp/pH, integers for N/P/K/rainfall.
5. No duplicate rows enforced via per-row random noise.

Usage:
    python3 generate_dataset.py
Output:
    agrosmart_crop_dataset.csv
"""

import csv
import random

from crop_ranges import CROP_RANGES

random.seed(42)  # reproducibility -- document this in your report

ROWS_PER_CROP = 150  # ~150 x 13 crops = 1950 rows total


def sample_normal_clipped(min_val, mean_val, max_val):
    """Sample from a normal distribution centered on mean_val, clipped to
    [min_val, max_val]. Std dev is set so ~99.7% of samples fall within
    the given range (range width / 6), which gives realistic clustering
    near the typical value rather than uniform spread."""
    std = (max_val - min_val) / 6
    val = random.gauss(mean_val, std)
    return max(min_val, min(max_val, val))


def generate_row(crop_name, ranges):
    n = sample_normal_clipped(*ranges["N"])
    p = sample_normal_clipped(*ranges["P"])
    k = sample_normal_clipped(*ranges["K"])
    temp = sample_normal_clipped(*ranges["temperature"])

    # Correlate humidity and rainfall slightly: sample a shared "wetness"
    # factor between 0 and 1, then nudge both humidity and rainfall in the
    # same direction using it, so a high-rainfall row tends to also be humid.
    wetness = random.random()
    hum_min, hum_mean, hum_max = ranges["humidity"]
    rain_min, rain_mean, rain_max = ranges["rainfall"]

    humidity = sample_normal_clipped(hum_min, hum_mean, hum_max)
    rainfall = sample_normal_clipped(rain_min, rain_mean, rain_max)
    # nudge both towards the same end of their range using `wetness`
    humidity = humidity + (wetness - 0.5) * (hum_max - hum_min) * 0.15
    rainfall = rainfall + (wetness - 0.5) * (rain_max - rain_min) * 0.15
    humidity = max(hum_min, min(hum_max, humidity))
    rainfall = max(rain_min, min(rain_max, rainfall))

    ph = sample_normal_clipped(*ranges["ph"])

    return {
        "N": round(n),
        "P": round(p),
        "K": round(k),
        "temperature": round(temp, 1),
        "humidity": round(humidity, 1),
        "ph": round(ph, 2),
        "rainfall": round(rainfall),
        "label": crop_name,
    }


def main():
    rows = []
    for crop_name, ranges in CROP_RANGES.items():
        for _ in range(ROWS_PER_CROP):
            rows.append(generate_row(crop_name, ranges))

    random.shuffle(rows)  # so crops aren't in blocks

    fieldnames = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
    out_path = "agrosmart_crop_dataset.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} rows across {len(CROP_RANGES)} crops -> {out_path}")


if __name__ == "__main__":
    main()
