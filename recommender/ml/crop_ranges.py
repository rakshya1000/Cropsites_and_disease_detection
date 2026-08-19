"""
Crop parameter ranges for AgroSmart synthetic dataset generation.

Each crop's N/P/K values are anchored to real, cited Nepal-specific (or 
regional, where noted) agronomic studies -- see references.md for sources.
Temperature/humidity/rainfall/pH ranges reflect the actual growing regions 
of Nepal where each crop is cultivated.

Format: (min, typical/mean, max)
"""

CROP_RANGES = {
    "Rice": {
        "N": (60, 80, 100), "P": (20, 25, 30), "K": (20, 25, 30),
        "temperature": (22, 27, 32), "humidity": (75, 82, 90),
        "ph": (5.5, 6.3, 7.0), "rainfall": (1500, 2000, 2500),
        "source": "NARC/Krishipatrika Nepal general recommendation"
    },
    "Maize": {
        "N": (100, 130, 150), "P": (50, 58, 65), "K": (35, 42, 50),
        "temperature": (18, 23, 27), "humidity": (55, 65, 75),
        "ph": (5.8, 6.5, 7.2), "rainfall": (600, 850, 1100),
        "source": "Nepal mid-hills studies (Khumaltar, Rampur)"
    },
    "Wheat": {
        "N": (100, 125, 150), "P": (25, 38, 50), "K": (25, 38, 50),
        "temperature": (12, 17, 22), "humidity": (55, 62, 70),
        "ph": (5.9, 6.0, 7.0), "rainfall": (400, 600, 800),
        "source": "Lalitpur, Nepal (PLOS One study)"
    },
    "Potato": {
        "N": (80, 100, 120), "P": (80, 100, 120), "K": (40, 60, 80),
        "temperature": (15, 18, 22), "humidity": (65, 75, 85),
        "ph": (5.0, 5.8, 6.5), "rainfall": (500, 700, 900),
        "source": "Bajhang district, Nepal (RDF)"
    },
    "Mustard": {
        "N": (40, 50, 65), "P": (30, 37, 45), "K": (30, 37, 45),
        "temperature": (10, 18, 25), "humidity": (40, 55, 65),
        "ph": (6.0, 6.8, 7.5), "rainfall": (300, 450, 600),
        "source": "Regional oilseed studies (irrigated mustard)"
    },
    "Lentil": {
        "N": (15, 22, 30), "P": (20, 35, 50), "K": (20, 35, 50),
        "temperature": (15, 20, 25), "humidity": (45, 55, 65),
        "ph": (6.0, 6.8, 7.5), "rainfall": (300, 450, 600),
        "source": "Nepal (Kailali) + regional rainfed trials"
    },
    "Millet": {
        "N": (30, 45, 60), "P": (15, 22, 30), "K": (15, 22, 30),
        "temperature": (22, 27, 32), "humidity": (40, 52, 65),
        "ph": (5.5, 6.2, 7.0), "rainfall": (400, 600, 800),
        "source": "Finger millet standard, Nepal midhills context"
    },
    "Barley": {
        "N": (25, 42, 60), "P": (15, 22, 30), "K": (8, 11, 15),
        "temperature": (10, 15, 20), "humidity": (50, 58, 65),
        "ph": (6.0, 7.0, 7.8), "rainfall": (300, 450, 600),
        "source": "Nepal-specific (Arthik Sandesh/Krishi Suchana)"
    },
    "Sugarcane": {
        "N": (100, 120, 140), "P": (40, 55, 65), "K": (30, 40, 50),
        "temperature": (24, 29, 34), "humidity": (70, 78, 85),
        "ph": (6.0, 6.8, 7.5), "rainfall": (1000, 1400, 1800),
        "source": "Nepal blanket recommendation"
    },
    "Ginger": {
        "N": (60, 75, 90), "P": (40, 50, 60), "K": (40, 50, 60),
        "temperature": (20, 25, 30), "humidity": (75, 82, 90),
        "ph": (5.5, 6.2, 6.8), "rainfall": (1200, 1700, 2200),
        "source": "Salyan, Nepal trial"
    },
    "Tea": {
        "N": (100, 120, 140), "P": (60, 80, 100), "K": (100, 120, 140),
        "temperature": (15, 22, 28), "humidity": (75, 82, 90),
        "ph": (4.2, 5.0, 5.7), "rainfall": (1800, 2400, 3000),
        "source": "Nepal (Krishi Diary)"
    },
    "Buckwheat": {
        "N": (20, 25, 30), "P": (12, 15, 18), "K": (15, 22, 30),
        "temperature": (10, 15, 20), "humidity": (45, 55, 65),
        "ph": (5.0, 6.0, 7.0), "rainfall": (400, 600, 800),
        "source": "International (Cornell) -- no Nepal-specific found"
    },
    "Cardamom": {
        "N": (60, 75, 90), "P": (60, 75, 90), "K": (120, 150, 160),
        "temperature": (15, 20, 25), "humidity": (80, 88, 95),
        "ph": (5.0, 5.8, 6.5), "rainfall": (2000, 2700, 3500),
        "source": "Nepal (Taplejung/Ilam), elevation 900-2100m"
    },
}
