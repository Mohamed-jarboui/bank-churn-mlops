import matplotlib
matplotlib.use("Agg")  # OBLIGATOIRE pour Docker / Azure

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def detect_drift(reference_file, production_file, threshold=0.05, output_dir="drift_reports"):
    """
    Détecte le drift entre les données de référence et de production.
    Génère un rapport JSON horizontal.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(reference_file) or not os.path.exists(production_file):
        print(f"⚠️ Fichiers manquants : Reference={os.path.exists(reference_file)}, Production={os.path.exists(production_file)}")
        return {}

    ref = pd.read_csv(reference_file)
    prod = pd.read_csv(production_file)

    results = {}

    for col in ref.columns:
        # On ignore la cible 'Exited' et on traite les colonnes présentes dans les deux sets
        if col != "Exited" and col in prod.columns and ref[col].dtype in [np.float64, np.int64]:
            stat, p = ks_2samp(ref[col].dropna(), prod[col].dropna())
            results[col] = {
                "p_value": float(p),
                "statistic": float(stat),
                "drift_detected": bool(p < threshold)
            }

    # Sauvegarde du rapport JSON
    report_path = f"{output_dir}/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Rapport de drift généré: {report_path}")
    return results
