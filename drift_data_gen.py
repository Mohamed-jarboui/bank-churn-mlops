import pandas as pd
import numpy as np
import os

def generate_drifted_data(
    reference_file="data/bank_churn.csv",
    output_file="data/production_data.csv",
    drift_level="medium"
):
    """
    Génère des données de production avec drift artificiel
    
    drift_level:
    - low    : bruit léger
    - medium : décalage significatif
    - high   : changement fort
    """

    os.makedirs("data", exist_ok=True)

    if not os.path.exists(reference_file):
        print(f"❌ Erreur: {reference_file} non trouvé.")
        return

    ref = pd.read_csv(reference_file)
    prod = ref.copy()

    np.random.seed(42)

    drift_map = {
        "low": 0.05,
        "medium": 0.15,
        "high": 0.30
    }

    intensity = drift_map.get(drift_level, 0.15)

    drift_features = [
        "CreditScore",
        "Age",
        "Balance",
        "EstimatedSalary"
    ]

    for col in drift_features:
        if col in prod.columns:
            std = prod[col].std()
            prod[col] = prod[col] + np.random.normal(
                loc=std * intensity,
                scale=std * intensity,
                size=len(prod)
            )

    # Nettoyage minimal pour rester réaliste
    if 'Age' in prod.columns:
        prod['Age'] = prod['Age'].clip(18, 100)
    if 'CreditScore' in prod.columns:
        prod['CreditScore'] = prod['CreditScore'].clip(300, 850)

    prod.to_csv(output_file, index=False)

    print(f"✅ Données de production générées avec drift '{drift_level}'")
    print(f"📁 Fichier : {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["low", "medium", "high"], default="medium")
    args = parser.parse_args()
    
    generate_drifted_data(drift_level=args.level)
