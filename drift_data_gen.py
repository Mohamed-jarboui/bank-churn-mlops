# drift_data_gen.py
import pandas as pd
import numpy as np
import os

def generate_production_data(reference_file="data/bank_churn.csv", output_file="data/production_data.csv", drift_level="medium"):
    """
    Génère des données de production avec un niveau de drift contrôlé.
    drift_level: "none", "low", "medium", "high"
    """
    if not os.path.exists(reference_file):
        print(f"Erreur: {reference_file} non trouvé.")
        return

    df = pd.read_csv(reference_file)
    prod_df = df.sample(n=1000, replace=True).copy()

    # Simulation de drift
    if drift_level == "low":
        # Légère augmentation de l'âge
        prod_df['Age'] = prod_df['Age'] + np.random.randint(0, 5, size=len(prod_df))
    elif drift_level == "medium":
        # Augmentation notable de l'âge et baisse du score de crédit
        prod_df['Age'] = prod_df['Age'] + np.random.randint(5, 15, size=len(prod_df))
        prod_df['CreditScore'] = prod_df['CreditScore'] - np.random.randint(20, 50, size=len(prod_df))
    elif drift_level == "high":
        # Changements massifs
        prod_df['Age'] = prod_df['Age'] + np.random.randint(15, 30, size=len(prod_df))
        prod_df['Balance'] = prod_df['Balance'] * 1.5
        prod_df['CreditScore'] = prod_df['CreditScore'] - 100

    # Nettoyage des valeurs hors limites
    prod_df['Age'] = prod_df['Age'].clip(18, 100)
    prod_df['CreditScore'] = prod_df['CreditScore'].clip(300, 850)

    # Sauvegarde
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    prod_df.to_csv(output_file, index=False)
    print(f"✅ Données de production générées ({drift_level} drift) dans {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["none", "low", "medium", "high"], default="medium")
    args = parser.parse_args()
    
    generate_production_data(drift_level=args.level)
