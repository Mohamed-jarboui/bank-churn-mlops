# generate_data.py
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 10000

data = {
    'CreditScore': np.random.randint(300, 850, n_samples),
    'Age': np.random.randint(18, 80, n_samples),
    'Tenure': np.random.randint(0, 11, n_samples),
    'Balance': np.random.uniform(0, 200000, n_samples),
    'NumOfProducts': np.random.randint(1, 5, n_samples),
    'HasCrCard': np.random.choice([0, 1], n_samples),
    'IsActiveMember': np.random.choice([0, 1], n_samples),
    'EstimatedSalary': np.random.uniform(20000, 150000, n_samples),
    'Geography_Germany': np.random.choice([0, 1], n_samples),
    'Geography_Spain': np.random.choice([0, 1], n_samples),
}

# Target : plus de chance de partir si inactif, peu de produits, etc.
churn_prob = (
    (1 - data['IsActiveMember']) * 0.3 +
    (data['NumOfProducts'] == 1) * 0.2 +
    (data['Age'] > 60) * 0.15 +
    (data['Balance'] == 0) * 0.25
)
data['Exited'] = (np.random.random(n_samples) < churn_prob).astype(int)

df = pd.DataFrame(data)
df.to_csv('data/bank_churn.csv', index=False)
print(f"Dataset cree : {len(df)} lignes")
print(f"Taux de churn : {df['Exited'].mean():.2%}")