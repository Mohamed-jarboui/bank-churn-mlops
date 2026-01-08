import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score, 
    roc_auc_score,
    confusion_matrix
)
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration MLflow
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("bank-churn-prediction")

print("Chargement des donnees...")
df = pd.read_csv("data/bank_churn.csv")

print(f"Dataset : {len(df)} lignes, {len(df.columns)} colonnes")
print(f"Taux de churn : {df['Exited'].mean():.2%}")

# Separation features/target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain : {len(X_train)} lignes")
print(f"Test : {len(X_test)} lignes")

# Entrainement avec MLflow tracking
print("\nEntrainement du modele...")
with mlflow.start_run(run_name="random-forest-v1"):
    
    # Parametres du modele
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    
    # Entrainement
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des metriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Log des parametres et metriques dans MLflow
    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    })
    
    # Creation et sauvegarde de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Predite')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()
    
    # Enregistrement du modele dans MLflow
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="bank-churn-classifier"
    )
    
    # Sauvegarde locale du modele
    joblib.dump(model, "model/churn_model.pkl")
    
    # Tags
    mlflow.set_tags({
        "environment": "development",
        "model_type": "RandomForest",
        "task": "binary_classification"
    })
    
    # Affichage des resultats
    print("\n" + "="*50)
    print("RESULTATS DE L'ENTRAINEMENT")
    print("="*50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC AUC   : {auc:.4f}")
    print("="*50)
    
    print(f"\nModele sauvegarde dans : model/churn_model.pkl")
    print(f"MLflow UI : mlflow ui --port 5000")