# 🧪 Bank Churn API - Testing Guide

Complete guide for testing your Bank Churn Prediction API locally.

---

## 📋 Prerequisites

1. **Activate virtual environment** (if using one):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify model exists**:
   - Check that `model/churn_model.pkl` exists (✅ Already present)

---

## 🚀 Starting the API Server

### Method 1: Basic Start
```powershell
uvicorn app.main:app --reload --port 8000
```

### Method 2: With Custom Host
```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

---

## 🧪 Testing with curl (PowerShell)

### 1. Health Check
```powershell
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### 2. Simple Prediction
```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
  }'
```

**Expected Response:**
```json
{
  "churn_probability": 0.3245,
  "prediction": 0,
  "risk_level": "Medium"
}
```

---

### 3. Root Endpoint
```powershell
curl http://localhost:8000/
```

**Expected Response:**
```json
{
  "message": "Bank Churn Prediction API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs"
}
```

---

## 📓 Testing with Jupyter Lab

### 1. Install Jupyter (if not already installed)
```powershell
pip install jupyterlab
```

### 2. Start Jupyter Lab
```powershell
jupyter lab
```

### 3. Create a New Notebook and Run:

```python
import requests
import json

# URL de l'API FastAPI
url = "http://localhost:8000/predict"

# Données à envoyer
data = {
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000,
    "Geography_Germany": 0,
    "Geography_Spain": 1
}

# Envoyer la requête POST
response = requests.post(url, json=data)

# Afficher la réponse
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
```

**Expected Output:**
```
Status Code: 200
Response: {
  "churn_probability": 0.3245,
  "prediction": 0,
  "risk_level": "Medium"
}
```

---

### 4. Test Multiple Scenarios

```python
# Test different customer profiles
test_cases = [
    {
        "name": "High Risk Customer",
        "data": {
            "CreditScore": 400,
            "Age": 45,
            "Tenure": 1,
            "Balance": 0,
            "NumOfProducts": 1,
            "HasCrCard": 0,
            "IsActiveMember": 0,
            "EstimatedSalary": 30000,
            "Geography_Germany": 1,
            "Geography_Spain": 0
        }
    },
    {
        "name": "Low Risk Customer",
        "data": {
            "CreditScore": 800,
            "Age": 30,
            "Tenure": 8,
            "Balance": 120000,
            "NumOfProducts": 3,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 100000,
            "Geography_Germany": 0,
            "Geography_Spain": 0
        }
    }
]

for test in test_cases:
    print(f"\n{'='*50}")
    print(f"Testing: {test['name']}")
    print('='*50)
    
    response = requests.post(url, json=test['data'])
    result = response.json()
    
    print(f"Churn Probability: {result['churn_probability']:.2%}")
    print(f"Prediction: {'CHURN' if result['prediction'] == 1 else 'NO CHURN'}")
    print(f"Risk Level: {result['risk_level']}")
```

---

## 🌐 Interactive API Documentation

FastAPI automatically generates interactive documentation:

1. **Swagger UI**: http://localhost:8000/docs
   - Interactive interface to test all endpoints
   - Automatic request/response validation

2. **ReDoc**: http://localhost:8000/redoc
   - Alternative documentation format
   - Better for reading API specifications

---

## 🔍 Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint - API info |
| `GET` | `/health` | Health check - verify API status |
| `POST` | `/predict` | Single prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `POST` | `/drift/check` | Check for data drift |
| `POST` | `/drift/alert` | Manual drift alert |

---

## 🐛 Troubleshooting

### Issue: "Model not loaded"
**Solution:** Check that `model/churn_model.pkl` exists in your project

### Issue: "Module not found"
**Solution:** Install dependencies:
```powershell
pip install -r requirements.txt
```

### Issue: Port 8000 already in use
**Solution:** Use a different port:
```powershell
uvicorn app.main:app --reload --port 8001
```

### Issue: curl not recognized (Windows)
**Solution:** Use PowerShell's `Invoke-RestMethod`:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/health -Method GET
```

Or install curl:
```powershell
winget install curl
```

---

## ✅ Success Checklist

- [ ] API starts without errors
- [ ] `/health` endpoint returns `{"status": "healthy", "model_loaded": true}`
- [ ] `/predict` endpoint returns valid predictions
- [ ] Interactive docs accessible at `/docs`
- [ ] Jupyter Lab can connect and get predictions

---

## 📊 Sample Test Data

Here are more test cases for comprehensive testing:

### Young Active Customer
```json
{
  "CreditScore": 720,
  "Age": 28,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 85000,
  "Geography_Germany": 0,
  "Geography_Spain": 0
}
```

### Senior Inactive Customer
```json
{
  "CreditScore": 550,
  "Age": 60,
  "Tenure": 10,
  "Balance": 0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 0,
  "EstimatedSalary": 40000,
  "Geography_Germany": 1,
  "Geography_Spain": 0
}
```

---

## 🎯 Next Steps

1. ✅ **Test the API locally** using this guide
2. **Deploy to production** (Azure, AWS, etc.)
3. **Set up monitoring** with Azure Application Insights
4. **Create automated tests** in the `tests/` directory
5. **Configure CI/CD** for automatic deployment

Happy Testing! 🚀
