import json, os, sys
# Ensure project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.app import app

# Use Flask test client
with app.test_client() as client:
    # Health check
    health = client.get('/api/health')
    print('Health:', health.status_code, health.get_json())
    # Model info
    info = client.get('/api/model-info')
    print('Model info:', info.status_code, info.get_json())
    # Dummy prediction payload matching XGBoost schema
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "firstClass",
        "agency": "FlyingDrops",
        "time": 1.76,
        "distance": 676.53,
        "flight_year": 2019,
        "flight_month": 9,
        "flight_day": 26,
        "flight_dayofweek": 3,
    }
    resp = client.post('/api/predict', json=payload)
    print('Predict:', resp.status_code, resp.get_json())
