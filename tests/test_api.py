from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_root():
    assert client.get("/").status_code == 200
