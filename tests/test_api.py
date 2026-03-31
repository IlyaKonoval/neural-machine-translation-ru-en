import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestTranslateEndpoint:
    def test_translate_without_model_returns_503(self, client):
        resp = client.post("/translate", json={"text": "привет"})
        assert resp.status_code == 503

    def test_translate_empty_text_returns_422(self, client):
        resp = client.post("/translate", json={"text": ""})
        assert resp.status_code == 422

    def test_translate_invalid_beam_size(self, client):
        resp = client.post("/translate", json={"text": "привет", "beam_size": 0})
        assert resp.status_code == 422

        resp = client.post("/translate", json={"text": "привет", "beam_size": 100})
        assert resp.status_code == 422
