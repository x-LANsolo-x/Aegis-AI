from fastapi.testclient import TestClient
from services.api.app.main import app, APP_VERSION

client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # Contract test: ensure the schema is as expected
    assert response.json() == {"status": "ok"}


def test_version_endpoint():
    """Test the version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": APP_VERSION}


def test_unknown_endpoint_returns_404():
    """Negative test: ensure an unknown endpoint returns 404 Not Found."""
    response = client.get("/nonexistent-endpoint")
    assert response.status_code == 404
