def test_liveness_is_process_only(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_dependency_state(client, monkeypatch):
    monkeypatch.setattr("app.api.health._check_database", lambda: None)
    monkeypatch.setattr("app.api.health._check_redis", lambda: None)
    monkeypatch.setattr("app.api.health._check_qdrant", lambda: None)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readiness_returns_503_when_dependency_is_down(client, monkeypatch):
    monkeypatch.setattr("app.api.health._check_database", lambda: None)
    monkeypatch.setattr("app.api.health._check_redis", lambda: None)

    def down():
        raise TimeoutError("qdrant timeout")

    monkeypatch.setattr("app.api.health._check_qdrant", down)
    response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["qdrant"]["status"] == "down"
