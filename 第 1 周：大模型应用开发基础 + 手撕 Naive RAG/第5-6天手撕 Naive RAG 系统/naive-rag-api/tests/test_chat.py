from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_chat_request_validation():
    response = client.post("/api/v1/chat/query", json={"question": "", "top_k": 4})

    assert response.status_code == 422

