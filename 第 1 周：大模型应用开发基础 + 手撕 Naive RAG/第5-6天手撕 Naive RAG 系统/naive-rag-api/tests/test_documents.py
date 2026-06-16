from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_upload_unsupported_file_type():
    files = {"file": ("bad.exe", b"fake content", "application/octet-stream")}

    response = client.post("/api/v1/documents/upload", files=files)

    assert response.status_code == 400

