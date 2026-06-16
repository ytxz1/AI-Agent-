from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    embedding_provider: str
    chat_provider: str
    vector_store_dir: str

