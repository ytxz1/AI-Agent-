from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="FastAPI Basic Demo",
    description="Day 1 practice: routes, path parameters, query parameters, and request bodies.",
    version="0.1.0",
)


class Item(BaseModel):
    name: str
    price: float
    description: str | None = None
    is_offer: bool = False


items_db: dict[int, Item] = {
    1: Item(name="Keyboard", price=199.0, description="Mechanical keyboard"),
    2: Item(name="Mouse", price=89.0),
}


@app.get("/")
def read_root():
    return {
        "message": "Hello FastAPI",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/items")
def list_items(skip: int = 0, limit: int = 10):
    items = list(items_db.items())[skip : skip + limit]
    return {
        "skip": skip,
        "limit": limit,
        "total": len(items_db),
        "items": [{"id": item_id, **item.model_dump()} for item_id, item in items],
    }


@app.get("/items/{item_id}")
def read_item(item_id: int):
    item = items_db.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, **item.model_dump()}


@app.post("/items", status_code=201)
def create_item(item: Item):
    new_id = max(items_db.keys(), default=0) + 1
    items_db[new_id] = item
    return {
        "message": "created",
        "id": new_id,
        "item": item,
    }


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item, notify: bool = False):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_id] = item
    return {
        "message": "updated",
        "id": item_id,
        "notify": notify,
        "item": item,
    }


@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    deleted = items_db.pop(item_id)
    return {
        "message": "deleted",
        "id": item_id,
        "item": deleted,
    }
