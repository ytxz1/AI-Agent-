# FastAPI 路由与请求处理实战任务

## 1. 实战目标

本实战的目标是完成一个“内存版商品管理 API”。它不连接数据库，数据保存在 Python 字典里。虽然很简单，但它能覆盖 FastAPI 入门最关键的能力：

1. 创建应用。
2. 创建路由。
3. 处理路径参数。
4. 处理查询参数。
5. 处理请求体。
6. 返回 JSON。
7. 抛出 HTTP 错误。
8. 使用 `/docs` 测试接口。

最终你要实现这些接口：

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | `/` | 欢迎页 |
| GET | `/health` | 健康检查 |
| GET | `/items` | 商品列表 |
| GET | `/items/{item_id}` | 商品详情 |
| POST | `/items` | 创建商品 |
| PUT | `/items/{item_id}` | 更新商品 |
| DELETE | `/items/{item_id}` | 删除商品 |

## 2. 准备项目

进入本目录：

```powershell
cd "D:\vscode项目\AI Agent 开发工程师学习路线图（工程落地版）\第 1 周：大模型应用开发基础 + 手撕 Naive RAG\第1天FastAPI 快速入门\fastapi-basic-demo"
```

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install fastapi uvicorn
```

启动：

```powershell
python -m uvicorn main:app --reload
```

验证：

1. 打开 `http://127.0.0.1:8000/`
2. 打开 `http://127.0.0.1:8000/docs`

如果 PowerShell 中出现 `Fatal error in launcher: Unable to create process using ...`，说明 `pip.exe` 或 `uvicorn.exe` 启动器路径异常。不要直接运行 `pip` 或 `uvicorn`，改用：

```powershell
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

## 3. 逐步实现

### 任务 1：最小应用

```python
from fastapi import FastAPI

app = FastAPI(title="FastAPI Basic Demo")


@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}
```

检查点：

1. `/` 返回 JSON。
2. `/docs` 出现接口文档。

### 任务 2：健康检查

```python
@app.get("/health")
def health_check():
    return {"status": "ok"}
```

理解重点：

1. 这个接口不需要参数。
2. 它适合被部署平台、监控系统或前端用来判断服务是否可用。
3. 后续 RAG 服务也应该保留类似接口。

### 任务 3：定义请求体模型

```python
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    price: float
    description: str | None = None
    is_offer: bool = False
```

理解重点：

1. `name: str` 是必填字段。
2. `price: float` 是必填字段。
3. `description: str | None = None` 是可选字段。
4. `is_offer: bool = False` 有默认值。
5. `Item` 可以作为 POST/PUT 请求体。

### 任务 4：准备内存数据

```python
items_db: dict[int, Item] = {
    1: Item(name="Keyboard", price=199.0, description="Mechanical keyboard"),
    2: Item(name="Mouse", price=89.0),
}
```

### 任务 5：商品列表

```python
@app.get("/items")
def list_items(skip: int = 0, limit: int = 10):
    items = list(items_db.items())[skip : skip + limit]
    return {
        "skip": skip,
        "limit": limit,
        "total": len(items_db),
        "items": [{"id": item_id, **item.model_dump()} for item_id, item in items],
    }
```

测试：

```text
GET /items
GET /items?skip=0&limit=1
GET /items?skip=1&limit=1
```

### 任务 6：商品详情

```python
from fastapi import HTTPException


@app.get("/items/{item_id}")
def read_item(item_id: int):
    item = items_db.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"id": item_id, **item.model_dump()}
```

测试：

```text
GET /items/1
GET /items/999
GET /items/abc
```

观察：

1. `/items/1` 返回商品。
2. `/items/999` 返回 404。
3. `/items/abc` 返回参数校验错误。

### 任务 7：创建商品

```python
@app.post("/items", status_code=201)
def create_item(item: Item):
    new_id = max(items_db.keys(), default=0) + 1
    items_db[new_id] = item
    return {"message": "created", "id": new_id, "item": item}
```

请求体：

```json
{
  "name": "Notebook",
  "price": 12.5,
  "description": "A paper notebook",
  "is_offer": false
}
```

### 任务 8：更新商品

```python
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
```

测试：

```text
PUT /items/1
PUT /items/1?notify=true
PUT /items/999
```

理解重点：

1. `item_id` 是路径参数。
2. `item` 是请求体。
3. `notify` 是查询参数。
4. 一个路由函数可以同时接收三类参数。

### 任务 9：删除商品

```python
@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    deleted = items_db.pop(item_id)
    return {"message": "deleted", "id": item_id, "item": deleted}
```

测试：

```text
DELETE /items/1
GET /items/1
```

## 4. 验收清单

完成后逐项打勾：

1. 能启动 `uvicorn main:app --reload`。
2. 能打开 `/docs`。
3. `GET /` 正常。
4. `GET /health` 正常。
5. `GET /items` 正常。
6. `GET /items?skip=0&limit=1` 正常。
7. `GET /items/1` 正常。
8. `GET /items/999` 返回 404。
9. `GET /items/abc` 返回参数校验错误。
10. `POST /items` 能创建商品。
11. 创建商品后，`GET /items` 能看到新商品。
12. `PUT /items/{item_id}` 能更新商品。
13. `PUT /items/{item_id}?notify=true` 能接收查询参数。
14. `DELETE /items/{item_id}` 能删除商品。
15. 删除后再次查询返回 404。

## 5. 加分练习

如果你已经完成基础任务，可以继续做这些：

1. 给 `limit` 增加最大值限制，例如最多 100。
2. 给 `price` 增加必须大于 0 的校验。
3. 增加 `category` 字段。
4. 增加 `GET /items/search?q=xxx` 搜索接口。
5. 增加 `PATCH /items/{item_id}` 局部更新接口。
6. 把数据存储从内存字典换成 JSON 文件。
7. 把路由拆到 `routers/items.py`。
8. 增加响应模型 `response_model`。

## 6. 今日复盘模板

学习结束后，建议写下这些答案：

1. 今天我学会了哪些 FastAPI 基础概念？
2. 路径参数、查询参数、请求体分别是什么？
3. FastAPI 是如何根据函数参数判断请求来源的？
4. Pydantic 模型解决了什么问题？
5. `/docs` 对开发调试有什么帮助？
6. 我在哪个错误上卡住最久？
7. 如果要把一个 RAG 问答功能包装成 API，我会设计什么路由？
