import uvicorn
from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(title="RAG CSV Expert")

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to RAG CSV Expert API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
