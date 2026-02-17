import uvicorn
from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings
from app.utils.logger import logger

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG CSV Expert")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to RAG CSV Expert API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
