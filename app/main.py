from fastapi import FastAPI
from app.api.routes import router


app = FastAPI(title="Hybrid RAG Pipeline API", version="1.0.0")
app.include_router(router)


@app.get("/")
async def root():
    return {"message": "Hybrid RAG Pipeline API", "docs": "/docs"}