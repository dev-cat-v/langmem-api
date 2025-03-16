from fastapi import FastAPI
from api.routers import chat

app = FastAPI(
    title="BonBon AI",
    description="LangMEM を用いた自己意識型AI",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "BonBon AI is running!"}

app.include_router(chat.router, prefix="/chat", tags=["Chat"])

# FastAPIアプリの起動（デバッグ用）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
