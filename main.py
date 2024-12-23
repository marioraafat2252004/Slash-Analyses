from fastapi import FastAPI
from routes.chatRouter import router as chat_router
from routes.productRouter import router as product_router

app = FastAPI(title="AI Chatbot API", version="1.0.0")

# chat routes
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(product_router, prefix="/api/product", tags=["Product"])

@app.get("/", summary="Health Check")
async def health_check():
    return {"message": "AI Chatbot API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)