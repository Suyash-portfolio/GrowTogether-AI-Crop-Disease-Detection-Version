from fastapi import FastAPI
from app.routes import predict
import uvicorn

app = FastAPI(title="GrowTogether-AI", version="1.0.0")

# Include Routes
app.include_router(predict.router)

@app.get("/")
async def root():
    return {"message": "GrowTogether-AI Production API is Live"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
