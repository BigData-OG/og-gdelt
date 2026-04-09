from fastapi import FastAPI
from train_routes import router as train_router

app = FastAPI()

app.include_router(train_router)
