from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from app.api.v1.routes import router as v1_router


app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

origins = [
    "http://localhost:8082",  # your frontend
]


# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

app.include_router(v1_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host="127.0.0.1")
