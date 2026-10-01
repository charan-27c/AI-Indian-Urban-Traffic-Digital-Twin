from fastapi import FastAPI


app = FastAPI(
    title="AI Indian Urban Traffic Digital Twin",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "AI Indian Urban Traffic Digital Twin",
        "status": "running",
        "phase": 0,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }