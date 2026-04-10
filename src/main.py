from fastapi import FastAPI

app = FastAPI(
    title="AI Knowledge Agent",
    description="AI-powered documentation audit tool",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/health")
def health():
    return {"status": "ok"}
