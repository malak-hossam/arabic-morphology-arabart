"""Backward-compatible API entrypoint."""

from src.api.app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
