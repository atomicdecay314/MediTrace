import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from database import Base, engine
from routers.documents import router as documents_router
from routers.timeline import router as timeline_router
from routers.interview import router as interview_router
from routers.session import router as session_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    # Phase 6A migration: add manually_edited column if DB predates this phase
    from sqlalchemy import inspect as sa_inspect, text
    with engine.connect() as conn:
        cols = [c["name"] for c in sa_inspect(engine).get_columns("events")]
        if "manually_edited" not in cols:
            conn.execute(text(
                "ALTER TABLE events ADD COLUMN manually_edited BOOLEAN NOT NULL DEFAULT 0"
            ))
            conn.commit()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    yield


app = FastAPI(title="MediTrace", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router)
app.include_router(interview_router)
app.include_router(documents_router)
app.include_router(timeline_router)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
