from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.db.database import engine, Base
from app.services.ml_service import model_service
from app.api.auth_routes import router as auth_router
from app.api.predict_routes import router as predict_router

# The frontend directory is now one level up from the backend directory
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Load ML model into memory once
    model_service.load()
    print("✅ ML model loaded")

    yield  # app runs here

    # ── Shutdown ─────────────────────────────────────────────
    await engine.dispose()


app = FastAPI(
    title="Autism Screening API",
    version="2.0.0",
    description="AQ-10 based autism screening with async FastAPI + PostgreSQL",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(auth_router)
app.include_router(predict_router)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(str(FRONTEND_DIR / "dashboard.html"))


@app.get("/{full_path:path}", include_in_schema=False)
async def catch_all(full_path: str):
    """Serve frontend index for any unmatched route (SPA fallback)."""
    file = FRONTEND_DIR / full_path
    if file.exists() and file.is_file():
        return FileResponse(str(file))
    return FileResponse(str(FRONTEND_DIR / "index.html"))
