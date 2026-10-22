import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.database import engine, init_db, SessionLocal, Base
from app.models import User
from app.services.auth_service import get_password_hash
from app.routers import auth, applications, assessments, admin, pages


def _seed_admin():
    """Create default admin user if it doesn't exist."""
    try:
        db = SessionLocal()
        admin_user = db.query(User).filter(User.email == "admin@creditrisk.ai").first()
        if not admin_user:
            hashed_pw = get_password_hash("admin123")
            new_admin = User(
                email="admin@creditrisk.ai",
                full_name="System Admin",
                hashed_password=hashed_pw,
                is_admin=True
            )
            db.add(new_admin)
            db.commit()
        db.close()
    except Exception as e:
        print(f"Admin seeding notice: {e}")

# Eager startup: execute immediately upon module load for serverless environments
try:
    init_db()
    _seed_admin()
except Exception as e:
    print(f"Eager DB initialization notice: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks for long-running servers."""
    try:
        init_db()
        _seed_admin()
    except Exception as e:
        print(f"Lifespan initialization error: {e}")
    yield


app = FastAPI(
    title="Credit Risk AI",
    description="AI-powered credit risk assessment platform",
    version="1.0.0",
    lifespan=lifespan
)

# Exception handling middleware to log full traceback for serverless debugging
@app.middleware("http")
async def db_error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        import traceback
        err_msg = traceback.format_exc()
        print(f"SERVER ERROR ON {request.method} {request.url.path}:\n{err_msg}")
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            f"<html><body style='font-family:sans-serif;padding:30px;'><h2>Internal Server Error</h2><p>An unexpected error occurred while processing your request:</p><pre style='background:#f8d7da;color:#721c24;padding:15px;border-radius:6px;overflow:auto;'>{err_msg}</pre><p><a href='/'>&larr; Return to Home</a></p></body></html>",
            status_code=500
        )

base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(pages.router)
app.include_router(auth.router)
app.include_router(applications.router)
app.include_router(assessments.router)
app.include_router(admin.router)


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc):
    return RedirectResponse("/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
