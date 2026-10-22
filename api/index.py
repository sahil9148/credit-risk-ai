import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_db
from app.main import app, _seed_admin

# Guarantee database tables and admin user exist on lambda cold start
try:
    init_db()
    _seed_admin()
except Exception as e:
    print(f"Lambda cold-start database initialization: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
