import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test import
try:
    from app.main import app
    print("App imported OK")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Test rendering the index page directly
from starlette.testclient import TestClient

client = TestClient(app, raise_server_exceptions=False)

# Test landing page
r = client.get("/")
print(f"\nGET / -> {r.status_code}")
if r.status_code != 200:
    print(f"Body: {r.text[:2000]}")

# Test login page
r = client.get("/login")
print(f"GET /login -> {r.status_code}")
if r.status_code != 200:
    print(f"Body: {r.text[:500]}")

# Test register page
r = client.get("/register")
print(f"GET /register -> {r.status_code}")
if r.status_code != 200:
    print(f"Body: {r.text[:500]}")

print("\nDone!")
