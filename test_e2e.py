# -*- coding: utf-8 -*-
"""End-to-end test for Credit Risk AI application."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from starlette.testclient import TestClient
from app.main import app

results = []

def test(name, response, expected_status=200):
    ok = response.status_code == expected_status
    results.append((name, ok, response.status_code))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} -> {response.status_code}")
    if not ok:
        body = response.text[:400] if hasattr(response, 'text') else ''
        if body:
            print(f"         Body: {body}")
    return response

# Use context manager so lifespan (startup) runs properly
with TestClient(app, raise_server_exceptions=False) as client:

    # ============ PUBLIC PAGES ============
    print("\n=== PUBLIC PAGES ===")
    test("Landing page", client.get("/"))
    test("Login page", client.get("/login"))
    test("Register page", client.get("/register"))

    # ============ AUTH FLOW ============
    print("\n=== REGISTRATION & LOGIN ===")
    r = client.post("/api/auth/register", data={
        "email": "test@example.com",
        "full_name": "Test User",
        "password": "testpass123"
    }, follow_redirects=False)
    test("Register user", r, 303)

    r = client.post("/api/auth/login", data={
        "email": "test@example.com",
        "password": "testpass123"
    }, follow_redirects=False)
    test("Login user", r, 303)
    cookies = dict(r.cookies)

    # ============ AUTHENTICATED PAGES ============
    print("\n=== AUTHENTICATED PAGES ===")
    test("Dashboard", client.get("/dashboard", cookies=cookies))
    test("Apply form", client.get("/dashboard/apply", cookies=cookies))
    test("History page", client.get("/dashboard/history", cookies=cookies))

    # ============ LOAN APPLICATION ============
    print("\n=== LOAN APPLICATION & AI ASSESSMENT ===")
    r = client.post("/api/applications/", data={
        "full_name": "Test User",
        "age": "35",
        "annual_income": "75000",
        "employment_length": "8",
        "employment_type": "salaried",
        "home_ownership": "mortgage",
        "existing_debt": "15000",
        "num_credit_lines": "5",
        "credit_history_length": "10",
        "late_payments": "1",
        "bankruptcies": "0",
        "loan_amount": "50000",
        "loan_purpose": "home",
        "loan_tenure": "60"
    }, cookies=cookies, follow_redirects=False)
    test("Submit application", r, 303)

    # Update cookies from redirect
    if r.status_code == 303:
        report_url = r.headers.get("location", "/dashboard")
        print(f"         Redirected to: {report_url}")

    test("Risk report page", client.get("/dashboard/report/1", cookies=cookies))
    test("Recommendations page", client.get("/dashboard/recommendations/1", cookies=cookies))
    test("History with data", client.get("/dashboard/history", cookies=cookies))

    # ============ WHAT-IF ANALYSIS ============
    print("\n=== WHAT-IF ANALYSIS ===")
    r = client.post("/api/assessments/whatif", json={
        "age": 35, "annual_income": 120000, "employment_length": 8,
        "employment_type": "salaried", "home_ownership": "mortgage",
        "existing_debt": 5000, "num_credit_lines": 5, "credit_history_length": 10,
        "late_payments": 0, "bankruptcies": 0, "loan_amount": 50000,
        "loan_purpose": "home", "loan_tenure": 60
    }, cookies=cookies)
    test("What-if analysis", r)
    if r.status_code == 200:
        try:
            data = r.json()
            print(f"         Score: {data.get('credit_score')}, Risk: {data.get('risk_category')}, Default Prob: {data.get('probability_of_default', 0):.1%}")
        except Exception as e:
            print(f"         JSON parse error: {e}")

    # ============ ADMIN ============
    print("\n=== ADMIN PANEL ===")
    r = client.post("/api/auth/login", data={
        "email": "admin@creditrisk.ai",
        "password": "admin123"
    }, follow_redirects=False)
    test("Admin login", r, 303)
    admin_cookies = dict(r.cookies)

    test("Admin dashboard page", client.get("/admin", cookies=admin_cookies))

    r = client.get("/api/admin/stats", cookies=admin_cookies)
    test("Admin stats API", r)
    if r.status_code == 200:
        try:
            stats = r.json()
            print(f"         Users: {stats.get('total_users')}, Apps: {stats.get('total_applications')}, Avg Score: {stats.get('avg_credit_score')}")
        except Exception:
            pass

# ============ SUMMARY ============
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"\n{'='*50}")
print(f"RESULTS: {passed}/{total} tests passed")
if passed == total:
    print("ALL TESTS PASSED!")
else:
    print("Some tests failed:")
    for name, ok, code in results:
        if not ok:
            print(f"   FAILED: {name} (got {code})")
