import urllib.request, urllib.error

BASE = "http://127.0.0.1:8000"
pages = ["/", "/login", "/register"]
for path in pages:
    try:
        r = urllib.request.urlopen(BASE + path, timeout=5)
        size = len(r.read())
        print(f"GET {path} -> {r.status} ({size} bytes)")
    except urllib.error.HTTPError as e:
        print(f"GET {path} -> {e.code}")
    except Exception as e:
        print(f"GET {path} -> ERROR: {e}")
