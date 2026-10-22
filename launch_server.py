"""Startup script — redirects all output to a log file so the process stays alive."""
import sys, os, subprocess, time

project_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(project_dir, "server.log")

with open(log_path, "w") as log:
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import uvicorn; uvicorn.run('app.main:app', host='127.0.0.1', port=8000, log_level='info')"],
        cwd=project_dir,
        stdout=log,
        stderr=log,
    )
    # Wait up to 12 seconds for server to come up
    import urllib.request, urllib.error
    for i in range(12):
        time.sleep(1)
        try:
            r = urllib.request.urlopen("http://127.0.0.1:8000/login", timeout=2)
            print(f"Server is UP! (PID {proc.pid}) — http://127.0.0.1:8000")
            print(f"Logs at: {log_path}")
            break
        except Exception:
            print(f"  waiting... ({i+1}s)")
    else:
        print("Server did not respond in time. Check server.log")
        proc_poll = proc.poll()
        if proc_poll is not None:
            print(f"Process exited with code {proc_poll}")
            with open(log_path) as f:
                print(f.read()[-2000:])
