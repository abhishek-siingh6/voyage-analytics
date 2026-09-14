"""
Voyage Analytics Unified Launcher
Starts Flask REST API backend and opens the Streamlit Dashboard in the browser.
"""

import os
import sys
import time
import subprocess
import urllib.request
from pathlib import Path

# Setup paths and environment
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(PROJECT_ROOT)


def wait_for_api(host="127.0.0.1", port=5000, timeout=15):
    """Wait until Flask API health endpoint returns 200 OK."""
    health_url = f"http://{host}:{port}/api/health"
    start_time = time.time()
    print(f"Waiting for Flask API backend at {health_url}...")
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.urlopen(health_url, timeout=1)
            if req.getcode() == 200:
                print("✓ Flask API backend is ready!")
                return True
        except Exception:
            time.sleep(0.5)
    print("Warning: Flask API health check timed out. Proceeding to launch dashboard...")
    return False


def main():
    print("==================================================")
    print(" Starting Voyage Analytics Project")
    print("==================================================")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # 1. Launch Flask REST API subprocess
    api_script = PROJECT_ROOT / "scripts" / "run_local.py"
    print(f"Starting Flask API server via {api_script}...")
    api_process = subprocess.Popen(
        [sys.executable, str(api_script), "--create-dummy"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    # 2. Wait for API to respond
    wait_for_api()

    # 3. Launch Streamlit UI Dashboard
    dashboard_script = PROJECT_ROOT / "dashboard" / "app.py"
    print(f"\nLaunching Streamlit Dashboard in browser from: {dashboard_script}")
    print("App web URL: http://localhost:8501\n")

    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_script),
        "--server.headless=false",
    ]

    try:
        subprocess.run(streamlit_cmd, cwd=str(PROJECT_ROOT), env=env)
    except KeyboardInterrupt:
        print("\nShutting down Voyage Analytics...")
    finally:
        if api_process.poll() is None:
            api_process.terminate()


if __name__ == "__main__":
    main()
