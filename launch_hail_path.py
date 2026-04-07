import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_FILE = os.path.join(BASE_DIR, "hail_path_streamlit_app.py")

cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    APP_FILE,
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
]

subprocess.Popen(cmd, cwd=BASE_DIR)