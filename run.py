"""
Convenience launcher: (re)train if needed, then serve the dashboard.

    python run.py            # serve the web dashboard at http://localhost:8777
    python run.py --train    # regenerate data + model first, then serve
"""
import http.server
import os
import socketserver
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
PORT = 8777


def main():
    if "--train" in sys.argv or not os.path.exists(os.path.join(WEB, "model.js")):
        print("Training model (generating cohort + features)...")
        from neuroscreen.train import main as train_main
        train_main()

    os.chdir(WEB)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        url = f"http://localhost:{PORT}/"
        print(f"\nNeuroScreen dashboard running at  {url}")
        print("Press Ctrl+C to stop.")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        httpd.serve_forever()


if __name__ == "__main__":
    main()
