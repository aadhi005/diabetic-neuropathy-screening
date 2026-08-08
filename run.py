"""
Convenience launcher: (re)build whatever is missing, then serve the dashboard.

    python run.py             # serve http://localhost:8777, building if needed
    python run.py --train     # force-regenerate the synthetic model first
    python run.py --all       # rebuild synthetic + real model, then serve
    python run.py --no-serve  # build only, don't start the server

The dashboard reads two exports: web/model.js (synthetic pathways) and
web/realmodel.js (real patients). The second only exists once a real model has
been trained, so the launcher builds it when the data file is present rather
than leaving the "Real patients" tab silently missing.
"""
import http.server
import os
import socketserver
import sys
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
PORT = 8777

REAL_DATA = os.path.join(ROOT, "data", "physionet_diabetes",
                         "GE-71_Data_Summary_Table.csv")
REAL_ARTIFACT = os.path.join(ROOT, "models", "realmodel.joblib")


def build_synthetic(force=False):
    if force or not os.path.exists(os.path.join(WEB, "model.js")):
        print("[synthetic] generating cohort, extracting features, training...")
        from neuroscreen.train import main as train_main
        train_main()
    else:
        print("[synthetic] web/model.js present - skipping (use --train to rebuild)")


def build_real(force=False):
    """Train + export the real-patient model when its dataset is available."""
    if not os.path.exists(REAL_DATA):
        print("[real] dataset not found - skipping. Fetch it with:")
        print("       python -m neuroscreen.fetch_data")
        return
    if force or not os.path.exists(REAL_ARTIFACT):
        print("[real] training on the PhysioNet cohort...")
        import subprocess
        subprocess.run([sys.executable, "-m", "neuroscreen.realtrain",
                        "--dataset", "vasoreg", "--data", REAL_DATA],
                       cwd=ROOT, check=True)
    if force or not os.path.exists(os.path.join(WEB, "realmodel.js")):
        print("[real] exporting to the dashboard...")
        from neuroscreen.export_real import main as export_main
        export_main()
    else:
        print("[real] web/realmodel.js present - skipping (use --all to rebuild)")


def main():
    force_all = "--all" in sys.argv
    build_synthetic(force="--train" in sys.argv or force_all)
    build_real(force=force_all)

    if "--no-serve" in sys.argv:
        print("\nBuild complete (--no-serve given).")
        return

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
