"""Produces the screenshots / captured outputs used in the report:
  img/map_eda.png, img/map_pred_a.png, img/map_pred_b.png, img/ui.png,
  img/api_response.txt, img/api_log.txt
It starts the real API and UI locally (student_test as the evaluation folder)."""
import os, sys, json, time, subprocess, signal, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/report")
from screenshot import shot
from pathsnap.pipeline import PathSnapPipeline

IMG = f"{ROOT}/report/img"; os.makedirs(IMG, exist_ok=True)
EVAL = f"{ROOT}/data/student_test"
hpt = pd.read_csv(f"{ROOT}/results/hmm/student_test_per_trip_tuned.csv", dtype={"trip": str})
hpt["gain"] = hpt.hmm_top1 - hpt.clf_top1
A = hpt.sort_values("gain", ascending=False).trip.iloc[0]
B = hpt.sort_values("gain", ascending=False).trip.iloc[1]

shot(f"{ROOT}/results/maps/eda_student_test_00000052.html", f"{IMG}/map_eda.png", 1200, 800)
shot(f"{ROOT}/results/maps/student_test_{A}.html", f"{IMG}/map_pred_a.png", 1200, 800)

# zoomed view of B where classifier-only is wrong but the HMM is right
pipe = PathSnapPipeline(f"{ROOT}/artifacts")
out = pipe.predict(EVAL, B)
from pathsnap.labels import align_route
from pathsnap.io import load_route_only, load_trip
P = out["P"]; tr = load_trip(EVAL, B)
_, lab, _ = align_route(P.pxy, P.net.nodes_xy, tr.arcs, load_route_only(EVAL, B))
bad = np.where((out["clf_arcs"] != lab) & (out["hmm_arcs"] == lab))[0]
# densest cluster of fixes the HMM corrected
k = bad[np.argmax([np.sum(np.abs(bad - b) < 30) for b in bad])] if len(bad) else 0
lat, lon = P.track[k, 1], P.track[k, 0]


def zoom(pg):
    pg.wait_for_timeout(1500)
    pg.evaluate(f"""() => {{ for (const k in window) if (k.startsWith('map_') && window[k] && window[k].setView) {{
        window[k].setView([{lat}, {lon}], 17);
        document.querySelectorAll('.leaflet-control-layers-selector').forEach(cb => {{
            if (cb.nextSibling && cb.nextSibling.textContent.includes('Classifier') && !cb.checked) cb.click(); }});
    }} }}""")
shot(f"{ROOT}/results/maps/student_test_{B}.html", f"{IMG}/map_pred_b.png", 1200, 800, 2500, zoom)

# --- live API + UI -------------------------------------------------------------
env = {**os.environ, "PATHSNAP_EVAL_DIR": EVAL}
logf = open(f"{IMG}/_api_raw.log", "w")
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
                       cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
ui = subprocess.Popen([sys.executable, "-m", "http.server", "5500"], cwd=f"{ROOT}/ui",
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(6)
try:
    req = urllib.request.Request("http://127.0.0.1:8000/predict", data=json.dumps({"trip_id": "00000052"}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req))
    d["gps"] = d["gps"][:2] + ["... %d more points" % (len(d["gps"]) - 2)]
    d["predicted_route"] = d["predicted_route"][:1] + ["... %d more arcs" % (len(d["predicted_route"]) - 1)]
    open(f"{IMG}/api_response.txt", "w").write("$ curl -X POST http://<laptop1>:8000/predict -d '{\"trip_id\":\"00000052\"}'\n"
                                               + json.dumps(d, indent=1))

    def ui_actions(pg):
        pg.fill("#api", "http://127.0.0.1:8000")
        pg.click("#connect"); pg.wait_for_timeout(1500)
        pg.click(f"#trips button:has-text('{A}')")
        pg.wait_for_selector("#results", state="visible", timeout=60000)
    shot("http://127.0.0.1:5500/index.html", f"{IMG}/ui.png", 1400, 860, 2000, ui_actions)
finally:
    time.sleep(1)
    api.send_signal(signal.SIGINT); ui.terminate(); api.wait(10)
logf.close()
lines = [l for l in open(f"{IMG}/_api_raw.log") if "pathsnap" not in l.lower() or "|" in l]
open(f"{IMG}/api_log.txt", "w").write("$ uvicorn api.main:app --host 0.0.0.0 --port 8000\n" + "".join(lines[-14:]))
json.dump({"MAPA": A, "MAPB": B}, open(f"{IMG}/map_trips.json", "w"))
print("screenshots done", A, B)
