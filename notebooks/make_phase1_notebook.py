"""Builds and executes notebooks/phase1_preprocessing_features.ipynb."""
import os, nbformat as nbf
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# PathSnap: Phase 1, preprocessing and feature engineering
Each step runs on the raw data using the same library code as the full pipeline (`src/pathsnap/`).
**Partitions:** the live demos use TRAIN trips. The partition-level tables come from `scripts/02_eda.py` (TRAIN + STUDENT_TEST).
The evaluator-test data is never used.""")
code("""import os, sys, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('../src'))
from pathsnap.io import load_trip, list_trips
from pathsnap.geo import clean_track, LocalProjection, Network, heading_from_track
from pathsnap.labels import align_route
from pathsnap.features import prepare_trip, filter_candidates, FEATURES, FINAL_CFG
DATA = os.environ.get('PATHSNAP_DATA', '../data')
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
print('train trips:', len(list_trips(f'{DATA}/train')), '| student_test trips:', len(list_trips(f'{DATA}/student_test')))""")

md("## 1. Raw files of one trip\nEach file has no header. The row index is the id: node id for `.nodes`, arc id for `.arcs`.")
code("""TID = '00000011'   # a trip with a loop and twin arcs, which is where labelling is hard
trip = load_trip(f'{DATA}/train', TID)
for name, arr in [('track (lon, lat, t)', trip.track), ('nodes (lon, lat)', trip.nodes),
                  ('arcs (from_node, to_node)', trip.arcs), ('route (arc_id, in order)', trip.route)]:
    print(f'{name:28s} shape={arr.shape}  first rows: {np.asarray(arr)[:3].tolist()}')""")

md("## 2. Dataset inventory and data quality (TRAIN vs STUDENT_TEST)")
code("""inv = pd.read_csv('../results/eda/inventory.csv', dtype={'trip': str})
summ = inv.groupby('split').agg(trips=('trip','count'), gps_obs=('n_obs','sum'), obs_median=('n_obs','median'),
        missing_values=('n_nan','sum'), non_monotonic_t=('non_monotonic','sum'), gaps_over_5s=('n_gaps_gt5s','sum'),
        frac_1s_steps=('dt_1s_frac','mean'), median_arcs=('n_arcs','median'), isolated_nodes=('isolated_nodes','sum'),
        median_weak_components=('weak_components','median'), routes_contiguous=('route_contiguous','mean'),
        twin_arc_frac=('twin_arc_frac','mean'))
summ.T.round(3)""")
code("""fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for s, c in [('train', '#2b6cb0'), ('student_test', '#dd6b20')]:
    d = inv[inv.split == s]
    ax[0].hist(np.log10(d.n_obs), bins=20, alpha=.6, color=c, label=s)
    ax[1].hist(d.path_len_km, bins=20, alpha=.6, color=c, label=s)
    ax[2].hist(d.dt_max.clip(upper=600), bins=20, alpha=.6, color=c, label=s)
ax[0].set_xlabel('log10(# GPS fixes)'); ax[1].set_xlabel('path length (km)'); ax[2].set_xlabel('largest time gap (s, clipped at 600)')
for a in ax: a.legend()
plt.tight_layout()""")
md("""**What we see:** there are no missing values, and timestamps are always in order. The data is nominally 1 Hz, but there are 1,216 gaps over 5 s in TRAIN.
The networks are fragmented into many components, and 11 of the 90 reference routes are not contiguous. Trips are spread worldwide, so each trip needs its own projection.""")

md("## 3. Cleaning and projection\nWe drop invalid, out-of-order and duplicate-time fixes, then project to metres with a local equirectangular projection centred on the trip.")
code("""track, rep = clean_track(trip.track)
print('cleaning report:', rep)
proj = LocalProjection(np.median(track[:,0]), np.median(track[:,1]))
xy = proj.fwd(track[:,0], track[:,1])
deg = np.hypot(*np.diff(track[:,:2], axis=0).T)        # naive degree distance (wrong)
m = np.hypot(*np.diff(xy, axis=0).T)                    # metres
print(f'median step: {np.median(deg):.6f} degrees  vs  {np.median(m):.1f} m  (~{np.median(m)*3.6:.0f} km/h at 1 Hz)')
lat0 = np.radians(proj.lat0); print(f'1 deg lon = {proj.kx/1000:.1f} km but 1 deg lat = {proj.ky/1000:.1f} km here, which is why degrees cannot be used as distance')""")

md("## 4. Network cropping")
code("""net = Network(trip.nodes, trip.arcs, proj, (xy.min(0), xy.max(0)), buffer_m=800)
print(f'arcs in file: {len(trip.arcs):,} -> kept after crop (GPS bbox + 800 m): {len(net.arc_ids):,}')
print(f'route arcs inside crop: {np.isin(trip.route, net.arc_ids).mean():.0%}')""")

md("""## 5. Per-fix labels from the route (monotonic DP alignment)
`.route` gives the arcs in order but does not say which fix is on which arc. The DP minimises total fix-to-arc distance and never lets the route position go backwards in time.
Below, we compare it with naive "snap each fix to the nearest route arc".""")
code("""pos, lab, ldist = align_route(xy, net.nodes_xy, trip.arcs, trip.route)
# naive: nearest route arc, ignoring order
from pathsnap.labels import _route_dists
D = _route_dists(xy, net.nodes_xy, trip.arcs, trip.route)
naive = trip.route[D.argmin(1)]
print(f'fixes: {len(xy)} | DP label distance median {np.median(ldist):.1f} m, p95 {np.percentile(ldist,95):.1f} m')
print(f'naive nearest-arc labels disagree with DP on {np.mean(naive != lab):.1%} of fixes')
twin = [(trip.arcs[a,0]==trip.arcs[b,1]) and (trip.arcs[a,1]==trip.arcs[b,0]) for a,b in zip(naive, lab)]
if (naive != lab).any(): print(f'  of those disagreements, {np.mean(np.array(twin)[naive != lab]):.0%} are the reverse-direction twin arc')
print('route position never decreases:', bool(np.all(np.diff(pos) >= 0)))""")
code("""rep = pd.read_csv('../results/eda/preprocess_report.csv', dtype={'trip': str})
print(json.load(open('../results/eda/excluded_trips.json')))
rep[rep.split=='train'].sort_values('label_dist_median', ascending=False)[['trip','n_clean','label_dist_median','label_dist_p95']].head(5)""")
md("Trips 33 and 54 sit 120–140 m away from their own reference route, so their labels are unreliable. Both are **excluded from training**; 78 train trips remain.")

md("## 6. Candidate generation: choosing the radius and K on TRAIN")
code("""grid = pd.read_csv('../results/eda/candidate_radius_grid_train.csv')
g = grid.pivot(index='radius', columns='k', values='miss_rate').mul(100).round(2)
g.columns = [f'K={k}' for k in g.columns]; g.index.name = 'radius (m)'
print('candidate miss rate (%) on TRAIN:'); display(g)
r = grid.set_index(['radius','k']).rows
print(f"chosen 50 m / K=8: {r[(50,8)]:,} rows;  80 m / K=12 would need {r[(80,12)]:,} rows (+{r[(80,12)]/r[(50,8)]-1:.0%}) for <1 pt gain")""")

md("## 7. Feature engineering: 21 features per (fix, candidate arc)")
code("""P = prepare_trip(trip, {'radius': 80, 'k_max': 12})          # wide pass, same as the training pipeline
df = filter_candidates(P.df, FINAL_CFG['radius'], FINAL_CFG['k_max'])
print(f'{P.n_obs} fixes -> {len(df)} candidate rows ({len(df)/P.n_obs:.1f} per fix), positives {df.y.mean():.1%}')
print(f"fixes whose true arc is in the candidate set: {df[df.y==1].obs.nunique()/P.n_obs:.2%}")
df[df.obs == 200][['arc','y'] + FEATURES].round(2)""")
code("""groups = {
 'Geometry': ['dist','dist_rank','dist_minus_min','dist_softmax'],
 'Direction (separates reverse twins)': ['head_diff','head_diff_long','head_diff_undirected'],
 'Kinematics / gaps': ['speed','step_len','dt'],
 'Temporal context (t-3..t+3)': ['win_dist_mean','win_dist_max','dist_prev','dist_next'],
 'Arc & topology': ['arc_len','proj_frac','proj_overshoot','out_deg_to','in_deg_from','has_twin','n_cands']}
pd.DataFrame([(g, f) for g, fs in groups.items() for f in fs], columns=['group','feature']).groupby('group').feature.apply(', '.join).to_frame()""")
md("### Do the features separate correct from wrong candidates? (8 TRAIN trips, live)")
code("""tids = list_trips(f'{DATA}/train')[:10]
tids = [t for t in tids if t not in ('00000033','00000054')][:8]
F = pd.concat([filter_candidates(prepare_trip(load_trip(f'{DATA}/train', t), {'radius':80,'k_max':12}).df, 50, 8) for t in tids])
show = ['dist','dist_minus_min','head_diff','head_diff_long','win_dist_mean','proj_overshoot']
fig, axs = plt.subplots(1, len(show), figsize=(18, 3.4))
for a, f in zip(axs, show):
    v0, v1 = F.loc[F.y==0, f], F.loc[F.y==1, f]
    hi = np.percentile(F[f], 99)
    a.hist(v0.clip(upper=hi), bins=40, alpha=.55, density=True, label='wrong arc (y=0)', color='#a0aec0')
    a.hist(v1.clip(upper=hi), bins=40, alpha=.65, density=True, label='correct arc (y=1)', color='#2b6cb0')
    a.set_title(f, fontsize=10)
axs[0].legend(fontsize=8); plt.tight_layout()
print(f'{len(F):,} candidate rows from {len(tids)} train trips')
F.groupby('y')[show].median().round(2).rename(index={0:'median, wrong arc',1:'median, correct arc'})""")
md("""**Reading the plots:** correct arcs are closer (`dist`, `win_dist_mean`) and point the same way the car is moving (`head_diff` ≈ 0°, while the reverse twin sits ≈ 180°).
They also rarely project past the end of the arc (`proj_overshoot`). Heading is the only feature that can separate an arc from its reverse twin: 92 % of arcs have one, and the twin has identical geometry.""")

md("## 8. Folium map: raw GPS, road network and reference route (TRAIN trip)")
code("""from pathsnap.viz import trip_map
m = trip_map(P, reference_route=trip.route, title=f'train {TID}: raw GPS, network, reference route')
m""")
md("""## Summary of Phase 1
| step | decision | evidence |
|---|---|---|
| cleaning | drop invalid / out-of-order / duplicate-time fixes; keep gaps | 0 missing, 1,216 gaps > 5 s (TRAIN) |
| projection | per-trip local equirectangular, metres | trips span latitude -34° to 62° |
| cropping | GPS bbox + 800 m | 100 % of route arcs kept; speeds up search and Dijkstra |
| labels | monotonic DP alignment to `.route` | naive snapping disagrees on 43 % of trip 11's fixes |
| anomalies | exclude train trips 33, 54 | median offset 122 m / 140 m |
| candidates | radius 50 m, K = 8 | 1.85 % miss on TRAIN, 0.42 % on student-test |
| features | 21 (geometry, heading, kinematics, context, topology) | clear class separation in Section 7 |""")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
nbf.write(nb, os.path.join(HERE, "phase1_preprocessing_features.ipynb"))
print("notebook written")
