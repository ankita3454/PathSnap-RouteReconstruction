"""Folium map generation."""
from __future__ import annotations

import folium
import numpy as np

LEGEND = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
 padding: 10px 12px; border: 1px solid #999; border-radius: 6px; font: 13px sans-serif;">
<b>{title}</b><br>
<span style="color:#a0aec0">&#9644;</span> Road network (near track)<br>
<span style="color:#2b6cb0">&#9679;</span> Raw GPS observations<br>{extra}
</div>"""


def _arc_lines(P, arc_ids):
    """Return list of [[lat,lon],[lat,lon]] for original arc ids."""
    out = []
    net = P.net
    for a in arc_ids:
        li = net.local.get(int(a))
        if li is None:
            continue
        u, v = net.u[li], net.v[li]
        out.append([[net.nodes_ll[u, 1], net.nodes_ll[u, 0]], [net.nodes_ll[v, 1], net.nodes_ll[v, 0]]])
    return out


def trip_map(P, reference_route=None, predicted_arcs=None, clf_arcs=None, title="", near_m=150.0,
             max_points=4000):
    tr = P.track
    m = folium.Map(location=[float(np.mean(tr[:, 1])), float(np.mean(tr[:, 0]))], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    # network near track
    net = P.net
    mid = (net.p0 + net.p1) / 2
    from scipy.spatial import cKDTree
    tree = cKDTree(P.pxy)
    d, _ = tree.query(mid, k=1)
    near = np.where(d < near_m + net.length / 2)[0][:15000]
    fg = folium.FeatureGroup(name="Road network", show=True)
    lines = [[[net.nodes_ll[net.u[i], 1], net.nodes_ll[net.u[i], 0]],
              [net.nodes_ll[net.v[i], 1], net.nodes_ll[net.v[i], 0]]] for i in near]
    if lines:
        folium.PolyLine(lines, color="#a0aec0", weight=2, opacity=.7).add_to(fg)
    fg.add_to(m)
    if reference_route is not None:
        fg = folium.FeatureGroup(name="Reference route", show=True)
        ls = _arc_lines(P, reference_route)
        if ls:
            folium.PolyLine(ls, color="#38a169", weight=7, opacity=.6).add_to(fg)
        fg.add_to(m)
    if clf_arcs is not None:
        fg = folium.FeatureGroup(name="Classifier-only argmax", show=False)
        ls = _arc_lines(P, clf_arcs)
        if ls:
            folium.PolyLine(ls, color="#d69e2e", weight=4, opacity=.8).add_to(fg)
        fg.add_to(m)
    if predicted_arcs is not None:
        fg = folium.FeatureGroup(name="Predicted route (HMM)", show=True)
        ls = _arc_lines(P, predicted_arcs)
        if ls:
            folium.PolyLine(ls, color="#e53e3e", weight=3.5, opacity=.9).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Raw GPS", show=True)
    step = max(1, len(tr) // max_points)
    for i in range(0, len(tr), step):
        folium.CircleMarker([tr[i, 1], tr[i, 0]], radius=2, color="#2b6cb0", fill=True, fill_opacity=.8,
                            weight=0, tooltip=f"t={tr[i,2]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    extra = ""
    if reference_route is not None:
        extra += '<span style="color:#38a169">&#9644;</span> Reference route<br>'
    if predicted_arcs is not None:
        extra += '<span style="color:#e53e3e">&#9644;</span> Predicted route (HMM)<br>'
    if clf_arcs is not None:
        extra += '<span style="color:#d69e2e">&#9644;</span> Classifier-only argmax arcs (toggle)<br>'
    m.get_root().html.add_child(folium.Element(LEGEND.format(title=title, extra=extra)))
    return m
