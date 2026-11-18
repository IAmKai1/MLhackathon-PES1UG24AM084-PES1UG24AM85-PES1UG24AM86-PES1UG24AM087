"""Folium maps: raw GPS, nearby road network, reference route and predicted route."""
import folium
import numpy as np

COLORS = {"gps": "#2a78d6", "network": "#9a9994", "reference": "#1baf7a", "predicted": "#eb6834"}


def _arc_lines(trip, arc_ids):
    return [[(trip.nodes[u, 1], trip.nodes[u, 0]), (trip.nodes[v, 1], trip.nodes[v, 0])]
            for u, v in trip.arcs[arc_ids]]


def _route_polyline(trip, arc_ids):
    """Chain consecutive arcs into polylines, splitting where they do not connect."""
    lines, cur = [], []
    prev_head = None
    for a in arc_ids:
        u, v = trip.arcs[a]
        if prev_head is not None and u != prev_head and cur:
            lines.append(cur)
            cur = []
        if not cur:
            cur.append((trip.nodes[u, 1], trip.nodes[u, 0]))
        cur.append((trip.nodes[v, 1], trip.nodes[v, 0]))
        prev_head = v
    if cur:
        lines.append(cur)
    return lines


def dedupe_consecutive(seq):
    seq = np.asarray(seq)
    if len(seq) == 0:
        return seq
    keep = np.r_[True, seq[1:] != seq[:-1]]
    return seq[keep]


def trip_map(trip, net, out_path, predicted_arcs=None, title="", network_radius=150.0, max_gps=3000):
    center = [float(trip.track[:, 1].mean()), float(trip.track[:, 0].mean())]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron", control_scale=True)

    # Road network near the trajectory only (the full network has up to 300k arcs)
    near = set()
    for p in trip.xy[:: max(1, len(trip.xy) // 1500)]:
        near.update(net.arcs_near(p, network_radius).tolist())
    fg_net = folium.FeatureGroup(name="Road network (near trajectory)", show=True)
    for line in _arc_lines(trip, np.array(sorted(near), dtype=int)):
        folium.PolyLine(line, color=COLORS["network"], weight=1, opacity=0.5).add_to(fg_net)
    fg_net.add_to(m)

    if trip.route is not None:
        fg_ref = folium.FeatureGroup(name="Reference route", show=True)
        for line in _route_polyline(trip, trip.route):
            folium.PolyLine(line, color=COLORS["reference"], weight=6, opacity=0.7,
                            tooltip="Reference route").add_to(fg_ref)
        fg_ref.add_to(m)

    if predicted_arcs is not None:
        fg_pred = folium.FeatureGroup(name="Predicted route", show=True)
        for line in _route_polyline(trip, dedupe_consecutive(predicted_arcs)):
            folium.PolyLine(line, color=COLORS["predicted"], weight=3, opacity=0.9,
                            tooltip="Predicted route").add_to(fg_pred)
        fg_pred.add_to(m)

    fg_gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    stride = max(1, len(trip.track) // max_gps)
    for lon, lat, t in trip.track[::stride]:
        folium.CircleMarker((lat, lon), radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(fg_gps)
    fg_gps.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    legend = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;background:{c};'
        f'margin-right:6px;vertical-align:middle"></span>{name}</div>'
        for name, c in [("Raw GPS", COLORS["gps"]), ("Road network", COLORS["network"]),
                        ("Reference route", COLORS["reference"]), ("Predicted route", COLORS["predicted"])]
        if not (name == "Predicted route" and predicted_arcs is None)
        and not (name == "Reference route" and trip.route is None))
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fcfcfb;'
        f'padding:8px 12px;border:1px solid #e4e3df;border-radius:6px;font:12px sans-serif;color:#0b0b0b">'
        f'<b>{title}</b>{legend}</div>'))
    lats, lons = trip.track[:, 1], trip.track[:, 0]
    m.fit_bounds([[lats.min(), lons.min()], [lats.max(), lons.max()]])
    m.save(str(out_path))
    return m
