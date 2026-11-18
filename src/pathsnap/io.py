"""Raw-file readers for the PathSnap dataset.

Each trip folder <id>/ contains:
  <id>.track  lon lat t          (GPS observations, ~1 Hz)
  <id>.nodes  lon lat            (row index = node id)
  <id>.arcs   from_node to_node  (row index = arc id, directed)
  <id>.route  arc_id             (chronological reference route; server-side only)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray          # (T,3) lon, lat, t
    nodes: np.ndarray          # (N,2) lon, lat
    arcs: np.ndarray           # (A,2) from, to
    route: np.ndarray | None   # (R,) arc ids, None if not loaded


def _load(path, dtype=float, cols=None):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.zeros((0, cols or 1), dtype=dtype)
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    return a


def list_trips(split_dir: str) -> list[str]:
    return sorted(d for d in os.listdir(split_dir)
                  if os.path.isdir(os.path.join(split_dir, d)))


def load_trip(split_dir: str, trip_id: str, with_route: bool = True) -> Trip:
    base = os.path.join(split_dir, trip_id, trip_id)
    track = _load(base + ".track", float, 3)
    nodes = _load(base + ".nodes", float, 2)
    arcs = _load(base + ".arcs", int, 2)
    route = None
    if with_route and os.path.exists(base + ".route"):
        route = _load(base + ".route", int, 1).ravel()
    return Trip(trip_id, track, nodes, arcs, route)


def load_route_only(split_dir: str, trip_id: str) -> np.ndarray:
    """Ground truth loader kept separate so the prediction path never touches it."""
    base = os.path.join(split_dir, trip_id, trip_id)
    return _load(base + ".route", int, 1).ravel()
