"""Offline navigation-plan variants; never select or override driving actions."""
from pathlib import Path
import xml.etree.ElementTree as ET

from env.sumo_env import SumoHighwayEnv  # Configure SUMO tools.
import sumolib


def straight_detour(base):
    """Replace the first left turn by a legal straight path back to that turn.

    Preserve the entire remaining original itinerary, including repeated edges.
    This is a preassigned demonstration mission, not runtime route recovery.
    """
    base = Path(base)
    root = ET.parse(base / "sumo/highway.rou.xml").getroot()
    original = root.find("route[@id='r0']").get("edges").split()
    net = sumolib.net.readNet(str(base / "sumo/highway.net.xml"))
    for k, (source, desired) in enumerate(zip(original, original[1:])):
        outgoing = net.getEdge(source).getOutgoing()
        if not any(c.getDirection() in ("l", "L") for target, cs in outgoing.items()
                   if target.getID() == desired for c in cs):
            continue
        candidates = []
        for target, connections in outgoing.items():
            if target.getID() == desired or not any(c.getDirection() == "s" for c in connections):
                continue
            path, cost = net.getShortestPath(target, net.getEdge(desired), vClass="passenger")
            if path:
                candidates.append((cost, [edge.getID() for edge in path]))
        if not candidates:
            raise ValueError("First left turn has no legal straight-and-rejoin path")
        _, path = min(candidates)
        variant = original[:k + 1] + path + original[k + 2:]
        assert path[-1] == desired
        return dict(original=original, route=variant, decision_edge=source,
                    original_next_edge=desired, detour_edges=path,
                    extra_distance_m=sum(net.getEdge(e).getLength() for e in path[:-1]),
                    original_distance_m=sum(net.getEdge(e).getLength() for e in original),
                    variant_distance_m=sum(net.getEdge(e).getLength() for e in variant))
    raise ValueError("No left turn found in the original itinerary")
