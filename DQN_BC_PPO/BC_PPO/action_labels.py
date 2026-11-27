"""Executed lateral actions with lane-number changes across junctions removed."""
import numpy as np
import traci


def connected_entry_lanes(links, lane_api=None):
    """Physical normal-lane continuations through internal junction lanes."""
    lane_api = traci.lane if lane_api is None else lane_api
    entries, seen = set(), set()
    pending = [link[0] for link in links]
    while pending and len(seen) < 32:
        lane = pending.pop()
        if lane in seen:
            continue
        seen.add(lane)
        if not lane.startswith(":"):
            entries.add(lane)
        else:
            pending.extend(link[0] for link in lane_api.getLinks(lane))
    return entries


def lateral_label(before_lane, before_index, before_links, after_lane, after_index):
    before_edge = before_lane.rsplit("_", 1)[0]
    after_edge = after_lane.rsplit("_", 1)[0]
    if before_edge == after_edge:
        delta = after_index - before_index
        return float(np.sign(delta)), abs(delta) <= 1
    # A real lateral move can occur while ENTERING an internal edge too. Follow
    # the physical via lane before its normal destination, rather than dropping
    # every transition into a junction as an unlabelled lane-renumbering event.
    def next_lanes(links):
        return [link[4] if len(link) > 4 and link[4] else link[0] for link in links]
    entries, seen = set(), set()
    pending = next_lanes(before_links)
    while pending and len(seen) < 32:
        lane = pending.pop()
        if lane in seen:
            continue
        seen.add(lane)
        if lane.rsplit("_", 1)[0] == after_edge:
            entries.add(int(lane.rsplit("_", 1)[1]))
        elif lane.startswith(":"):
            pending.extend(next_lanes(traci.lane.getLinks(lane)))
    if len(entries) != 1:
        return 0.0, False
    delta = after_index - entries.pop()
    representable = not before_edge.startswith(":") or before_index + delta >= 0
    return float(np.sign(delta)), abs(delta) <= 1 and representable
