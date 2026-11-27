"""Training-only initial states sampled behind natural red-light queues.

Background vehicles and signals keep their normal SUMO controllers. This module
only chooses the ego's initial route, lane, position and speed; it never controls
an inserted ego. Failure to find a safe insertion falls back to normal training.
"""
import numpy as np


def add_queue_start(env, traci):
    route = list(traci.route.getEdges("r0"))
    candidates = []
    for vid in traci.vehicle.getIDList():
        edge = traci.vehicle.getRoadID(vid)
        if edge.startswith(":") or edge not in route[:-1]:
            continue
        if traci.vehicle.getSpeed(vid) >= 0.5:
            continue
        tls = traci.vehicle.getNextTLS(vid)
        if not tls or tls[0][3].lower() not in ("r", "u") or tls[0][2] > 150:
            continue
        lane = traci.vehicle.getLaneID(vid)
        k = route.index(edge)
        if not any(link[0].rsplit("_", 1)[0] == route[k + 1]
                   for link in traci.lane.getLinks(lane)):
            continue
        candidates.append((vid, edge, lane, k))
    np.random.shuffle(candidates)
    for vid, edge, lane, k in candidates:
        lead_rear = traci.vehicle.getLanePosition(vid) - traci.vehicle.getLength(vid)
        for _ in range(8):
            gap = float(np.random.uniform(15, 60))
            pos = lead_rear - gap
            if pos < 10:
                continue
            # No intervening vehicle; enough rear space for a slow insertion.
            clear = True
            for other in traci.lane.getLastStepVehicleIDs(lane):
                if other == vid:
                    continue
                op = traci.vehicle.getLanePosition(other)
                rear = op - traci.vehicle.getLength(other)
                if op >= pos - 5:
                    if rear < lead_rear:
                        clear = False
                        break
                elif pos - 5 - op < max(15, 2 * traci.vehicle.getSpeed(other)):
                    clear = False
                    break
            if not clear:
                continue
            rid = f"queue_from_{k}"
            if rid not in traci.route.getIDList():
                traci.route.add(rid, route[k:])
            stopped_prob = float(getattr(env, "queue_stopped_prob", 0.0))
            speed = (0.0 if stopped_prob > 0 and np.random.rand() < stopped_prob
                     else float(np.random.uniform(0, 3)))
            traci.vehicle.add(env.ego_id, rid, typeID="ego_type",
                              departLane=str(traci.vehicle.getLaneIndex(vid)),
                              departPos=str(pos), departSpeed=str(speed))
            env._queue_episode = True
            env._queue_start = dict(leader=vid, edge=edge, lane=lane,
                                    requested_gap=gap, requested_speed=speed)
            return edge
    return None
