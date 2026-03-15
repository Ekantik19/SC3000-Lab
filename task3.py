# import json
# import heapq
# import math
# import os

# # ─────────────────────────────────────────────
# # Load data files  (adjust paths if needed)
# # ─────────────────────────────────────────────
# #reading json files

# _dir = os.path.dirname(os.path.abspath(__file__))

# with open(os.path.join(_dir, 'Coord.json')) as a, open(os.path.join(_dir, 'Cost.json')) as b, open(os.path.join(_dir, 'Dist.json')) as c, open(os.path.join(_dir, 'G.json')) as d:
#     Coord = json.load(a)
#     Cost = json.load(b)
#     Dist = json.load(c)
#     G = json.load(d)


# SOURCE  = '1'
# TARGET  = '50'
# BUDGET  = 287932

# # ─────────────────────────────────────────────
# # Heuristic  –  straight-line (Euclidean) distance
# # to the goal, scaled to match the Dist units.
# #
# # Because Euclidean distance is always ≤ true
# # shortest path distance, the heuristic is
# # admissible and A* is guaranteed optimal.
# # ─────────────────────────────────────────────
# tx, ty = Coord[TARGET]

# def heuristic(node: str) -> float:
#     x, y = Coord[node]
#     return math.sqrt((x - tx) ** 2 + (y - ty) ** 2)


# # ─────────────────────────────────────────────
# # A* with energy-budget pruning
# #
# # State  : node
# # f(n)   : g(n)  +  h(n)
# # g(n)   : best known distance from SOURCE to n
# # h(n)   : Euclidean distance from n to TARGET
# #
# # Energy constraint is handled by tracking the
# # energy along each path and pruning any state
# # whose accumulated energy exceeds BUDGET.
# #
# # To correctly handle the constraint we store
# # the *best (dist, energy) pair* seen for every
# # node.  A new path to a node is useful only if
# # it improves distance OR lowers energy cost
# # (could enable future expansions that the
# # high-energy version would prune).
# # ─────────────────────────────────────────────
# def astar():
#     # best[(node)] = (best_dist, best_energy) seen so far
#     best = {}

#     # heap: (f, g, energy, node, path)
#     start_h = heuristic(SOURCE)
#     heap = [(start_h, 0, 0, SOURCE, [SOURCE])]

#     while heap:
#         f, g, energy, node, path = heapq.heappop(heap)

#         # ── Goal check ──────────────────────────────────
#         if node == TARGET:
#             return path, g, energy

#         # ── Dominance pruning ───────────────────────────
#         # Skip if we've already found a path to this node
#         # that is both shorter AND cheaper in energy.
#         prev = best.get(node)
#         if prev is not None:
#             prev_dist, prev_energy = prev
#             if prev_dist <= g and prev_energy <= energy:
#                 continue          # strictly dominated → skip

#         best[node] = (g, energy)

#         # ── Expand neighbours ───────────────────────────
#         for nb in G.get(node, []):
#             edge      = f"{node},{nb}"
#             new_dist  = g      + Dist[edge]
#             new_energy= energy + Cost[edge]

#             if new_energy > BUDGET:
#                 continue          # prune: over budget

#             nb_prev = best.get(nb)
#             if nb_prev is not None:
#                 pd, pe = nb_prev
#                 if pd <= new_dist and pe <= new_energy:
#                     continue      # dominated → skip

#             new_f = new_dist + heuristic(nb)
#             heapq.heappush(heap, (new_f, new_dist, new_energy, nb, path + [nb]))

#     return None, None, None       # no feasible path found


# # ─────────────────────────────────────────────
# # Run & print result
# # ─────────────────────────────────────────────
# path, dist, energy = astar()

# if path is None:
#     print("No feasible path found within the energy budget.")
# else:
#     print(f"Shortest path: {'->'.join(path)}.")
#     print(f"Shortest distance: {dist}.")
#     print(f"Total energy cost: {energy}.")
import json
import heapq
import math
import os

# ─────────────────────────────────────────────
# Load data files
# ─────────────────────────────────────────────
_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_dir, 'Coord.json')) as a, open(os.path.join(_dir, 'Cost.json')) as b, open(os.path.join(_dir, 'Dist.json')) as c, open(os.path.join(_dir, 'G.json')) as d:
    Coord = json.load(a)
    Cost = json.load(b)
    Dist = json.load(c)
    G = json.load(d)

SOURCE = '1'
TARGET = '50'
BUDGET = 287932

tx, ty = Coord[TARGET]

def heuristic(node: str) -> float:
    x, y = Coord[node]
    return math.sqrt((x - tx) ** 2 + (y - ty) ** 2)


def verify_path(path: list) -> dict:
    """
    Verifies a path by walking through it node by node and checking:
      1. Each consecutive pair (u, v) is a valid edge in the graph
      2. The edge exists in Dist and Cost dictionaries
      3. Accumulates true distance and energy cost at each step
      4. Checks if total energy is within budget

    Returns a dict with:
      - valid         : bool   — True if all edges exist and are reachable
      - total_dist    : float  — sum of edge distances along the path
      - total_energy  : float  — sum of edge energy costs along the path
      - within_budget : bool   — True if total_energy <= BUDGET
      - error         : str    — description of first error found (or None)
      - step_log      : list   — per-step breakdown [(u, v, dist, energy, cumul_dist, cumul_energy)]
    """
    if not path:
        return {"valid": False, "error": "Path is empty.", "total_dist": 0,
                "total_energy": 0, "within_budget": False, "step_log": []}

    total_dist   = 0.0
    total_energy = 0.0
    step_log     = []

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge = f"{u},{v}"

        # Check edge exists in graph adjacency list
        if v not in G.get(u, []):
            return {
                "valid"         : False,
                "error"         : f"Step {i+1}: edge {u}->{v} does not exist in graph G",
                "total_dist"    : total_dist,
                "total_energy"  : total_energy,
                "within_budget" : total_energy <= BUDGET,
                "step_log"      : step_log
            }

        # Check edge exists in Dist and Cost
        if edge not in Dist:
            return {
                "valid"         : False,
                "error"         : f"Step {i+1}: edge {edge} missing from Dist dictionary",
                "total_dist"    : total_dist,
                "total_energy"  : total_energy,
                "within_budget" : total_energy <= BUDGET,
                "step_log"      : step_log
            }

        if edge not in Cost:
            return {
                "valid"         : False,
                "error"         : f"Step {i+1}: edge {edge} missing from Cost dictionary",
                "total_dist"    : total_dist,
                "total_energy"  : total_energy,
                "within_budget" : total_energy <= BUDGET,
                "step_log"      : step_log
            }

        step_dist    = Dist[edge]
        step_energy  = Cost[edge]
        total_dist   += step_dist
        total_energy += step_energy

        step_log.append((u, v, step_dist, step_energy, total_dist, total_energy))

    return {
        "valid"         : True,
        "error"         : None,
        "total_dist"    : total_dist,
        "total_energy"  : total_energy,
        "within_budget" : total_energy <= BUDGET,
        "step_log"      : step_log
    }


def astar():
    best    = {}
    start_h = heuristic(SOURCE)
    heap    = [(start_h, 0, 0, SOURCE, [SOURCE])]

    while heap:
        f, g, energy, node, path = heapq.heappop(heap)

        if node == TARGET:
            return path, g, energy

        prev = best.get(node)
        if prev is not None:
            prev_dist, prev_energy = prev
            if prev_dist <= g and prev_energy <= energy:
                continue

        best[node] = (g, energy)

        for nb in G.get(node, []):
            edge       = f"{node},{nb}"
            new_dist   = g      + Dist[edge]
            new_energy = energy + Cost[edge]

            if new_energy > BUDGET:
                continue

            nb_prev = best.get(nb)
            if nb_prev is not None:
                pd, pe = nb_prev
                if pd <= new_dist and pe <= new_energy:
                    continue

            new_f = new_dist + heuristic(nb)
            heapq.heappush(heap, (new_f, new_dist, new_energy, nb, path + [nb]))

    return None, None, None


# ─────────────────────────────────────────────
# Run A*
# ─────────────────────────────────────────────
path, dist, energy = astar()

if path is None:
    print("No feasible path found within the energy budget.")
else:
    print(f"Shortest path: {'->'.join(path)}.")
    print(f"Shortest distance: {dist}.")
    print(f"Total energy cost: {energy}.")

    # ─────────────────────────────────────────
    # Verify the path independently
    # ─────────────────────────────────────────
    print("\n--- Path Verification ---")
    result = verify_path(path)

    if not result["valid"]:
        print(f"INVALID PATH: {result['error']}")
    else:
        print(f"Path is valid        : {result['valid']}")
        print(f"Verified distance    : {result['total_dist']}")
        print(f"Verified energy      : {result['total_energy']}")
        print(f"Within energy budget : {result['within_budget']} (budget={BUDGET})")

        # Sanity check — verified values should match A* reported values
        dist_match   = abs(result['total_dist']   - dist)   < 1e-6
        energy_match = abs(result['total_energy'] - energy) < 1e-6
        print(f"Distance matches A*  : {dist_match}")
        print(f"Energy matches A*    : {energy_match}")

       