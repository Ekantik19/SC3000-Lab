import json
import heapq
import math
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────
#  Load Data Files
# ─────────────────────────────────────────────────────────

_dir = Path(__file__).resolve().parent

with open(_dir / 'Coord.json') as a, \
     open(_dir / 'Cost.json')  as b, \
     open(_dir / 'Dist.json')  as c, \
     open(_dir / 'G.json')     as d:
    Coord = json.load(a)
    Cost  = json.load(b)
    Dist  = json.load(c)
    G     = json.load(d)

# ─────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────

SOURCE = '1'
TARGET = '50'
BUDGET = 287932

# Precompute target coordinates for heuristic
tx, ty = Coord[TARGET]

# ─────────────────────────────────────────────────────────
#  Shared Heuristic
#  Euclidean distance to target — admissible (never overestimates)
# ─────────────────────────────────────────────────────────

def heuristic(node: str) -> float:
    x, y = Coord[node]
    return math.sqrt((x - tx) ** 2 + (y - ty) ** 2)


# ─────────────────────────────────────────────────────────
#  Path Verification Helper
#  Independently verifies a path's distance and energy cost
# ─────────────────────────────────────────────────────────

def verify_path(path: list) -> dict:
    if not path:
        return {"valid": False, "error": "Path is empty.",
                "total_dist": 0, "total_energy": 0,
                "within_budget": False}

    total_dist   = 0.0
    total_energy = 0.0

    for i in range(len(path) - 1):
        u    = path[i]
        v    = path[i + 1]
        edge = f"{u},{v}"

        if v not in G.get(u, []):
            return {"valid": False,
                    "error": f"Step {i+1}: edge {u}->{v} not in graph G",
                    "total_dist": total_dist, "total_energy": total_energy,
                    "within_budget": total_energy <= BUDGET}

        if edge not in Dist:
            return {"valid": False,
                    "error": f"Step {i+1}: edge {edge} missing from Dist",
                    "total_dist": total_dist, "total_energy": total_energy,
                    "within_budget": total_energy <= BUDGET}

        if edge not in Cost:
            return {"valid": False,
                    "error": f"Step {i+1}: edge {edge} missing from Cost",
                    "total_dist": total_dist, "total_energy": total_energy,
                    "within_budget": total_energy <= BUDGET}

        total_dist   += Dist[edge]
        total_energy += Cost[edge]

    return {"valid": True, "error": None,
            "total_dist": total_dist, "total_energy": total_energy,
            "within_budget": total_energy <= BUDGET}


# ─────────────────────────────────────────────────────────
#  TASK 1: A* Search — Relaxed (no energy constraint)
#
#  Since the energy constraint is removed, this is a standard
#  shortest path problem. A* uses Euclidean distance as the
#  heuristic to guide the search toward the target efficiently.
# ─────────────────────────────────────────────────────────

def task1_astar():
    counter = 0
    heap    = [(heuristic(SOURCE), 0, counter, SOURCE)]
    visited = set()
    prev    = {SOURCE: None}
    g_score = {SOURCE: 0}

    while heap:
        f, g, _, node = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)

        if node == TARGET:
            break

        for nb in G.get(node, []):
            if nb in visited:
                continue
            edge     = f"{node},{nb}"
            edge_dist = Dist.get(edge)
            if edge_dist is None:
                continue
            new_g = g + edge_dist
            if new_g < g_score.get(nb, float('inf')):
                g_score[nb] = new_g
                prev[nb]    = node
                counter    += 1
                heapq.heappush(heap, (new_g + heuristic(nb), new_g, counter, nb))

    # Reconstruct path
    path, node = [], TARGET
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()

    return path, g_score.get(TARGET, float('inf')), len(visited)


# ─────────────────────────────────────────────────────────
#  TASK 2: UCS — Uniform Cost Search (with energy budget)
# ─────────────────────────────────────────────────────────

def task2_ucs():
    heap    = [(0, SOURCE, 0, [SOURCE])]   # (dist, node, energy, path)
    visited = {}                           # node -> best energy seen

    while heap:
        dist, node, energy, path = heapq.heappop(heap)

        if node == TARGET:
            return path, dist, energy

        # Prune if we've visited this node with equal or lower energy
        if node in visited and visited[node] <= energy:
            continue
        visited[node] = energy

        for nb in G.get(node, []):
            edge       = f"{node},{nb}"
            new_dist   = dist   + Dist[edge]
            new_energy = energy + Cost[edge]

            if new_energy <= BUDGET:
                heapq.heappush(heap, (new_dist, nb, new_energy, path + [nb]))

    return None, None, None


# ─────────────────────────────────────────────────────────
#  TASK 3: A* Search (with energy budget)
# ─────────────────────────────────────────────────────────

def task3_astar():
    best     = {}                                      
    start_h  = heuristic(SOURCE)
    heap     = [(start_h, 0, 0, SOURCE, [SOURCE])]    

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


# ─────────────────────────────────────────────────────────
#  Display Helper
# ─────────────────────────────────────────────────────────

def print_result(task_name, path, dist, energy):
    print(f"\n{'='*60}")
    print(f"  {task_name}")
    print(f"{'='*60}")
    if path is None:
        print("  No feasible path found within the energy budget.")
        return
    print(f"  Shortest path     : {'->'.join(path)}.")
    print(f"  Shortest distance : {dist}.")
    print(f"  Total energy cost : {energy}.")

    # Independent verification
    result = verify_path(path)
    if not result["valid"]:
        print(f"\n  WARNING — Verification failed: {result['error']}")
    else:
        print(f"\n  Verification:")
        print(f"    Path valid       : {result['valid']}")
        print(f"    Distance matches : {abs(result['total_dist']   - dist)   < 1e-6}")
        print(f"    Energy matches   : {abs(result['total_energy'] - energy) < 1e-6}")
        print(f"    Within budget    : {result['within_budget']} (budget={BUDGET})")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SC3000 Lab Assignment 1 — Part 1")
    print("  Shortest Path with Energy Budget")
    print("=" * 60)
    print(f"  Source : {SOURCE}  |  Target : {TARGET}  |  Budget : {BUDGET}")

    # ── Task 1: A* without energy constraint ──────────────
    print("\n\nRunning Task 1: A* (no energy constraint)...")
    t1_path, t1_dist, t1_nodes = task1_astar()
    print(f"\n{'='*60}")
    print("  TASK 1 — A* Search (Relaxed, no energy constraint)")
    print(f"{'='*60}")
    if t1_path:
        print(f"  Shortest path     : {'->'.join(t1_path)}.")
        print(f"  Shortest distance : {t1_dist}.")
        print(f"  Nodes explored    : {t1_nodes}")
    else:
        print("  No path found.")

    # ── Task 2: UCS with energy constraint ────────────────
    print("\n\nRunning Task 2: UCS (with energy budget)...")
    t2_path, t2_dist, t2_energy = task2_ucs()
    print_result("TASK 2 — UCS (Uninformed Search, energy budget)",
                 t2_path, t2_dist, t2_energy)

    # ── Task 3: A* with energy constraint ─────────────────
    print("\n\nRunning Task 3: A* (with energy budget)...")
    t3_path, t3_dist, t3_energy = task3_astar()
    print_result("TASK 3 — A* Search (with energy budget)",
                 t3_path, t3_dist, t3_energy)

    # ── Cross-task comparison ─────────────────────────────
    print(f"\n{'='*60}")
    print("  Summary Comparison")
    print(f"{'='*60}")
    print(f"  {'Task':<35} {'Distance':>12}  {'Energy':>12}")
    print(f"  {'-'*60}")
    if t1_path:
        print(f"  {'Task 1 — A* (no energy constraint)':<35} {t1_dist:>12}  {'N/A':>12}")
    if t2_path:
        print(f"  {'Task 2 — UCS (energy budget)':<35} {t2_dist:>12}  {t2_energy:>12}")
    if t3_path:
        print(f"  {'Task 3 — A* (energy budget)':<35} {t3_dist:>12}  {t3_energy:>12}")

    if t2_path and t3_path:
        if abs(t2_dist - t3_dist) < 1e-6:
            print(f"\n  ✓ Task 2 (UCS) and Task 3 (A*) found the same optimal distance.")
        else:
            print(f"\n  ✗ Task 2 and Task 3 distances differ — check implementations.")