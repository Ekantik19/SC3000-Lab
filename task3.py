import json
import heapq
import math


# ----------------------------
# Load graph data
# ----------------------------
with open('Coord.json') as a, open('Cost.json') as b, open('Dist.json') as c, open('G.json') as d:
    Coord = json.load(a)
    Cost = json.load(b)
    Dist = json.load(c)
    G = json.load(d)


# ----------------------------
# Heuristic function
# Straight-line (Euclidean) distance
# ----------------------------
def heuristic(n, goal, Coord):
    x1, y1 = Coord[n]
    x2, y2 = Coord[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ----------------------------
# A* Search
# ----------------------------
def astar(G, Dist, Cost, Coord, start, goal, budget):

    pq = []
    heapq.heappush(pq, (heuristic(start, goal, Coord), start, 0, 0))
    # (f, node, distance_so_far, energy_so_far)

    parent = {}
    best_dist = {start: 0}

    while pq:

        f, node, dist, energy = heapq.heappop(pq)

        if node == goal:
            return dist, energy, parent

        # Skip outdated entries
        if dist > best_dist.get(node, float('inf')):
            continue

        for nbr in G[node]:

            edge_dist = Dist[f"{node},{nbr}"]
            edge_energy = Cost[f"{node},{nbr}"]

            new_dist = dist + edge_dist
            new_energy = energy + edge_energy

            # Respect energy constraint
            if new_energy > budget:
                continue

            # Check if better path found
            if nbr not in best_dist or new_dist < best_dist[nbr]:

                best_dist[nbr] = new_dist
                parent[nbr] = node

                h = heuristic(nbr, goal, Coord)
                f_new = new_dist + h

                heapq.heappush(pq, (f_new, nbr, new_dist, new_energy))

    return None


# ----------------------------
# Reconstruct path from parents
# ----------------------------
def reconstruct(parent, start, goal):

    path = []
    node = goal

    while node != start:
        path.append(node)

        if node not in parent:
            return []   # no path found

        node = parent[node]

    path.append(start)
    path.reverse()

    return path


# ----------------------------
# Main execution
# ----------------------------
start = '1'
goal = '50'
energy_budget = 287932


result = astar(G, Dist, Cost, Coord, start, goal, energy_budget)

if result is None:
    print("No feasible path found.")
else:
    distance, energy, parent = result

    path = reconstruct(parent, start, goal)

    print("Shortest path:", " -> ".join(path))
    print("Shortest distance:", distance)
    print("Total energy cost:", energy)