import json
import heapq


# ----------------------------
# Load graph data
# ----------------------------
with open('Coord.json') as a, open('Cost.json') as b, open('Dist.json') as c, open('G.json') as d:
    Coord = json.load(a)
    Cost = json.load(b)
    Dist = json.load(c)
    G = json.load(d)


# ----------------------------
# Uniform Cost Search (UCS)
# with energy budget constraint
# ----------------------------
def ucs(G, Dist, Cost, start, goal, budget):

    pq = []
    counter = 0
    heapq.heappush(pq, (0, counter, start, 0))
    # (distance_so_far, tiebreaker, node, energy_so_far)

    # best[node] = (best_dist, best_energy)
    # We may revisit a node if we arrive with less energy used
    best = {start: (0, 0)}
    parent = {start: None}

    while pq:

        dist, _, node, energy = heapq.heappop(pq)

        if node == goal:
            return dist, energy, parent

        # Skip if we already found a strictly better way to reach this node
        if node in best:
            best_dist, best_energy = best[node]
            if dist > best_dist and energy >= best_energy:
                continue

        for nbr in G.get(node, []):

            edge_dist = Dist.get(f"{node},{nbr}")
            edge_energy = Cost.get(f"{node},{nbr}")

            if edge_dist is None or edge_energy is None:
                continue

            new_dist = dist + edge_dist
            new_energy = energy + edge_energy

            # Respect energy constraint
            if new_energy > budget:
                continue

            # A state is worth exploring if:
            # 1. We haven't visited this neighbour yet, OR
            # 2. We found a shorter distance, OR
            # 3. We found the same or longer distance but with less energy
            #    (which may allow reaching the goal via paths that were
            #     previously pruned by the energy constraint)
            if nbr not in best:
                best[nbr] = (new_dist, new_energy)
                parent[nbr] = node
                counter += 1
                heapq.heappush(pq, (new_dist, counter, nbr, new_energy))
            else:
                prev_dist, prev_energy = best[nbr]
                if new_dist < prev_dist or new_energy < prev_energy:
                    if new_dist <= prev_dist:
                        best[nbr] = (new_dist, min(new_energy, prev_energy))
                        parent[nbr] = node
                    elif new_energy < prev_energy:
                        # Keep the old parent for shorter dist, but still explore
                        best[nbr] = (min(new_dist, prev_dist), new_energy)
                    counter += 1
                    heapq.heappush(pq, (new_dist, counter, nbr, new_energy))

    return None


# ----------------------------
# Reconstruct path from parents
# ----------------------------
def reconstruct(parent, start, goal):

    path = []
    node = goal

    while node != start:
        path.append(node)

        if node not in parent or parent[node] is None:
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


result = ucs(G, Dist, Cost, start, goal, energy_budget)

if result is None:
    print("No feasible path found.")
else:
    distance, energy, parent = result

    path = reconstruct(parent, start, goal)

    print("Shortest path:", " -> ".join(path))
    print("Shortest distance:", distance)
    print("Total energy cost:", energy)
