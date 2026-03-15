import json
import heapq

START = '1'
GOAL = '50'
ENERGY_BUDGET = 287932


# ----------------------------
# Load graph files
# ----------------------------
def load_data():
    with open("G.json") as f:
        G = json.load(f)

    with open("Dist.json") as f:
        Dist = json.load(f)

    with open("Cost.json") as f:
        Cost = json.load(f)

    return G, Dist, Cost


# ----------------------------
# Uniform Cost Search with energy constraint
# ----------------------------
def ucs_energy(G, Dist, Cost):

    pq = []
    heapq.heappush(pq, (0, START, 0, [START]))

    visited = {}

    while pq:

        dist, node, energy, path = heapq.heappop(pq)

        if node == GOAL:
            return path, dist, energy

        if node in visited and visited[node] <= energy:
            continue

        visited[node] = energy

        for neighbor in G[node]:

            edge = f"{node},{neighbor}"

            new_dist = dist + Dist[edge]
            new_energy = energy + Cost[edge]

            if new_energy <= ENERGY_BUDGET:

                heapq.heappush(
                    pq,
                    (new_dist, neighbor, new_energy, path + [neighbor])
                )

    return None, None, None


# ----------------------------
# Main
# ----------------------------
def main():

    G, Dist, Cost = load_data()

    path, distance, energy = ucs_energy(G, Dist, Cost)

    if path:

        print("Shortest path:", "->".join(path))
        print("Shortest distance:", distance)
        print("Total energy cost:", energy)

    else:
        print("No feasible path found")


if __name__ == "__main__":
    main()