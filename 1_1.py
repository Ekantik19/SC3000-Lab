#reading json files
import json
import heapq
import math
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).resolve().parent

with open(base_dir / 'Coord.json') as a, open(base_dir / 'Cost.json') as b, open(base_dir / 'Dist.json') as c, open(base_dir / 'G.json') as d:
    Coord = json.load(a)
    Cost = json.load(b)
    Dist = json.load(c)
    G = json.load(d)

start = '1'
goal = '50'

def heuristic(node, goal, Coord):
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def astar(G, Dist, Coord, start, goal):
    heap = [(heuristic(start, goal, Coord), 0, 0, start)]
    counter = 0
    visited = set()
    prev = {start: None}
    g_score = {start: 0}
    while heap:
        f, _, g, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            break
        for neighbour in G.get(node, []):
            if neighbour in visited:
                continue
            edge_dist = Dist.get(f'{node},{neighbour}')
            if edge_dist is None:
                continue
            new_g = g + edge_dist
            if new_g < g_score.get(neighbour, float('inf')):
                g_score[neighbour] = new_g
                prev[neighbour] = node
                counter += 1
                heapq.heappush(heap, (new_g + heuristic(neighbour, goal, Coord), counter, new_g, neighbour))
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path, g_score.get(goal, float('inf')), visited

a_path, a_dist, a_visited = astar(G, Dist, Coord, start, goal)

print(f"Shortest path: {'->'.join(a_path)}")
print(f"Shortest distance: {a_dist}")
print(f"A* explored:       {len(a_visited)} nodes | distance: {a_dist}")
