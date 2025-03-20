import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque

# Step 1: Generate random points and build the adjacency graph
def generate_points(n):
    return np.random.rand(n, 2)

def segments_cross(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    o1 = ccw(p1, p2, q1)
    o2 = ccw(p1, p2, q2)
    o3 = ccw(q1, q2, p1)
    o4 = ccw(q1, q2, p2)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return False

def build_graph(points):
    n = len(points)
    adj = {i: set() for i in range(n)}
    edges = []
    points_list = [tuple(p) for p in points]
    
    while True:
        available = [i for i in range(n) if len(adj[i]) < n-1]
        if not available:
            break
        x = random.choice(available)
        candidates = []
        for y in range(n):
            if y == x or y in adj[x]:
                continue
            new_edge = (x, y) if x < y else (y, x)
            edge_points = (points[x], points[y])
            cross = False
            for (u, v) in edges:
                if segments_cross(points[u], points[v], points[x], points[y]):
                    cross = True
                    break
            if not cross:
                candidates.append((y, np.linalg.norm(points[x] - points[y])))
        if not candidates:
            continue
        y = min(candidates, key=lambda t: t[1])[0]
        adj[x].add(y)
        adj[y].add(x)
        edges.append((x, y) if x < y else (y, x))
    return adj

# Step 2: Coloring algorithms
def min_conflicts(graph, k, max_steps=10000):
    assignment = {node: random.randint(1, k) for node in graph}
    for _ in range(max_steps):
        conflicted = [node for node in graph if any(assignment[node] == assignment[nei] for nei in graph[node])]
        if not conflicted:
            return assignment
        node = random.choice(conflicted)
        min_color = assignment[node]
        min_count = sum(1 for nei in graph[node] if assignment[nei] == min_color)
        for color in range(1, k+1):
            if color == assignment[node]:
                continue
            count = sum(1 for nei in graph[node] if assignment[nei] == color)
            if count < min_count:
                min_color = color
                min_count = count
        assignment[node] = min_color
    return None

def backtracking(graph, k):
    nodes = list(graph.keys())
    def backtrack(assignment, node_idx):
        if node_idx == len(nodes):
            return assignment
        node = nodes[node_idx]
        for color in range(1, k+1):
            if all(assignment.get(nei, -1) != color for nei in graph[node]):
                assignment[node] = color
                result = backtrack(assignment, node_idx + 1)
                if result is not None:
                    return result
                del assignment[node]
        return None
    return backtrack({}, 0)

def backtracking_fc(graph, k):
    nodes = list(graph.keys())
    domains = {node: list(range(1, k+1)) for node in nodes}
    def forward_check(node, color, domains):
        new_domains = {n: list(d) for n, d in domains.items()}
        for nei in graph[node]:
            if color in new_domains[nei]:
                new_domains[nei].remove(color)
                if not new_domains[nei]:
                    return None
        return new_domains
    def backtrack(assignment, node_idx, domains):
        if node_idx == len(nodes):
            return assignment
        node = nodes[node_idx]
        for color in domains[node]:
            if all(assignment.get(nei, -1) != color for nei in graph[node]):
                new_assignment = assignment.copy()
                new_assignment[node] = color
                new_domains = forward_check(node, color, domains)
                if new_domains is None:
                    continue
                result = backtrack(new_assignment, node_idx + 1, new_domains)
                if result is not None:
                    return result
        return None
    return backtrack({}, 0, domains)

def backtracking_mac(graph, k):
    nodes = list(graph.keys())
    domains = {node: list(range(1, k+1)) for node in nodes}
    def revise(x, y, domains):
        revised = False
        for color_x in list(domains[x]):
            if all(color_x == color_y for color_y in domains[y]):
                domains[x].remove(color_x)
                revised = True
        return revised
    def ac3(domains, node=None):
        queue = deque()
        if node is not None:
            for nei in graph[node]:
                queue.append((nei, node))
        else:
            for x in graph:
                for y in graph[x]:
                    queue.append((x, y))
        while queue:
            x, y = queue.popleft()
            if revise(x, y, domains):
                if not domains[x]:
                    return False
                for nei in graph[x]:
                    if nei != y:
                        queue.append((nei, x))
        return True
    if not ac3(domains):
        return None
    def backtrack(assignment, node_idx):
        if node_idx == len(nodes):
            return assignment
        node = nodes[node_idx]
        original_domain = domains[node].copy()
        for color in original_domain:
            if any(assignment.get(nei, -1) == color for nei in graph[node]):
                continue
            assignment[node] = color
            old_domains = {n: list(d) for n, d in domains.items()}
            domains[node] = [color]
            if ac3(domains, node):
                result = backtrack(assignment, node_idx + 1)
                if result is not None:
                    return result
            domains.update(old_domains)
            del assignment[node]
        return None
    return backtrack({}, 0)

# Step 3: Benchmarking
def benchmark(n_values, k_values, algorithms, runs=5):
    results = {n: {alg: {k: [] for k in k_values} for alg in algorithms} for n in n_values}
    for n in n_values:
        for _ in range(runs):
            points = generate_points(n)
            graph = build_graph(points)
            for k in k_values:
                for alg_name in algorithms:
                    start = time.time()
                    if alg_name == 'min_conflicts':
                        result = min_conflicts(graph, k)
                    elif alg_name == 'backtracking':
                        result = backtracking(graph, k)
                    elif alg_name == 'backtracking_fc':
                        result = backtracking_fc(graph, k)
                    elif alg_name == 'backtracking_mac':
                        result = backtracking_mac(graph, k)
                    else:
                        raise ValueError("Unknown algorithm")
                    elapsed = time.time() - start
                    if result is not None:
                        results[n][alg_name][k].append(elapsed)
    # Compute averages
    avg_results = {}
    for n in n_values:
        avg_results[n] = {}
        for alg in algorithms:
            avg_results[n][alg] = {}
            for k in k_values:
                times = results[n][alg][k]
                avg = sum(times)/len(times) if times else float('inf')
                avg_results[n][alg][k] = avg
    return avg_results

# Parameters
n_values = [5, 8, 10]  # Adjust based on computational limits
k_values = [3, 4]
algorithms = ['min_conflicts', 'backtracking', 'backtracking_fc', 'backtracking_mac']

# Run benchmark
results = benchmark(n_values, k_values, algorithms, runs=3)

# Print results
print("Average runtimes (seconds):")
for n in n_values:
    print(f"\nN = {n}")
    print(f"{'Algorithm':<20} | {'k=3':<10} | {'k=4':<10}")
    print("-" * 45)
    for alg in algorithms:
        row = f"{alg:<20} | {results[n][alg][3]:<10.4f} | {results[n][alg][4]:<10.4f}"
        print(row)