import numpy as np

# ─── Environment Setup ───────────────────────────────────────────────────────

GRID_SIZE = 5
GAMMA = 0.9
THETA = 1e-6

ACTIONS = {
    'U': (0, 1),
    'D': (0, -1),
    'L': (-1, 0),
    'R': (1, 0),
}
ACTION_LIST = ['U', 'D', 'L', 'R']

PERP = {
    'U': ['L', 'R'],
    'D': ['L', 'R'],
    'L': ['U', 'D'],
    'R': ['U', 'D'],
}

START = (0, 0)
GOAL  = (4, 4)
ROADBLOCKS = {(2, 1), (2, 3)}

def is_valid(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) not in ROADBLOCKS

def get_states():
    return [(x, y) for x in range(GRID_SIZE)
                   for y in range(GRID_SIZE)
                   if (x, y) not in ROADBLOCKS]

def move(state, action):
    dx, dy = ACTIONS[action]
    nx, ny = state[0] + dx, state[1] + dy
    if is_valid(nx, ny):
        return (nx, ny)
    return state

def get_transitions(state, action):
    """Returns list of (next_state, probability) using the 0.8/0.1/0.1 model."""
    if state == GOAL:
        return [(GOAL, 1.0)]
    p1, p2 = PERP[action]
    outcomes = [
        (move(state, action), 0.8),
        (move(state, p1),     0.1),
        (move(state, p2),     0.1),
    ]
    # Merge probabilities if two slips land on the same next state
    merged = {}
    for ns, p in outcomes:
        merged[ns] = merged.get(ns, 0.0) + p
    return list(merged.items())

def step_reward(state, next_state):
    if state == GOAL:
        return 0
    if next_state == GOAL:
        return -1 + 10  # step cost + goal bonus
    return -1

# ─── Value Iteration ─────────────────────────────────────────────────────────

def value_iteration():
    states = get_states()
    V = {s: 0.0 for s in states}

    iteration = 0
    while True:
        delta = 0
        new_V = {}
        for s in states:
            if s == GOAL:
                new_V[s] = 0.0
                continue
            # V(s) = max_a Σ p(s'|s,a) [R(s,s') + γ·V(s')]
            q_values = []
            for a in ACTION_LIST:
                q = sum(p * (step_reward(s, ns) + GAMMA * V[ns])
                        for ns, p in get_transitions(s, a))
                q_values.append(q)
            new_V[s] = max(q_values)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        iteration += 1
        if delta < THETA:
            break

    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = None
            continue
        q_values = {a: sum(p * (step_reward(s, ns) + GAMMA * V[ns])
                           for ns, p in get_transitions(s, a))
                    for a in ACTION_LIST}
        policy[s] = max(q_values, key=q_values.get)

    print(f"Value Iteration converged in {iteration} iterations.\n")
    return V, policy

# ─── Policy Iteration ────────────────────────────────────────────────────────

def policy_evaluation(policy, V):
    states = get_states()
    while True:
        delta = 0
        for s in states:
            if s == GOAL:
                continue
            a = policy[s]
            v_new = sum(p * (step_reward(s, ns) + GAMMA * V[ns])
                        for ns, p in get_transitions(s, a))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < THETA:
            break
    return V

def policy_iteration():
    states = get_states()
    policy = {s: 'U' for s in states}
    policy[GOAL] = None
    V = {s: 0.0 for s in states}

    iteration = 0
    while True:
        V = policy_evaluation(policy, V)

        policy_stable = True
        for s in states:
            if s == GOAL:
                continue
            old_action = policy[s]
            q_values = {a: sum(p * (step_reward(s, ns) + GAMMA * V[ns])
                               for ns, p in get_transitions(s, a))
                        for a in ACTION_LIST}
            best_action = max(q_values, key=q_values.get)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        iteration += 1
        if policy_stable:
            break

    print(f"Policy Iteration converged in {iteration} iterations.\n")
    return V, policy

# ─── Display Helpers ─────────────────────────────────────────────────────────

ARROW = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', None: 'G'}

def print_value_function(V, title):
    print(f"{'─'*52}")
    print(f"  {title}")
    print(f"{'─'*52}")
    print(f"        x=0     x=1     x=2     x=3     x=4")
    for y in range(GRID_SIZE - 1, -1, -1):
        row = ""
        for x in range(GRID_SIZE):
            if (x, y) in ROADBLOCKS:
                row += "   ████ "
            else:
                row += f"  {V.get((x, y), 0):6.2f}"
        print(f"  y={y}  |{row}|")
    print()

def print_policy(policy, title):
    print(f"{'─'*34}")
    print(f"  {title}")
    print(f"{'─'*34}")
    print(f"      x=0  x=1  x=2  x=3  x=4")
    for y in range(GRID_SIZE - 1, -1, -1):
        row = ""
        for x in range(GRID_SIZE):
            if (x, y) in ROADBLOCKS:
                row += "  ██ "
            else:
                a = policy.get((x, y))
                row += f"   {ARROW[a]} "
        print(f"  y={y} |{row}|")
    print()

def compare_policies(p1, p2, label1, label2):
    states = get_states()
    diffs = [(s, p1[s], p2[s]) for s in states
             if s != GOAL and p1[s] != p2[s]]
    if not diffs:
        print(f"✓ {label1} and {label2} produce identical policies.\n")
    else:
        print(f"Differences between {label1} and {label2}:")
        for s, a1, a2 in diffs:
            print(f"  State {s}: {label1}={ARROW[a1]}, {label2}={ARROW[a2]}")
        print()

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== VALUE ITERATION ===\n")
    V_vi, policy_vi = value_iteration()
    print_value_function(V_vi, "Value Function (VI)")
    print_policy(policy_vi, "Optimal Policy (VI)")

    print("\n=== POLICY ITERATION ===\n")
    V_pi, policy_pi = policy_iteration()
    print_value_function(V_pi, "Value Function (PI)")
    print_policy(policy_pi, "Optimal Policy (PI)")

    print("\n=== COMPARISON ===\n")
    compare_policies(policy_vi, policy_pi, "VI", "PI")