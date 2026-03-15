"""
SC3000 Lab Assignment 1
Part 2 — Grid World: Tasks 1, 2, 3
====================================
Grid World: 5×5
Start: (0,0)  |  Goal: (4,4)  |  Roadblocks: (2,1), (2,3)
Discount factor γ = 0.9
Stochastic transitions: 0.8 intended, 0.1 perp-left, 0.1 perp-right

Task 1: Value Iteration + Policy Iteration (model known, stochastic)
Task 2: Monte Carlo Control (model-free, ε=0.1 fixed, first-visit)
Task 3: Q-Learning (model-free, ε=0.1 fixed, α=0.1 fixed)
"""

import random
import copy
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────────────────
#  Grid World Constants
# ─────────────────────────────────────────────────────────
GRID_SIZE    = 5
START        = (0, 0)
GOAL         = (4, 4)
ROADBLOCKS   = {(2, 1), (2, 3)}

GAMMA        = 0.9      # discount factor
EPSILON      = 0.1      # ε-greedy exploration (fixed)
ALPHA        = 0.1      # Q-learning learning rate (fixed)
THETA        = 1e-6     # convergence threshold for VI / PI
NUM_EPISODES = 500000   # training episodes for MC and Q-Learning
MAX_STEPS    = 500      # max steps per episode

ACTIONS = ['U', 'D', 'L', 'R']

ACTION_DELTA = {
    'U': ( 0,  1),
    'D': ( 0, -1),
    'L': (-1,  0),
    'R': ( 1,  0),
}

# Perpendicular actions:
#   U → perp: L, R  |  D → perp: L, R
#   L → perp: U, D  |  R → perp: U, D
PERP = {
    'U': ('L', 'R'),
    'D': ('L', 'R'),
    'L': ('U', 'D'),
    'R': ('U', 'D'),
}


# ─────────────────────────────────────────────────────────
#  Shared Environment Helpers
# ─────────────────────────────────────────────────────────

def all_states():
    return [
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if (x, y) not in ROADBLOCKS
    ]

def is_valid(x, y):
    return (0 <= x < GRID_SIZE and
            0 <= y < GRID_SIZE and
            (x, y) not in ROADBLOCKS)

def move(state, action):
    dx, dy = ACTION_DELTA[action]
    nx, ny = state[0] + dx, state[1] + dy
    if is_valid(nx, ny):
        return (nx, ny)
    return state

def stochastic_transition(state, action):
    perp_left, perp_right = PERP[action]
    transitions = [
        (action,     0.8),
        (perp_left,  0.1),
        (perp_right, 0.1),
    ]
    r = random.random()
    cumulative = 0.0
    for a, p in transitions:
        cumulative += p
        if r <= cumulative:
            return move(state, a)
    return move(state, action)  

def env_step(state, action):
    next_state = stochastic_transition(state, action)
    if next_state == GOAL:
        return next_state, 10.0, True
    return next_state, -1.0, False

def get_transitions(state, action):
    if state == GOAL:
        return [(GOAL, 1.0)]
    p1, p2 = PERP[action]
    outcomes = [
        (move(state, action), 0.8),
        (move(state, p1),     0.1),
        (move(state, p2),     0.1),
    ]
    merged = {}
    for ns, p in outcomes:
        merged[ns] = merged.get(ns, 0.0) + p
    return list(merged.items())

def step_reward(state, next_state):
    if state == GOAL:
        return 0
    if next_state == GOAL:
        return -1 + 10  
    return -1

def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_vals = [Q[(state, a)] for a in ACTIONS]
    max_q  = max(q_vals)
    best   = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
    return random.choice(best)


# ─────────────────────────────────────────────────────────
#  TASK 1A: Value Iteration
# ─────────────────────────────────────────────────────────

def value_iteration():
    states = all_states()
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
            for a in ACTIONS:
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
            policy[s] = 'GOAL'
            continue
        q_values = {a: sum(p * (step_reward(s, ns) + GAMMA * V[ns])
                           for ns, p in get_transitions(s, a))
                    for a in ACTIONS}
        policy[s] = max(q_values, key=q_values.get)

    print(f"  Value Iteration converged in {iteration} iterations.")
    return V, policy


# ─────────────────────────────────────────────────────────
#  TASK 1B: Policy Iteration
# ─────────────────────────────────────────────────────────

def policy_evaluation(policy, V):
    states = all_states()
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
    states = all_states()
    policy = {s: 'U' for s in states}
    policy[GOAL] = 'GOAL'
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
                        for a in ACTIONS}
            best_action = max(q_values, key=q_values.get)
            policy[s] = best_action
            if best_action != old_action:
                policy_stable = False

        iteration += 1
        if policy_stable:
            break

    print(f"  Policy Iteration converged in {iteration} iterations.")
    return V, policy


# ─────────────────────────────────────────────────────────
#  TASK 2: Monte Carlo Control
#  Model-free | ε=0.1 fixed | First-Visit | from START
# ─────────────────────────────────────────────────────────

def generate_episode(Q, epsilon):
    episode = []
    state   = START
    for _ in range(MAX_STEPS):
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done = env_step(state, action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

def mc_control(num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA):
    Q             = defaultdict(float)
    returns_sum   = defaultdict(float)
    returns_count = defaultdict(int)

    print(f"\n  Starting MC training: {num_episodes} episodes ...")

    for ep in range(1, num_episodes + 1):
        episode = generate_episode(Q, epsilon)

        # First-Visit MC: backward pass
        G       = 0.0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G  = gamma * G + r
            sa = (s, a)

            if sa not in visited:           # first-visit check
                visited.add(sa)
                returns_sum[sa]   += G
                returns_count[sa] += 1
                Q[sa] = returns_sum[sa] / returns_count[sa]

        if ep % 100000 == 0:
            print(f"    Episode {ep:>7} / {num_episodes} complete.")

    # Extract greedy policy from learned Q
    policy = {}
    for s in all_states():
        if s == GOAL:
            policy[s] = 'GOAL'
            continue
        policy[s] = max(ACTIONS, key=lambda a: Q[(s, a)])

    return Q, policy


# ─────────────────────────────────────────────────────────
#  TASK 3: Q-Learning
#  Model-free | ε=0.1 fixed | α=0.1 fixed | from START
# ─────────────────────────────────────────────────────────

def q_learning(num_episodes=NUM_EPISODES, epsilon=EPSILON,
               alpha=ALPHA, gamma=GAMMA):
    Q = defaultdict(float)  # Q[(state, action)] initialised to 0.0

    print(f"\n  Starting Q-Learning training: {num_episodes} episodes ...")

    for ep in range(1, num_episodes + 1):
        state = START   # always start from (0,0) per assignment

        for _ in range(MAX_STEPS):
            # 1. Select action: fixed ε-greedy (ε=0.1)
            action = epsilon_greedy(Q, state, epsilon)

            # 2. Interact with stochastic environment
            next_state, reward, done = env_step(state, action)

            # 3. Bellman TD update with fixed α=0.1
            best_next          = max(Q[(next_state, a)] for a in ACTIONS)
            Q[(state, action)] += alpha * (
                reward + gamma * best_next - Q[(state, action)]
            )

            state = next_state
            if done:
                break

        if ep % 100000 == 0:
            print(f"    Episode {ep:>7} / {num_episodes} complete.")

    # Extract greedy policy from learned Q
    policy = {}
    for s in all_states():
        if s == GOAL:
            policy[s] = 'GOAL'
            continue
        policy[s] = max(ACTIONS, key=lambda a: Q[(s, a)])

    return Q, policy


# ─────────────────────────────────────────────────────────
#  Display Helpers
# ─────────────────────────────────────────────────────────

ARROW = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→', 'GOAL': ' G'}

def print_policy_grid(policy, label="Policy"):
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print("  +" + "-----+" * GRID_SIZE)
    for y in range(GRID_SIZE - 1, -1, -1):
        row = "  |"
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += "  X  |"
            else:
                a   = policy.get(s, '?')
                sym = ARROW.get(a, a)
                row += f"  {sym}  |"
        print(row)
        print("  +" + "-----+" * GRID_SIZE)
    print("       x=0   x=1   x=2   x=3   x=4")
    print("  (rows: y=4 top → y=0 bottom)\n")

def print_value_function(V, label="Value Function"):
    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"{'='*48}")
    for y in range(GRID_SIZE - 1, -1, -1):
        row = f"  y={y} |"
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += "   X  "
            elif s == GOAL:
                row += "   G   "
            else:
                row += f" {V.get(s, 0.0):5.2f} "
        print(row)
    print("        " + "  x=0  " * GRID_SIZE)

def print_q_table(Q, label="Q-Table"):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  {'State':<10} {'Q(U)':>8} {'Q(D)':>8} {'Q(L)':>8} {'Q(R)':>8}  Best")
    print(f"  {'-'*62}")
    for y in range(GRID_SIZE - 1, -1, -1):
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                continue
            if s == GOAL:
                print(f"  {str(s):<10}   --- GOAL STATE ---")
                continue
            q_vals = [Q[(s, a)] for a in ACTIONS]
            best_a = ACTIONS[q_vals.index(max(q_vals))]
            print(
                f"  {str(s):<10} {q_vals[0]:>8.3f} {q_vals[1]:>8.3f}"
                f" {q_vals[2]:>8.3f} {q_vals[3]:>8.3f}  {ARROW[best_a]}"
            )

def compare_policies(p1, p2, label1="Policy 1", label2="Policy 2"):
    states   = all_states()
    non_goal = [s for s in states if s != GOAL]
    diffs    = [(s, p1.get(s), p2.get(s))
                for s in non_goal if p1.get(s) != p2.get(s)]
    agree    = len(non_goal) - len(diffs)
    total    = len(non_goal)

    print(f"\n{'='*52}")
    print(f"  {label1} vs {label2}")
    print(f"{'='*52}")
    print(f"  Agreement : {agree}/{total} states  ({100*agree/total:.1f}%)")
    if diffs:
        print(f"  Differing states ({len(diffs)}):")
        for s, a1, a2 in diffs:
            print(f"    State {s}: {label1}={ARROW.get(a1,a1)}   {label2}={ARROW.get(a2,a2)}")
    else:
        print(f"  ✓ Policies are IDENTICAL.")

def simulate_greedy(Q, label="Policy", num_runs=1000):
    successes    = 0
    example_path = None

    for i in range(num_runs):
        state = START
        path  = [state]
        for _ in range(MAX_STEPS):
            if state == GOAL:
                break
            q_vals = [Q[(state, a)] for a in ACTIONS]
            best_a = ACTIONS[q_vals.index(max(q_vals))]
            next_state, _, done = env_step(state, best_a)
            state = next_state
            path.append(state)
            if done:
                break
        if state == GOAL:
            successes += 1
            if example_path is None:
                example_path = path

    print(f"\n  Greedy Simulation — {label} ({num_runs} runs):")
    print(f"    Success rate : {successes}/{num_runs}  ({successes/num_runs*100:.1f}%)")
    if example_path:
        print(f"    Example path : {' → '.join(str(s) for s in example_path)}")


# ─────────────────────────────────────────────────────────
#  Convergence Analysis (Task 3)
#  Runs QL and MC side-by-side to compare reward over time
# ─────────────────────────────────────────────────────────

def convergence_analysis(num_episodes=100000, interval=10000):
    Q_ql   = defaultdict(float)
    Q_mc   = defaultdict(float)
    rs_mc  = defaultdict(float)
    rc_mc  = defaultdict(int)

    ql_rewards = []
    mc_rewards = []

    for ep in range(1, num_episodes + 1):
        # --- Q-Learning step ---
        state      = START
        total_r_ql = 0
        for _ in range(MAX_STEPS):
            a           = epsilon_greedy(Q_ql, state, EPSILON)
            ns, r, done = env_step(state, a)
            best_next   = max(Q_ql[(ns, aa)] for aa in ACTIONS)
            Q_ql[(state, a)] += ALPHA * (
                r + GAMMA * best_next - Q_ql[(state, a)]
            )
            total_r_ql += r
            state = ns
            if done:
                break
        ql_rewards.append(total_r_ql)

        # --- MC step ---
        episode    = generate_episode(Q_mc, EPSILON)
        total_r_mc = sum(r for _, _, r in episode)
        mc_rewards.append(total_r_mc)
        visited = set()
        G = 0.0
        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = GAMMA * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                rs_mc[(s, a)]  += G
                rc_mc[(s, a)]  += 1
                Q_mc[(s, a)]    = rs_mc[(s, a)] / rc_mc[(s, a)]

    print(f"\n{'='*52}")
    print("  Convergence Analysis: Q-Learning vs Monte Carlo")
    print(f"{'='*52}")
    print(f"  {'Episode':>10}  {'QL Avg Reward':>15}  {'MC Avg Reward':>15}")
    print("  " + "-" * 46)
    for i in range(0, num_episodes, interval):
        end    = min(i + interval, num_episodes)
        ql_avg = np.mean(ql_rewards[i:end])
        mc_avg = np.mean(mc_rewards[i:end])
        print(f"  {end:>10}  {ql_avg:>15.3f}  {mc_avg:>15.3f}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("  CZ3005 Lab Assignment 1 — Part 2")
    print("  Tasks 1, 2, 3: Grid World MDP & Reinforcement Learning")
    print("=" * 60)
    print(f"\n  Grid       : {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Start      : {START}   |  Goal      : {GOAL}")
    print(f"  Roadblocks : {sorted(ROADBLOCKS)}")
    print(f"  γ (gamma)  : {GAMMA}   |  ε (epsilon): {EPSILON}  |  α (alpha): {ALPHA}")
    print(f"  Episodes   : {NUM_EPISODES}")
    print(f"  Transitions: 0.8 intended | 0.1 perp-left | 0.1 perp-right")

    # ──────────────────────────────────────────────────────
    # TASK 1: Value Iteration & Policy Iteration
    # ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  TASK 1: VALUE ITERATION & POLICY ITERATION")
    print("  (Model known | Stochastic transitions | γ=0.9)")
    print("=" * 60)

    print("\n--- Value Iteration ---")
    V_vi, policy_vi = value_iteration()
    print_value_function(V_vi,   label="Value Function (Value Iteration)")
    print_policy_grid(policy_vi, label="Optimal Policy (Value Iteration)")

    print("\n--- Policy Iteration ---")
    V_pi, policy_pi = policy_iteration()
    print_value_function(V_pi,   label="Value Function (Policy Iteration)")
    print_policy_grid(policy_pi, label="Optimal Policy (Policy Iteration)")

    print("\n--- Comparison: VI vs PI ---")
    compare_policies(policy_vi, policy_pi, "VI", "PI")

    # ──────────────────────────────────────────────────────
    # TASK 2: Monte Carlo Control
    # ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  TASK 2: MONTE CARLO CONTROL")
    print("  (Model-free | ε=0.1 fixed | First-Visit | 500k episodes)")
    print("=" * 60)

    Q_mc, policy_mc = mc_control()

    print("\n  Training complete!")
    print_q_table(Q_mc,     label="Learned Q-Values (MC Control)")
    print_policy_grid(policy_mc, label="Learned Policy — MC Control")

    print("\n--- Comparison: MC vs VI Optimal ---")
    compare_policies(policy_mc, policy_vi, "MC", "VI-Optimal")

    simulate_greedy(Q_mc, label="MC Policy")

    print(f"\n{'='*60}")
    print("  Task 2 Analysis")
    print(f"{'='*60}")
    print(f"""
  Monte Carlo Control learns purely from sampled experience —
  it has NO knowledge of the transition probabilities (0.8/0.1/0.1).

  Algorithm highlights:
  ─ First-Visit MC: Q(s,a) updated only on the FIRST occurrence
    of each (state, action) pair within an episode.
  ─ ε-greedy (ε=0.1): ensures continued exploration throughout.
  ─ Incremental mean update: O(1) per (s,a), memory efficient.
  ─ No bootstrapping: waits until episode end before computing
    returns — higher variance but zero bias vs TD methods.

  Small disagreements with VI optimal policy are normal due to
  sampling variance, especially in rarely-visited states.
  """)

    # ──────────────────────────────────────────────────────
    # TASK 3: Q-Learning
    # ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  TASK 3: Q-LEARNING")
    print("  (Model-free | ε=0.1 fixed | α=0.1 fixed | 500k episodes)")
    print("=" * 60)

    Q_ql, policy_ql = q_learning()

    print("\n  Training complete!")
    print_q_table(Q_ql,     label="Learned Q-Values (Q-Learning)")
    print_policy_grid(policy_ql, label="Learned Policy — Q-Learning")

    print("\n--- Comparison: Q-Learning vs VI Optimal ---")
    compare_policies(policy_ql, policy_vi, "Q-Learning", "VI-Optimal")

    print("\n--- Comparison: Q-Learning vs Monte Carlo ---")
    compare_policies(policy_ql, policy_mc, "Q-Learning", "MC")

    simulate_greedy(Q_ql, label="Q-Learning Policy")

    print(f"\n{'='*60}")
    print("  Task 3 Analysis")
    print(f"{'='*60}")

    # ──────────────────────────────────────────────────────
    # CONVERGENCE ANALYSIS
    # ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  CONVERGENCE ANALYSIS: Q-Learning vs Monte Carlo")
    print("  Average episode reward every 10,000 episodes")
    print("  (higher = better | faster rise = faster convergence)")
    print("=" * 60)
    convergence_analysis(num_episodes=100000, interval=10000)

    print("\n\n  [All Tasks Complete]\n")