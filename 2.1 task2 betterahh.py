"""
CZ3005 / SC3005 Lab Assignment 1
Part 2 — Task 2: Monte Carlo (MC) Control
==========================================
Grid World: 5×5
Start: (0,0)  |  Goal: (4,4)  |  Roadblocks: (2,1), (2,3)
Discount factor γ = 0.9
Exploration rate ε = 0.1 (fixed)
Stochastic transitions: 0.8 intended, 0.1 perp-left, 0.1 perp-right
(environment is stochastic but transition probabilities are UNKNOWN to agent)

Algorithm: First-Visit MC Control with ε-greedy policy improvement
"""

import random
import copy
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
NUM_EPISODES = 500000   # number of training episodes
MAX_STEPS    = 500      # max steps per episode (prevents infinite loops)

ACTIONS = ['U', 'D', 'L', 'R']

# Deterministic displacement for each action
ACTION_DELTA = {
    'U': ( 0,  1),
    'D': ( 0, -1),
    'L': (-1,  0),
    'R': ( 1,  0),
}

# Perpendicular actions as clarified by the professor:
#   U → perp: L, R
#   D → perp: L, R
#   L → perp: U, D
#   R → perp: U, D
PERP = {
    'U': ('L', 'R'),
    'D': ('L', 'R'),
    'L': ('U', 'D'),
    'R': ('U', 'D'),
}


# ─────────────────────────────────────────────────────────
#  Helper: all valid states
# ─────────────────────────────────────────────────────────
def all_states():
    return [
        (x, y)
        for x in range(GRID_SIZE)
        for y in range(GRID_SIZE)
        if (x, y) not in ROADBLOCKS
    ]


# ─────────────────────────────────────────────────────────
#  Deterministic move (bounds + roadblock check)
# ─────────────────────────────────────────────────────────
def move(state, action):
    """Apply action deterministically. Returns same state if move is invalid."""
    dx, dy = ACTION_DELTA[action]
    nx, ny = state[0] + dx, state[1] + dy
    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in ROADBLOCKS:
        return (nx, ny)
    return state  # stay in place if out of bounds or roadblock


# ─────────────────────────────────────────────────────────
#  Stochastic Environment Transition
#  (Hidden from the agent — this IS the environment)
#
#  Prof's clarification:
#    U → 0.8 U, 0.1 L, 0.1 R
#    D → 0.8 D, 0.1 L, 0.1 R
#    L → 0.8 L, 0.1 U, 0.1 D
#    R → 0.8 R, 0.1 U, 0.1 D
# ─────────────────────────────────────────────────────────
def stochastic_transition(state, action):
    """
    Samples next state from the stochastic environment.
    The agent does NOT have access to this function directly —
    it only observes the resulting (next_state, reward) pair.
    """
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
    return move(state, action)  # numerical fallback


# ─────────────────────────────────────────────────────────
#  Environment Step
# ─────────────────────────────────────────────────────────
def env_step(state, action):
    """
    Agent calls this to interact with the environment.
    Returns: (next_state, reward, done)
    """
    next_state = stochastic_transition(state, action)
    if next_state == GOAL:
        return next_state, 10.0, True
    return next_state, -1.0, False


# ─────────────────────────────────────────────────────────
#  ε-Greedy Action Selection
# ─────────────────────────────────────────────────────────
def epsilon_greedy(Q, state, epsilon):
    """
    With probability ε  → choose a random action (explore)
    With probability 1-ε → choose the greedy action (exploit)
    Ties broken randomly.
    """
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    q_vals = [Q[(state, a)] for a in ACTIONS]
    max_q  = max(q_vals)
    best   = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
    return random.choice(best)


# ─────────────────────────────────────────────────────────
#  Generate One Episode
# ─────────────────────────────────────────────────────────
def generate_episode(Q, epsilon):
    """
    Runs one episode from START using the ε-greedy policy.
    Returns list of (state, action, reward) tuples.
    """
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


# ─────────────────────────────────────────────────────────
#  First-Visit Monte Carlo Control
# ─────────────────────────────────────────────────────────
def mc_control(num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA):
    """
    First-Visit MC Control with ε-greedy policy improvement.

    For each episode:
      1. Generate episode using current ε-greedy policy.
      2. Compute discounted return G backwards.
      3. On first visit of each (s,a) pair, update Q(s,a)
         using an incremental mean (O(1) per update, memory efficient).

    Returns: Q (state-action values), policy (greedy over Q)
    """
    # Q(s,a) initialised to 0.0
    Q             = defaultdict(float)
    # Incremental mean: track sum and count separately
    returns_sum   = defaultdict(float)
    returns_count = defaultdict(int)

    print(f"\n  Starting training: {num_episodes} episodes ...")

    for ep in range(1, num_episodes + 1):

        episode = generate_episode(Q, epsilon)

        # ── First-Visit MC return computation (backward pass) ──
        G        = 0.0
        visited  = set()  # tracks (state, action) pairs seen in this episode

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G  = gamma * G + r
            sa = (s, a)

            if sa not in visited:          # first-visit check
                visited.add(sa)
                returns_sum[sa]   += G
                returns_count[sa] += 1
                # Incremental mean — O(1), no growing list
                Q[sa] = returns_sum[sa] / returns_count[sa]

        if ep % 100000 == 0:
            print(f"    Episode {ep:>7} / {num_episodes} complete.")

    # ── Extract greedy policy from learned Q ──
    policy = {}
    for s in all_states():
        if s == GOAL:
            policy[s] = 'GOAL'
            continue
        policy[s] = max(ACTIONS, key=lambda a: Q[(s, a)])

    return Q, policy


# ─────────────────────────────────────────────────────────
#  Reference: Value Iteration (Task 1 optimal policy)
#  Included here for direct comparison with MC learned policy
# ─────────────────────────────────────────────────────────
def value_iteration(gamma=GAMMA, theta=1e-8):
    """
    Computes the optimal policy via Value Iteration.
    The agent in Task 2 does NOT use this — it is only used here
    as a reference to compare the MC-learned policy against.
    """
    states = all_states()
    V      = {s: 0.0 for s in states}

    def transitions(state, action):
        """Returns list of (next_state, prob, reward)."""
        if state == GOAL:
            return []
        perp_left, perp_right = PERP[action]
        raw = [(action, 0.8), (perp_left, 0.1), (perp_right, 0.1)]
        merged = {}
        for a, p in raw:
            ns = move(state, a)
            r  = 10.0 if ns == GOAL else -1.0
            if ns not in merged:
                merged[ns] = [0.0, r]
            merged[ns][0] += p
        return [(ns, d[0], d[1]) for ns, d in merged.items()]

    # Bellman optimality iteration
    while True:
        delta = 0.0
        V_new = copy.copy(V)
        for s in states:
            if s == GOAL:
                V_new[s] = 0.0
                continue
            best = max(
                sum(p * (r + gamma * V[ns]) for ns, p, r in transitions(s, a))
                for a in ACTIONS
            )
            delta    = max(delta, abs(best - V[s]))
            V_new[s] = best
        V = V_new
        if delta < theta:
            break

    # Extract greedy policy
    policy = {}
    for s in states:
        if s == GOAL:
            policy[s] = 'GOAL'
            continue
        policy[s] = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[ns]) for ns, p, r in transitions(s, a))
        )

    return V, policy


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
        row = f"  |"
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += "  ██ |"
            else:
                a = policy.get(s, '?')
                sym = ARROW.get(a, a)
                row += f"  {sym}  |"
        print(row)
        print("  +" + "-----+" * GRID_SIZE)
    print("       x=0   x=1   x=2   x=3   x=4")
    print("  (rows: y=4 top → y=0 bottom)\n")


def print_q_table(Q):
    print(f"\n{'='*70}")
    print("  Learned Q-Values (MC Control)")
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


def print_value_function(V, label="Value Function"):
    print(f"\n{'='*48}")
    print(f"  {label}")
    print(f"{'='*48}")
    for y in range(GRID_SIZE - 1, -1, -1):
        row = f"  y={y} |"
        for x in range(GRID_SIZE):
            s = (x, y)
            if s in ROADBLOCKS:
                row += "   ██  "
            elif s == GOAL:
                row += "   G   "
            else:
                row += f" {V.get(s, 0.0):5.2f} "
        print(row)
    print("        " + "  x=0  " * GRID_SIZE)


def compare_policies(mc_policy, vi_policy):
    states = all_states()
    non_goal = [s for s in states if s != GOAL]
    diffs = [
        (s, mc_policy.get(s), vi_policy.get(s))
        for s in non_goal
        if mc_policy.get(s) != vi_policy.get(s)
    ]
    agree = len(non_goal) - len(diffs)
    total = len(non_goal)

    print(f"\n{'='*52}")
    print("  MC Policy vs Value Iteration Optimal Policy")
    print(f"{'='*52}")
    print(f"  Agreement : {agree}/{total} states  ({100*agree/total:.1f}%)")
    if diffs:
        print(f"  Differing states ({len(diffs)}):")
        for s, a_mc, a_vi in diffs:
            print(f"    State {s}: MC = {a_mc}   VI-Optimal = {a_vi}")
    else:
        print("  ✓ MC policy is IDENTICAL to the VI optimal policy.")


def simulate_greedy(Q, num_runs=1000):
    """Run multiple greedy simulations, report success rate."""
    successes = 0
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

    print(f"\n  Greedy Policy Simulation ({num_runs} runs):")
    print(f"    Success rate : {successes}/{num_runs}  ({successes/num_runs*100:.1f}%)")
    if example_path:
        print(f"    Example path : {' → '.join(str(s) for s in example_path)}")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(42)

    print("\n" + "=" * 60)
    print("  CZ3005 Lab Assignment 1 — Part 2, Task 2")
    print("  Monte Carlo Control (First-Visit MC, ε-greedy)")
    print("=" * 60)
    print(f"\n  Grid         : {GRID_SIZE}×{GRID_SIZE}")
    print(f"  Start        : {START}      Goal       : {GOAL}")
    print(f"  Roadblocks   : {sorted(ROADBLOCKS)}")
    print(f"  γ (gamma)    : {GAMMA}      ε (epsilon) : {EPSILON}")
    print(f"  Episodes     : {NUM_EPISODES}")
    print(f"  Transitions  : 0.8 intended | 0.1 perp-left | 0.1 perp-right")
    print(f"  (Transition probabilities are UNKNOWN to the agent)")

    # ── Task 2: Train MC Control ──────────────────────────
    Q, mc_policy = mc_control()

    print("\n  Training complete!")
    print_q_table(Q)
    print_policy_grid(mc_policy, label="Learned Policy — MC Control (greedy over Q)")

    # ── Reference: Value Iteration (Task 1) ───────────────
    print("\n  Computing reference policy via Value Iteration (Task 1) ...")
    vi_V, vi_policy = value_iteration()
    print_value_function(vi_V, label="Reference Value Function (Value Iteration)")
    print_policy_grid(vi_policy, label="Reference Policy — Value Iteration (Task 1)")

    # ── Policy Comparison ─────────────────────────────────
    compare_policies(mc_policy, vi_policy)

    # ── Greedy Simulation ─────────────────────────────────
    simulate_greedy(Q, num_runs=1000)

    # ── Analysis ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Analysis")
    print(f"{'='*60}")
    print(f"""
  Monte Carlo Control learns purely from sampled experience —
  it has NO knowledge of the transition probabilities (0.8/0.1/0.1).

  Algorithm highlights:
  ─ First-Visit MC: Q(s,a) is updated only on the FIRST occurrence
    of each (state, action) pair within an episode.
  ─ ε-greedy (ε=0.1): ensures continued exploration of all (s,a)
    pairs throughout training (GLIE-like behaviour).
  ─ Incremental mean update: O(1) per (s,a) update; memory
    efficient since no full returns list is stored.
  ─ No bootstrapping: the agent waits until episode end before
    computing returns, leading to higher variance but zero bias
    compared to TD methods (Task 3 Q-learning).

  Convergence:
  ─ With {NUM_EPISODES:,} episodes the learned policy should closely
    match the VI-optimal policy from Task 1.
  ─ Small disagreements in rarely-visited states are normal due
    to sampling variance inherent in MC methods.
  ─ Increasing NUM_EPISODES further reduces disagreements.
  """)

    print("  [Task 2 Complete]\n")