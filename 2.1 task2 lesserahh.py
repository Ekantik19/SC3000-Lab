import random
import numpy as np
from collections import defaultdict


# ============================================================
# Part 2, Task 2 — Monte Carlo (MC) Control
# 5x5 Grid World with Stochastic Transitions
# ============================================================


# ----------------------------
# Constants
# ----------------------------
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
ROADBLOCKS = [(2, 1), (2, 3)]
ACTIONS = ["U", "D", "L", "R"]
GAMMA = 0.9          # discount factor
EPSILON = 0.1        # exploration rate (fixed)
NUM_EPISODES = 500000
MAX_STEPS = 200      # max steps per episode to avoid infinite loops


# ----------------------------
# Environment: Movement Logic
# ----------------------------
def move(state, action):
    """Deterministic move given a state and action.
    Returns the same state if the move is invalid."""
    x, y = state
    if action == "U":
        next_state = (x, y + 1)
    elif action == "D":
        next_state = (x, y - 1)
    elif action == "L":
        next_state = (x - 1, y)
    elif action == "R":
        next_state = (x + 1, y)
    else:
        return state

    # Check boundaries and roadblocks
    if (next_state[0] < 0 or next_state[0] > 4 or
        next_state[1] < 0 or next_state[1] > 4 or
        next_state in ROADBLOCKS):
        return state
    return next_state


def stochastic_transition(state, action):
    """Stochastic transition: 80% intended, 10% perp-left, 10% perp-right."""
    if action == "U":
        transitions = [("U", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "D":
        transitions = [("D", 0.8), ("L", 0.1), ("R", 0.1)]
    elif action == "L":
        transitions = [("L", 0.8), ("U", 0.1), ("D", 0.1)]
    elif action == "R":
        transitions = [("R", 0.8), ("U", 0.1), ("D", 0.1)]
    else:
        return state

    r = random.random()
    cum_prob = 0.0
    for a, p in transitions:
        cum_prob += p
        if r <= cum_prob:
            return move(state, a)
    return state


# ----------------------------
# Environment: Step Function
# ----------------------------
def step(state, action):
    """Take a stochastic step in the environment.
    Returns (next_state, reward, done)."""
    next_state = stochastic_transition(state, action)
    if next_state == GOAL:
        return next_state, 10, True
    return next_state, -1, False


# ----------------------------
# Epsilon-Greedy Policy
# ----------------------------
def epsilon_greedy(Q, state, epsilon):
    """Select an action using epsilon-greedy strategy."""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        q_values = [Q[(state, a)] for a in ACTIONS]
        max_q = max(q_values)
        # Break ties randomly
        best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
        return random.choice(best_actions)


# ----------------------------
# Generate an Episode
# ----------------------------
def generate_episode(Q, epsilon):
    """Generate a complete episode using epsilon-greedy policy.
    Returns a list of (state, action, reward) tuples."""
    episode = []
    state = START
    for _ in range(MAX_STEPS):
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done = step(state, action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode


# ----------------------------
# Monte Carlo Control
# (First-Visit MC with epsilon-greedy)
# ----------------------------
def mc_control(num_episodes=NUM_EPISODES, gamma=GAMMA, epsilon=EPSILON):
    """Monte Carlo Control using first-visit MC and epsilon-greedy."""

    # Initialize Q-table to zeros
    Q = defaultdict(float)
    # Count of returns for averaging
    returns_count = defaultdict(float)
    # Sum of returns
    returns_sum = defaultdict(float)

    for ep in range(1, num_episodes + 1):

        episode = generate_episode(Q, epsilon)

        # Calculate returns for each (state, action) pair
        G = 0.0
        visited = set()

        # Traverse episode in reverse to compute returns
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            sa_pair = (state, action)

            # First-visit MC: only update on first occurrence
            if sa_pair not in visited:
                visited.add(sa_pair)
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                Q[sa_pair] = returns_sum[sa_pair] / returns_count[sa_pair]

        # Progress logging
        if ep % 100000 == 0:
            print(f"  Episode {ep}/{num_episodes} completed.")

    return Q


# ----------------------------
# Extract Greedy Policy from Q
# ----------------------------
def extract_policy(Q):
    """Derive the greedy policy from the learned Q-table."""
    policy = {}
    for y in range(GRID_SIZE - 1, -1, -1):
        for x in range(GRID_SIZE):
            state = (x, y)
            if state == GOAL:
                policy[state] = "*"
            elif state in ROADBLOCKS:
                policy[state] = "X"
            else:
                q_values = [Q[(state, a)] for a in ACTIONS]
                best_action = ACTIONS[np.argmax(q_values)]
                policy[state] = best_action
    return policy


# ----------------------------
# Display Policy as Grid
# ----------------------------
def display_policy(policy):
    """Print the policy as a readable grid (top row = y=4)."""
    action_symbols = {"U": "↑", "D": "↓", "L": "←", "R": "→", "*": "★", "X": "▓"}
    print("\n  Learned Policy (MC Control):")
    print("  +" + "-----+" * GRID_SIZE)
    for y in range(GRID_SIZE - 1, -1, -1):
        row = "  |"
        for x in range(GRID_SIZE):
            state = (x, y)
            symbol = action_symbols.get(policy.get(state, "?"), "?")
            action_letter = policy.get(state, "?")
            row += f" {symbol} {action_letter}|"
        print(row)
        print("  +" + "-----+" * GRID_SIZE)
    print(f"  x →  0     1     2     3     4")
    print(f"  (y shown from top=4 to bottom=0)\n")


# ----------------------------
# Display Q-Table
# ----------------------------
def display_q_table(Q):
    """Print the Q-table in a readable format."""
    print("\n  Q-Table (state-action values):")
    print("  " + "-" * 72)
    print(f"  {'State':<10} {'U':>10} {'D':>10} {'L':>10} {'R':>10}")
    print("  " + "-" * 72)
    for y in range(GRID_SIZE - 1, -1, -1):
        for x in range(GRID_SIZE):
            state = (x, y)
            if state in ROADBLOCKS:
                continue
            q_vals = [Q[(state, a)] for a in ACTIONS]
            print(f"  {str(state):<10} {q_vals[0]:>10.2f} {q_vals[1]:>10.2f} {q_vals[2]:>10.2f} {q_vals[3]:>10.2f}")
    print("  " + "-" * 72)


# ----------------------------
# Simulate Agent Using Learned Policy
# ----------------------------
def simulate(Q, max_steps=50):
    """Simulate the agent following the greedy policy (no exploration)."""
    state = START
    path = [state]
    total_reward = 0

    for _ in range(max_steps):
        if state == GOAL:
            break
        q_values = [Q[(state, a)] for a in ACTIONS]
        action = ACTIONS[np.argmax(q_values)]
        next_state, reward, done = step(state, action)
        total_reward += reward
        state = next_state
        path.append(state)
        if done:
            break

    return path, total_reward


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("  Part 2 — Task 2: Monte Carlo (MC) Control")
    print("  5×5 Grid World | ε-greedy | First-Visit MC")
    print("=" * 60)
    print(f"\n  Parameters:")
    print(f"    Episodes   = {NUM_EPISODES}")
    print(f"    Gamma (γ)  = {GAMMA}")
    print(f"    Epsilon (ε)= {EPSILON}")
    print(f"    Start      = {START}")
    print(f"    Goal       = {GOAL}")
    print(f"    Roadblocks = {ROADBLOCKS}")
    print(f"\n  Training MC Control...")

    Q = mc_control()

    print("\n  Training complete!")

    # Display Q-table
    display_q_table(Q)

    # Extract and display policy
    policy = extract_policy(Q)
    display_policy(policy)

    # Simulate agent
    print("  Simulation (greedy policy, stochastic environment):")
    path, total_reward = simulate(Q)
    path_str = " -> ".join([str(s) for s in path])
    print(f"    Path:   {path_str}")
    print(f"    Reward: {total_reward}")
    reached = path[-1] == GOAL
    print(f"    Reached goal: {reached}")

    # Run multiple simulations to show success rate
    print("\n  Success rate over 1000 simulations:")
    successes = 0
    for _ in range(1000):
        p, r = simulate(Q)
        if p[-1] == GOAL:
            successes += 1
    print(f"    {successes}/1000 ({successes / 10:.1f}%)")
    print("\n" + "=" * 60)