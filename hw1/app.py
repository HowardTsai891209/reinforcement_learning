from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Hyperparameters
gamma = 0.9  # Discount factor
theta = 0.0001  # Convergence threshold

# Directions for policy (up, down, left, right)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def value_iteration(n, start, end, obstacles):
    V = np.zeros((n, n))  # Value function
    policy = np.zeros((n, n), dtype=int)  # Policy array
    delta = float('inf')

    while delta >= theta:
        delta = 0
        for i in range(n):
            for j in range(n):
                if (i, j) == start or (i, j) == end or (i, j) in obstacles:
                    continue  # Skip start, end, and obstacles
                v = V[i, j]
                max_value = -float('inf')
                best_action = -1

                for a_idx, (di, dj) in enumerate(actions):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n and (ni, nj) not in obstacles:
                        value = -1 + gamma * V[ni, nj]
                        if value > max_value:
                            max_value = value
                            best_action = a_idx

                V[i, j] = max_value
                policy[i, j] = best_action
                delta = max(delta, abs(v - V[i, j]))

    return V, policy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    data = request.json
    n = data['n']
    start = tuple(data['start'])
    end = tuple(data['end'])
    obstacles = [tuple(obs) for obs in data['obstacles']]

    # Run value iteration to get V(s) and policy
    V, policy = value_iteration(n, start, end, obstacles)
    
    return jsonify({
        'V': V.tolist(),
        'policy': policy.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
