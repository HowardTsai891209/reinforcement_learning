from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# 預設 Gridworld 大小
n = 5
gamma = 0.9  # 折扣因子
theta = 0.0001  # 收斂條件
rewards = np.zeros((n, n))
policy = np.full((n, n), ' ')
value_matrix = np.zeros((n, n))
actions = {'↑': (-1, 0), '↓': (1, 0), '←': (0, -1), '→': (0, 1)}
start_point = None
end_point = None

@app.route('/')
def index():
    return render_template('index.html', n=n)

@app.route('/set_grid_size', methods=['POST'])
def set_grid_size():
    global n, rewards, value_matrix, policy
    data = request.json
    n = int(data.get('size', 5))
    rewards = np.zeros((n, n))
    value_matrix = np.zeros((n, n))
    policy = np.full((n, n), ' ')
    return jsonify({'status': 'success', 'n': n})

@app.route('/set_cell', methods=['POST'])
def set_cell():
    global start_point, end_point
    data = request.json
    x, y, cell_type = int(data['x']), int(data['y']), data['type']

    if cell_type == 'start':
        start_point = (x, y)
    elif cell_type == 'end':
        end_point = (x, y)
        rewards[x, y] = 1

    return jsonify({'status': 'success'})

@app.route('/start_iteration', methods=['POST'])
def start_iteration():
    global value_matrix, policy

    if start_point is None or end_point is None:
        return jsonify({'error': 'Please set start and end points first!'}), 400

    value_iteration()
    return jsonify({'value_matrix': value_matrix.tolist(), 'policy_matrix': policy.tolist()})

def value_iteration():
    global value_matrix, policy
    while True:
        delta = 0
        new_value_matrix = np.copy(value_matrix)

        for i in range(n):
            for j in range(n):
                if (i, j) == end_point:
                    continue

                max_value = float('-inf')
                best_action = ' '

                for action, (dx, dy) in actions.items():
                    new_i, new_j = i + dx, j + dy
                    if 0 <= new_i < n and 0 <= new_j < n:
                        value = rewards[i, j] + gamma * value_matrix[new_i, new_j]
                        if value > max_value:
                            max_value = value
                            best_action = action

                new_value_matrix[i, j] = max_value
                policy[i, j] = best_action
                delta = max(delta, abs(value_matrix[i, j] - max_value))

        value_matrix = new_value_matrix
        if delta < theta:
            break

if __name__ == '__main__':
    app.run(debug=True)
