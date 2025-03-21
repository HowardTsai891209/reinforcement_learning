from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# 固定參數
GAMMA = 0.9  # 折扣因子
theta = 0.01  # 收斂條件
actions = {'↑': (-1, 0), '↓': (1, 0), '←': (0, -1), '→': (0, 1)}

def value_iteration(grid_size, start, end, obstacles):
    V = np.zeros((grid_size, grid_size))
    policy = np.full((grid_size, grid_size), '', dtype=object)
    
    while True:
        delta = 0
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in obstacles or (x, y) == end:
                    continue  # 障礙物與終點不更新
                
                v_old = V[x, y]
                best_value = float('-inf')
                best_actions = []
                
                for action, (dx, dy) in actions.items():
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in obstacles:
                        reward = 1 if (nx, ny) == end else 0  # 終點獲得獎勵
                        new_value = reward + GAMMA * V[nx, ny]
                        if new_value > best_value:
                            best_value = new_value
                            best_actions = [action]  # 更新最佳動作
                        elif new_value == best_value:
                            best_actions.append(action)  # 記錄多個最優動作
                
                V[x, y] = best_value if best_value != float('-inf') else 0
                policy[x, y] = ''.join(best_actions) if best_actions else ' '
                delta = max(delta, abs(v_old - V[x, y]))
        
        if delta < theta:
            break
    
    return V.tolist(), policy.tolist()

def traversal_path(grid_size, start, end, obstacles):
    visited = set()
    path = []
    
    def dfs(x, y):
        if (x, y) in visited or (x, y) in obstacles or x < 0 or x >= grid_size or y < 0 or y >= grid_size:
            return False
        visited.add((x, y))
        path.append((x, y))
        
        if (x, y) == end:
            return True
        
        for dx, dy in actions.values():
            if dfs(x + dx, y + dy):
                return True
        
        return False
    
    dfs(start[0], start[1])
    return path

@app.route('/', methods=['GET', 'POST'])
def index():
    n = 5
    if request.method == 'POST':
        n = int(request.form.get('n', 5))
        if n < 3 or n > 10:
            n = 5
    return render_template('index.html', n=n)

@app.route('/compute_policy', methods=['POST'])
def compute_policy():
    data = request.json
    grid_size = data['grid_size']
    start = tuple(data['start'])
    end = tuple(data['end'])
    obstacles = {tuple(obs) for obs in data['obstacles']}
    
    value_matrix, policy_matrix = value_iteration(grid_size, start, end, obstacles)
    return jsonify({'value_matrix': value_matrix, 'policy_matrix': policy_matrix})

@app.route('/compute_traversal', methods=['POST'])
def compute_traversal():
    data = request.json
    grid_size = data['grid_size']
    start = tuple(data['start'])
    end = tuple(data['end'])
    obstacles = {tuple(obs) for obs in data['obstacles']}
    
    traversal_path_list = traversal_path(grid_size, start, end, obstacles)
    return jsonify({'traversal_path': traversal_path_list})

if __name__ == '__main__':
    app.run(debug=True)
