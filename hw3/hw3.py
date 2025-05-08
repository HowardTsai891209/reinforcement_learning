import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定隨機種子以利重現結果

# 模擬 MAB 環境：每支拉霸機有不同的成功機率
class MultiArmedBandit:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.n_arms = len(probabilities)

    def pull(self, arm):
        return 1 if np.random.rand() < self.probabilities[arm] else 0


# ========= 1. Epsilon-Greedy =========
def epsilon_greedy(bandit, episodes=1000, epsilon=0.1):
    counts = np.zeros(bandit.n_arms)
    values = np.zeros(bandit.n_arms)
    rewards = []

    for t in range(episodes):
        if np.random.rand() < epsilon:
            arm = np.random.choice(bandit.n_arms)
        else:
            arm = np.argmax(values)

        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)

    return np.cumsum(rewards)


# ========= 2. Upper Confidence Bound (UCB) =========
def ucb(bandit, episodes=1000):
    counts = np.zeros(bandit.n_arms)
    values = np.zeros(bandit.n_arms)
    rewards = []

    for t in range(1, episodes + 1):
        if 0 in counts:
            arm = np.argmin(counts)
        else:
            confidence_bounds = values + np.sqrt(2 * np.log(t) / counts)
            arm = np.argmax(confidence_bounds)

        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)

    return np.cumsum(rewards)


# ========= 3. Softmax =========
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_strategy(bandit, episodes=1000, tau=0.1):
    counts = np.zeros(bandit.n_arms)
    values = np.zeros(bandit.n_arms)
    rewards = []

    for _ in range(episodes):
        probs = softmax(values / tau)
        arm = np.random.choice(bandit.n_arms, p=probs)
        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        rewards.append(reward)

    return np.cumsum(rewards)


# ========= 4. Thompson Sampling =========
def thompson_sampling(bandit, episodes=1000):
    alpha = np.ones(bandit.n_arms)
    beta = np.ones(bandit.n_arms)
    rewards = []

    for _ in range(episodes):
        samples = np.random.beta(alpha, beta)
        arm = np.argmax(samples)
        reward = bandit.pull(arm)

        if reward == 1:
            alpha[arm] += 1
        else:
            beta[arm] += 1

        rewards.append(reward)

    return np.cumsum(rewards)


# ========= 主程式 =========
if __name__ == "__main__":
    true_probabilities = [0.1, 0.5, 0.75, 0.3, 0.65]  # 各拉霸機的成功機率
    bandit = MultiArmedBandit(true_probabilities)
    episodes = 1000

    rewards_eps = epsilon_greedy(bandit, episodes, epsilon=0.1)
    bandit = MultiArmedBandit(true_probabilities)
    rewards_ucb = ucb(bandit, episodes)
    bandit = MultiArmedBandit(true_probabilities)
    rewards_softmax = softmax_strategy(bandit, episodes, tau=0.1)
    bandit = MultiArmedBandit(true_probabilities)
    rewards_thompson = thompson_sampling(bandit, episodes)

    # 畫圖
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_eps, label='Epsilon-Greedy')
    plt.plot(rewards_ucb, label='UCB')
    plt.plot(rewards_softmax, label='Softmax')
    plt.plot(rewards_thompson, label='Thompson Sampling')
    plt.title('Cumulative Reward of MAB Algorithms')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 第二張圖：平均獎勵（每一回合的 reward 平均）
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_eps / np.arange(1, episodes + 1), label='Epsilon-Greedy')
    plt.plot(rewards_ucb / np.arange(1, episodes + 1), label='UCB')
    plt.plot(rewards_softmax / np.arange(1, episodes + 1), label='Softmax')
    plt.plot(rewards_thompson / np.arange(1, episodes + 1), label='Thompson Sampling')
    plt.title('Average Reward Over Time (Convergence Speed)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 補充：計算每種策略的 arm 選擇次數
def get_arm_counts(strategy, bandit, episodes):
    counts = np.zeros(bandit.n_arms)
    values = np.zeros(bandit.n_arms)
    alpha = np.ones(bandit.n_arms)
    beta = np.ones(bandit.n_arms)

    for t in range(1, episodes + 1):
        if strategy == 'epsilon':
            epsilon = 0.1
            if np.random.rand() < epsilon:
                arm = np.random.randint(bandit.n_arms)
            else:
                arm = np.argmax(values)

            reward = bandit.pull(arm)
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]

        elif strategy == 'ucb':
            if 0 in counts:
                arm = np.argmin(counts)
            else:
                confidence = np.sqrt(2 * np.log(t) / counts)
                arm = np.argmax(values + confidence)

            reward = bandit.pull(arm)
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]

        elif strategy == 'softmax':
            tau = 0.1
            probs = softmax(values / tau)
            arm = np.random.choice(bandit.n_arms, p=probs)
            reward = bandit.pull(arm)
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]

        elif strategy == 'thompson':
            samples = np.random.beta(alpha, beta)
            arm = np.argmax(samples)
            reward = bandit.pull(arm)
            counts[arm] += 1
            alpha[arm] += reward
            beta[arm] += (1 - reward)

    return counts

# 模擬每種策略的 arm 選擇情形
strategies = {
    'Epsilon-Greedy': 'epsilon',
    'UCB': 'ucb',
    'Softmax': 'softmax',
    'Thompson Sampling': 'thompson'
}

counts_per_strategy = {}
for name, key in strategies.items():
    bandit = MultiArmedBandit(true_probabilities)
    counts_per_strategy[name] = get_arm_counts(key, bandit, episodes)

# 繪製柱狀圖
plt.figure(figsize=(12, 6))
bar_width = 0.2
x = np.arange(len(true_probabilities))

for i, (name, counts) in enumerate(counts_per_strategy.items()):
    plt.bar(x + i * bar_width, counts, width=bar_width, label=name)

plt.xticks(x + bar_width * 1.5, [f'Arm {i}' for i in range(len(true_probabilities))])
plt.title('Number of Times Each Arm Was Selected by Each Strategy')
plt.xlabel('Arm')
plt.ylabel('Number of Selections')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
