import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Algumas ideias básicas sobre multi-armed bandits
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt


    return np, plt


@app.cell
def _(np):

    class EpsilonGreedyBandit:
        def __init__(self, num_arms, epsilon=0.1):
            self.num_arms = num_arms
            self.epsilon = epsilon
            self.counts = np.zeros(num_arms)  # Vezes que cada braço foi puxado
            self.values = np.zeros(num_arms)  # Recompensa média estimada
        
        def select_arm(self):
            # Exploração: escolhe um braço aleatório
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.num_arms)
            # Explotação: escolhe o melhor braço estimado
            else:
                return np.argmax(self.values)
            
        def update(self, chosen_arm, reward):
            self.counts[chosen_arm] += 1
            n = self.counts[chosen_arm]
            value = self.values[chosen_arm]
            # Atualização incremental da média: Q_new = Q_old + (1/n) * (Reward - Q_old)
            self.values[chosen_arm] = value + (1.0 / n) * (reward - value)


    return (EpsilonGreedyBandit,)


@app.cell
def _(EpsilonGreedyBandit, np):
    # Exemplo de uso com 3 braços (probabilidades reais de ganho ocultas)
    true_probabilities = np.array([0.2, 0.8, 0.5])  # O braço 1 é o melhor!
    bandit = EpsilonGreedyBandit(num_arms=3, epsilon=0.1)

    distances = []

    for step in range(100):
        arm = bandit.select_arm()
        # Simula a recompensa (1 para ganho, 0 para perda) com base na probabilidade real
        reward = 1 if np.random.rand() < true_probabilities[arm] else 0
        bandit.update(arm, reward)

        # Distância L2 entre a estimativa atual e as probabilidades reais
        distances.append(np.linalg.norm(bandit.values - true_probabilities))

    return bandit, distances


@app.cell
def _(bandit):

    print("Estimativas finais dos braços:", bandit.values)
    return


@app.cell
def _(distances, plt):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(distances) + 1), distances, color="tab:blue")
    plt.title("Distância entre bandit.values e true_probabilities por passo")
    plt.xlabel("Passo")
    plt.ylabel("Distância L2")
    plt.grid(alpha=0.3)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
