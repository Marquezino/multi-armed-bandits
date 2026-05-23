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
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np


    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $\epsilon$-greedy bandits
    """)
    return


@app.cell
def _(np):
    class EpsilonGreedyBandit:
        def __init__(self, num_arms, epsilon=0.1, rng=None):
            self.num_arms = num_arms
            self.epsilon = epsilon
            self.rng = rng or np.random.default_rng()
            self.counts = np.zeros(num_arms)  # Vezes que cada braço foi puxado
            self.values = np.zeros(num_arms)  # Recompensa média estimada

        def select_arm(self, t=None):
            # Exploracao: escolhe um braco aleatorio
            if self.rng.random() < self.epsilon:
                return self.rng.integers(self.num_arms)
            # Explotacao: escolhe o melhor braco estimado
            return np.argmax(self.values)

        def update(self, chosen_arm, reward):
            self.counts[chosen_arm] += 1
            n = self.counts[chosen_arm]
            value = self.values[chosen_arm]
            # Atualizacao incremental da media: Q_new = Q_old + (1/n) * (Reward - Q_old)
            self.values[chosen_arm] = value + (1.0 / n) * (reward - value)


    return (EpsilonGreedyBandit,)


@app.cell
def _(mo):
    num_arms = mo.ui.slider(2, 20, value=3, label="Numero de arms")
    epsilon = mo.ui.slider(0.0, 1.0, step=0.05, value=0.1, label="Epsilon")
    seed = mo.ui.slider(0, 999, value=42, label="Seed")
    steps = mo.ui.slider(100, 10000, step=100, value=500, label="Steps")

    mo.vstack([
        mo.md("### Parametros da simulacao"),
        num_arms,
        epsilon,
        seed,
        steps
    ])
    return epsilon, num_arms, seed, steps


@app.cell
def _(EpsilonGreedyBandit, epsilon, np, num_arms, seed, steps):
    rng = np.random.default_rng(seed.value)
    true_probabilities = rng.uniform(0.05, 0.95, size=num_arms.value)
    bandit = EpsilonGreedyBandit(
        num_arms=num_arms.value,
        epsilon=epsilon.value,
        rng=rng,
        )

    distances = []

    for step in range(steps.value):
        arm = bandit.select_arm()
        # Simula a recompensa (1 para ganho, 0 para perda) com base na probabilidade real
        reward = 1 if rng.random() < true_probabilities[arm] else 0
        bandit.update(arm, reward)

        # Distancia L2 entre a estimativa atual e as probabilidades reais
        distances.append(np.linalg.norm(bandit.values - true_probabilities))

    return bandit, distances, true_probabilities


@app.cell
def _(bandit, np, true_probabilities):
    print("Probabilidades reais............: ", np.round(true_probabilities, 3))
    print("Estimativas finais dos bracos...: ", np.round(bandit.values, 3))
    return


@app.cell
def _(distances, plt):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(distances) + 1), distances)
    plt.title("Distância entre bandit.values e true_probabilities por passo")
    plt.xlabel("Passo")
    plt.ylabel("Distância L2")
    plt.grid(alpha=0.25)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
