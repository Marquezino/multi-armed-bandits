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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $\epsilon$-decaying greedy
    """)
    return


@app.cell
def _(EpsilonGreedyBandit, np):
    class EpsilonDecayingGreedyBandit(EpsilonGreedyBandit):
        def epsilon_t(self, t):
            return 1.0 / np.sqrt(max(t, 1))

        def select_arm(self, t=None):
            t = 1 if t is None else t
            epsilon_t = self.epsilon_t(t)
            if self.rng.random() < epsilon_t:
                return self.rng.integers(self.num_arms)
            return np.argmax(self.values)



    return (EpsilonDecayingGreedyBandit,)


@app.cell
def _(EpsilonDecayingGreedyBandit, np, num_arms, seed):
    rng_decay = np.random.default_rng(seed.value + 1)
    bandit_decay = EpsilonDecayingGreedyBandit(num_arms=num_arms.value, rng=rng_decay)
    return bandit_decay, rng_decay


@app.cell
def _():

    distances_decay = []
    epsilons_decay = []

    return distances_decay, epsilons_decay


@app.cell
def _(
    bandit_decay,
    distances_decay,
    epsilons_decay,
    np,
    rng_decay,
    steps,
    true_probabilities,
):
    for t in range(1, steps.value + 1):
        arm_d = bandit_decay.select_arm(t)
        reward_d = 1 if rng_decay.random() < true_probabilities[arm_d] else 0
        bandit_decay.update(arm_d, reward_d)

        eps_t = bandit_decay.epsilon_t(t)
        epsilons_decay.append(eps_t)
        distances_decay.append(np.linalg.norm(bandit_decay.values - true_probabilities))
    return


@app.cell
def _(bandit_decay, np, true_probabilities):

    print("Probabilidades reais (decaying).....: ", np.round(true_probabilities, 3))
    print("Estimativas finais (decaying greedy): ", np.round(bandit_decay.values, 3))

    return


@app.cell
def _(distances_decay, epsilons_decay, plt):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(range(1, len(distances_decay) + 1), distances_decay)
    ax1.set_title("Distância L2 com ε-decaying greedy")
    ax1.set_ylabel("Distância L2")
    ax1.grid(alpha=0.25)

    ax2.plot(range(1, len(epsilons_decay) + 1), epsilons_decay)
    ax2.set_title("Evolução de ε_t = 1/sqrt(t)")
    ax2.set_xlabel("Passo")
    ax2.set_ylabel("ε_t")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    EpsilonDecayingGreedyBandit,
    EpsilonGreedyBandit,
    epsilon,
    np,
    num_arms,
    plt,
    seed,
    steps,
):
    # Recalcula as duas estrategias nesta celula para evitar graficos vazios por ordem de execucao
    rng_cmp = np.random.default_rng(seed.value)
    true_probs_cmp = rng_cmp.uniform(0.05, 0.95, size=num_arms.value)

    # Epsilon fixo
    bandit_fixed = EpsilonGreedyBandit(
        num_arms=num_arms.value,
        epsilon=epsilon.value,
        rng=np.random.default_rng(seed.value + 100),
    )
    dist_fixed = []
    for _ in range(steps.value):
        arm_f = bandit_fixed.select_arm()
        reward_f = 1 if rng_cmp.random() < true_probs_cmp[arm_f] else 0
        bandit_fixed.update(arm_f, reward_f)
        dist_fixed.append(np.linalg.norm(bandit_fixed.values - true_probs_cmp))

    # Epsilon decaying
    bandit_dec = EpsilonDecayingGreedyBandit(
        num_arms=num_arms.value,
        rng=np.random.default_rng(seed.value + 200),
    )
    dist_dec = []
    for t in range(1, steps.value + 1):
        arm_d = bandit_dec.select_arm(t)
        reward_d = 1 if rng_cmp.random() < true_probs_cmp[arm_d] else 0
        bandit_dec.update(arm_d, reward_d)
        dist_dec.append(np.linalg.norm(bandit_dec.values - true_probs_cmp))

    plt.figure(figsize=(9, 4.5))
    x_fixed = range(1, len(dist_fixed) + 1)
    x_dec = range(1, len(dist_dec) + 1)
    plt.plot(x_fixed, dist_fixed, label=f"epsilon fixo = {epsilon.value:.2f}", alpha=0.9)
    plt.plot(x_dec, dist_dec, label="epsilon-decaying: 1/sqrt(t)", alpha=0.9)
    plt.title("Comparacao direta: distancia L2 por passo")
    plt.xlabel("Passo")
    plt.ylabel("Distancia L2")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
