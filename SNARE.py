import sys
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import yaml
from agent import Agent
from tqdm import tqdm
import aux_functions as aux
from ModelParameters import ModelParameters as mp
import multiprocessing

# social norm - new reputation : socialNorm[action][rec_reputation][emotion profile]
# [ [ [DBM,DBN],[DGM,DGN] ],[ [CBM,CBN],[CGM,CGN] ] ]
# R   ....0.... ....1....     ....0.... ....1....
# A  ..........0...........  ...........1..........
# ----------------------------------------------------
# B = 0, G = 1
# D = 0, C = 1
# M = 0, N = 1
BAD, DEFECT, MEAN = 0, 0, 0
GOOD, COOPERATE, NICE = 1, 1, 1
# -------------

ALWAYS_COOPERATE = (1, 1)
DISCRIMINATE = (0, 1)
PARADOXICALLY_DISC = (1, 0)
ALWAYS_DEFECT = (0, 0)


def simulation(model: mp):
    # Population Size
    z = model.z
    # Strategy Exploration Probability
    mu: float = model.mu / model.z
    # Reputation Assessment Error Probability
    chi: float = model.chi / model.z
    # (cooperation) Execution Error
    eps: float = model.eps / model.z
    # Judge assignment error
    alpha: float = model.alpha / model.z
    # Number of Generations to run
    gens: int = model.gens
    # Social Norm
    social_norm = model.social_norm
    # EB Social norm
    eb_social_norm = model.ebsn
    # Converging period
    converge: int = model.converge
    # benefit to cost ratio
    benefit: int = model.benefit
    cost: int = model.cost
    bc_ratio: float = benefit/cost
    # selection strength beta
    selection_strength: float = model.selection_strength
    min_gamma: float = model.min_gamma
    max_gamma: float = model.max_gamma

    games_played = 0
    cooperative_acts = 0
    number_mutations = 0
    # Initialization
    agents: list[Agent] = []
    a1: Agent
    a2: Agent

    cooperation_per_gen: np.array = np.zeros(gens)
    allD, Disc, pDisc, allC = np.zeros(gens), np.zeros(gens), np.zeros(gens), np.zeros(gens)
    mean, nice = np.zeros(gens), np.zeros(gens)
    bad, good = np.zeros(gens), np.zeros(gens)
    average_gammas = np.zeros(gens)

    for i in range(z):
        agents.append(Agent(i, min_gamma, max_gamma))

    for current_gen in range(gens):
        past_convergence: bool = current_gen > converge
        # 1 Gen = Z evolutionary steps
        for _ in range(z):
            # 1 evolutionary time-step = 1 Mutation or 1 Social Learning Step
            if rand.random() < mu:
                # Trait Exploration
                a1 = rand.choice(agents)
                a1.trait_mutation(min_gamma, max_gamma)
                number_mutations += 1
                #print("Agent " + str(a1.get_agent_id()) + " randomly explored. New trait: " + str(a1.get_trait()))
            else:
                # 1 Social Learning = 2 players playing Z games each
                a1, a2 = aux.get_random_agent_pair(agents)
                # print("Selected agents " + str(a1.get_agent_id()) + " and " + str(a2.get_agent_id()))
                a1.set_fitness(0)
                a2.set_fitness(0)
                # Make a new list without a1 and a2
                excluded_ids = {a1.get_agent_id(), a2.get_agent_id()}
                aux_list = [a for a in agents if a.get_agent_id() not in excluded_ids]

                # Each agent plays Z games
                for i in range(z):
                    # (each 2-player prisoner's dilemma has 2 actions
                    # and can have at most 2 cooperative acts)

                    az: Agent = rand.choice(aux_list)
                    res_z, n_z = aux.prisoners_dilemma(a1, az, eb_social_norm, social_norm, eps, chi, alpha, benefit, cost)
                    #print("Agent", a1.get_agent_id(), "[" + aux.rep_char(a1.get_reputation()) + "], with strategy", aux.strat_name(a1.strategy()),
                    #      "played against", az.get_agent_id(), "(", aux.rep_char(az.get_reputation()), "), with strategy", aux.strat_name(az.strategy()))
                    #print("The result of the interaction was", res_z[0], "for agent", a1.get_agent_id(), "and", res_z[1], "for agent", az.get_agent_id())
                    if past_convergence:
                        cooperative_acts += n_z
                        games_played += 2
                    a1.add_fitness(res_z[0])

                    ax: Agent = rand.choice(aux_list)
                    res_x, n_x = aux.prisoners_dilemma(a2, ax, eb_social_norm, social_norm, eps, chi, alpha, benefit, cost)
                    if past_convergence:
                        cooperative_acts += n_x
                        games_played += 2
                    a2.add_fitness(res_x[0])

                # normalize both players' fitness
                a1.set_fitness(a1.get_fitness() / z)
                a2.set_fitness(a2.get_fitness() / z)

                # Calculate Probability of imitation
                pi: float = (1 + np.exp(selection_strength*(a1.get_fitness() - a2.get_fitness()))) ** (-1)
                if rand.random() < pi:
                    a1.set_strategy(a2.strategy())
                    a1.set_emotion_profile(a2.emotion_profile())
                    a1.set_gamma(a2.gamma())

        if past_convergence:
            cooperation_per_gen[current_gen] = cooperative_acts/games_played
            #print(cooperative_acts, games_played, cooperation_per_gen[current_gen])

        strategy_frequencies = aux.calculate_strategy_frequency(agents)
        allD[current_gen] = strategy_frequencies.get(ALWAYS_DEFECT)
        Disc[current_gen] = strategy_frequencies.get(DISCRIMINATE)
        pDisc[current_gen] = strategy_frequencies.get(PARADOXICALLY_DISC)
        allC[current_gen] = strategy_frequencies.get(ALWAYS_COOPERATE)

        emotion_profile_frequencies = aux.calculate_ep_frequencies(agents)
        mean[current_gen] = emotion_profile_frequencies.get(0)
        nice[current_gen] = emotion_profile_frequencies.get(1)

        reputation_frequencies = aux.calculate_reputation_frequencies(agents)
        bad[current_gen] = reputation_frequencies.get(BAD)
        good[current_gen] = reputation_frequencies.get(GOOD)

        average_gammas[current_gen] = aux.calculate_average_gamma(agents)

    aux.export_results(100 * cooperation_per_gen[gens-1], model, agents)
    return cooperation_per_gen, (allD, Disc, pDisc, allC), (mean, nice), (bad, good), average_gammas


def read_args():
    # Reads the arguments of the .yaml file
    # Open and parse the YAML file
    with open(str(sys.argv[1]), "r") as f:
        data = yaml.safe_load(f)

    n_runs: int = data["running"]["runs"]
    cores: int = data["running"]["cores"]

    simulation_parameters = data["simulation"]

    ebsn_list: list = simulation_parameters["ebsn"]
    eb_sn: list[list] = aux.make_ebsn_from_list(ebsn_list)

    sn_list: list = simulation_parameters["sn"]
    sn: list[list] = aux.make_sn_from_list(sn_list)

    z: int = int(simulation_parameters["z"])
    mu: float = float(simulation_parameters["mu"])
    chi: float = float(simulation_parameters["chi"])
    eps: float = float(simulation_parameters["eps"])
    alpha: float = float(simulation_parameters["alpha"])
    benefit: int = int(simulation_parameters["benefit"])
    cost: int = int(simulation_parameters["cost"])
    beta: float = float(simulation_parameters["beta"])
    generations: int = int(simulation_parameters["generations"])
    min_gamma: float = float(simulation_parameters["gamma_min"])
    max_gamma: float = float(simulation_parameters["gamma_max"])
    convergence: float = float(simulation_parameters["convergence period"])

    model_parameters: mp = mp(
        str(simulation_parameters["sn"]), sn, str(simulation_parameters["ebsn"]), eb_sn, z, mu, chi, eps, alpha,
        min_gamma, max_gamma, generations, benefit, cost, beta, convergence
    )

    print(model_parameters.__str__())
    return n_runs, cores, model_parameters


def plot_time_series(all_results: list):
    cooperation_results = [r[0] for r in all_results]  # list of arrays, each array shape=(gens,)
    strategy_results = [r[1] for r in all_results]     # list of arrays, each array shape=(4, gens)
    ep_results = [r[2] for r in all_results]           # list of arrays, each array shape=(2, gens)
    rep_results = [r[3] for r in all_results]
    gammas = [r[4] for r in all_results]

    gens = cooperation_results[0].shape[0]
    x = np.arange(gens)

    # === Cooperation plot (single time series) ===
    coop_matrix = np.stack(cooperation_results)  # shape: (runs, gens)
    coop_mean = coop_matrix.mean(axis=0)
    coop_std = coop_matrix.std(axis=0)

    # === Strategy plot (4 time series) ===
    # Stack into shape (runs, 4, gens)
    strat_matrix = np.stack(strategy_results)
    strat_mean = strat_matrix.mean(axis=0)  # shape (4, gens)
    strat_std = strat_matrix.std(axis=0)    # shape (4, gens)

    # === EP plot (2 time series) ===
    ep_matrix = np.stack(ep_results)
    ep_mean = ep_matrix.mean(axis=0)  # shape (2, gens)
    ep_std = ep_matrix.std(axis=0)    # shape (2, gens)

    # === Reputations plot (2 time series) ===
    rep_matrix = np.stack(rep_results)
    rep_mean = rep_matrix.mean(axis=0)  # shape (2, gens)
    rep_std = rep_matrix.std(axis=0)  # shape (2, gens)

    # Average Gammas
    gammas_matrix = np.stack(gammas)
    gammas_mean = gammas_matrix.mean(axis=0)
    gammas_std = gammas_matrix.std(axis=0)

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    # --- Plot cooperation ---
    axes[0].plot(x, coop_mean, color='blue', label='Mean Cooperation Rate')
    axes[0].fill_between(x, coop_mean - coop_std, coop_mean + coop_std,
                         color='blue', alpha=0.3, label='±1 Std Dev')
    axes[0].set_title("Average Cooperation Rate Across Simulations")
    axes[0].set_ylabel("Cooperation Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True)

    # --- Plot strategies ---
    strategy_labels = ["AllD", "Disc", "pDisc", "AllC"]
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green']
    for i in range(4):
        axes[1].plot(x, strat_mean[i], color=colors[i], label=f"Strategy {strategy_labels[i]}")
        axes[1].fill_between(x, strat_mean[i] - strat_std[i], strat_mean[i] + strat_std[i],
                             color=colors[i], alpha=0.3)
    axes[1].set_ylabel("Strategy Frequency")
    axes[1].set_title("Strategy Frequencies Across Simulations")
    axes[1].legend()
    axes[1].grid(True)

    # --- Plot emotion profiles ---
    ep_labels = ["mean", "nice"]
    ep_colors = ['tab:brown', 'tab:cyan']
    for i in range(2):
        axes[2].plot(x, ep_mean[i], color=ep_colors[i], label=ep_labels[i])
        axes[2].fill_between(x, ep_mean[i] - ep_std[i], ep_mean[i] + ep_std[i],
                             color=ep_colors[i], alpha=0.3)
    axes[2].set_ylabel("EP Frequency")
    axes[2].set_title("Emotion Profile Frequencies Across Simulations")
    axes[2].legend()
    axes[2].grid(True)

    # --- Plot reputation frequencies ---
    rep_labels = ["bad", "good"]
    rep_colors = ['tab:red', 'tab:cyan']
    for i in range(2):
        axes[3].plot(x, rep_mean[i], color=rep_colors[i], label=rep_labels[i])
        axes[3].fill_between(x, rep_mean[i] - rep_std[i], rep_mean[i] + rep_std[i],
                             color=rep_colors[i], alpha=0.3)
    axes[3].set_ylabel("Reputation Frequency")
    axes[3].set_title("Reputation Frequencies Across Simulations")
    axes[3].legend()
    axes[3].grid(True)

    # --- Plot cooperation ---
    axes[4].plot(x, gammas_mean, color='blue', label='Average Gamma')
    axes[4].fill_between(x, gammas_mean - gammas_std, gammas_mean + gammas_std,
                         color='blue', alpha=0.3, label='±1 Std Dev')
    axes[4].set_title("Average Gammas Across Simulations")
    axes[4].set_ylabel("Gamma Frequency")
    axes[4].set_xlabel("Generation")
    axes[4].set_ylim(0, 1)
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    if len(sys.argv) != 2:
        raise ValueError("No experiment configuration passed!")

    num_runs, num_cores, model = read_args()

    aux.print_sn(model.social_norm)
    aux.print_ebnorm(model.ebsn)

    print("Running", num_runs, "independent parallel simulations over", num_cores, "cpu core(s).")

    all_models: list[mp] = [model] * num_runs
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_results = list(
            tqdm(pool.imap_unordered(simulation, all_models), total=num_runs)
        )

    plot_time_series(all_results)

