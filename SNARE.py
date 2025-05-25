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

    for i in range(z):
        agents.append(Agent(i, min_gamma, max_gamma))

    for current_gen in range(gens):
        past_convergence: bool = current_gen > converge
        # 1 Gen = Z evolutionary steps
        for _ in range(z):
            # 1 evolutionary time-step = 1 Mutation or 1 Social Learning Step
            if rand.random() < mu:
                # Trait Exploration
                a1 = aux.get_random_agent(agents)
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
                # increment # of played games, 4 because it's each PD has 2 donation games
                if past_convergence: games_played += 4
                for i in range(z):
                    # (each 2-player prisoner's dilemma has 4 actions
                    # and can have at most 2 cooperative acts)

                    az = rand.choice(aux_list)
                    res_z, n_z = aux.prisoners_dilemma(a1, az, eb_social_norm, social_norm, eps, chi, alpha, benefit, cost)
                    if past_convergence:
                        cooperative_acts += n_z
                    a1.add_fitness(res_z[0])

                    ax = rand.choice(aux_list)
                    res_x, n_x = aux.prisoners_dilemma(a2, ax, eb_social_norm, social_norm, eps, chi, alpha, benefit, cost)
                    if past_convergence:
                        cooperative_acts += n_x
                    a2.add_fitness(res_x[0])

                # normalize both players' fitness
                a1.set_fitness(a1.get_fitness() / z)
                a2.set_fitness(a2.get_fitness() / z)

                #if current_gen > 0.5*gens: print("a1 fitness:", a1.get_fitness(), "a2 fitness:", a2.get_fitness())
                # Calculate Probability of imitation
                pi: float = (1 + np.exp(selection_strength*(a1.get_fitness() - a2.get_fitness()))) ** (-1)
                #if current_gen > 0.5*gens: print("Prob of imitation:", pi)
                if rand.random() < pi:
                    a1.set_strategy(a2.strategy())
                    a1.set_emotion_profile(a2.emotion_profile())
                    a1.set_gamma(a2.gamma())
        if past_convergence:
            cooperation_per_gen[current_gen] = cooperative_acts/games_played

        strategy_frequencies = aux.calculate_strategy_frequency(agents)
        allD[current_gen] = strategy_frequencies.get((0, 0))
        Disc[current_gen] = strategy_frequencies.get((0, 1))
        pDisc[current_gen] = strategy_frequencies.get((1, 0))
        allC[current_gen] = strategy_frequencies.get((1, 1))

        emotion_profile_frequencies = aux.calculate_ep_frequencies(agents)
        mean[current_gen] = emotion_profile_frequencies.get(0)
        nice[current_gen] = emotion_profile_frequencies.get(1)

    aux.export_results(100 * cooperation_per_gen[gens-1], model, agents)
    return cooperation_per_gen, (allD, Disc, pDisc, allC), (mean, nice)


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


def plot_time_series(all_results):
    cooperation_results = all_results[0]  # list of arrays, each array shape=(gens,)
    strategy_results = all_results[1]     # list of arrays, each array shape=(4, gens)
    ep_results = all_results[2]           # list of arrays, each array shape=(2, gens)

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

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # --- Plot cooperation ---
    axes[0].plot(x, coop_mean, color='blue', label='Mean Cooperation Rate')
    axes[0].fill_between(x, coop_mean - coop_std, coop_mean + coop_std,
                         color='blue', alpha=0.3, label='±1 Std Dev')
    axes[0].set_ylabel("Cooperation Rate")
    axes[0].set_title("Average Cooperation Rate Across Simulations")
    axes[0].legend()
    axes[0].grid(True)

    # --- Plot strategies ---
    strategy_labels = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i in range(4):
        axes[1].plot(x, strat_mean[i], color=colors[i], label=f"Strategy {strategy_labels[i]}")
        axes[1].fill_between(x, strat_mean[i] - strat_std[i], strat_mean[i] + strat_std[i],
                             color=colors[i], alpha=0.3)
    axes[1].set_ylabel("Strategy Frequency")
    axes[1].set_title("Strategy Frequencies Across Simulations")
    axes[1].legend()
    axes[1].grid(True)

    # --- Plot emotion profiles ---
    ep_labels = ["EP 0", "EP 1"]
    ep_colors = ['tab:cyan', 'tab:brown']
    for i in range(2):
        axes[2].plot(x, ep_mean[i], color=ep_colors[i], label=ep_labels[i])
        axes[2].fill_between(x, ep_mean[i] - ep_std[i], ep_mean[i] + ep_std[i],
                             color=ep_colors[i], alpha=0.3)
    axes[2].set_ylabel("EP Frequency")
    axes[2].set_xlabel("Generation")
    axes[2].set_title("Emotion Profile Frequencies Across Simulations")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    if len(sys.argv) == 1:
        raise ValueError("No experiment configuration passed!")
    else:
        print("No .yaml input file given!")

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

