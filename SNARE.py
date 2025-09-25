import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
from agent import Agent
from tqdm import tqdm
import aux_functions as aux
from model import Model
import multiprocessing
from time import time
from copy import deepcopy
from constants import *
import pandas as pd
from datetime import timedelta


def run_simulations_for_model(model, n_runs, n_cores):
    all_models = [deepcopy(model) for _ in range(n_runs)]
    with multiprocessing.Pool(processes=n_cores) as pool:
        all_results = list(tqdm(pool.imap_unordered(simulation, all_models), total=n_runs))
    return all_results


def simulation(model: Model):
    z = model.population_size
    mu = model.mutation_rate / model.population_size
    gens = model.generations
    selection_strength = model.selection_strength

    games_played = 0
    cooperative_acts = 0

    agents = [
        Agent(i, model.min_gamma, model.max_gamma,
              model.gamma_normal_center, model.gamma_delta)
        for i in range(z)
    ]

    cooperation_per_gen = np.zeros(gens)
    allD, Disc, pDisc, allC = np.zeros(gens), np.zeros(gens), np.zeros(gens), np.zeros(gens)
    mean, nice = np.zeros(gens), np.zeros(gens)
    bad, good = np.zeros(gens), np.zeros(gens)
    avg_gammas = np.zeros(gens)

    # Preallocate RNG for efficiency
    # Estimate an upper bound on the number of random numbers needed per generation
    # (mutation + PD plays + strategy imitation + errors)
    max_randoms_per_gen = z * (1 + z * 16) * 50 # rough conservative estimate, oversized by 50
    rng = np.random.default_rng()

    for current_gen in tqdm(range(gens)):
        past_convergence = current_gen > model.converge
        random_values = rng.random(size=max_randoms_per_gen)
        ri = 0
        for i in range(z):
            if random_values[ri] < mu:
                rnd_agent_idx = rng.integers(len(agents))
                a1 = agents[rnd_agent_idx]
                a1.trait_mutation(model.min_gamma, model.max_gamma, model.gamma_delta)
            else:
                rnd_agent_idxs = rng.integers(len(agents), size=2)
                a1: Agent = agents[rnd_agent_idxs[0]]
                a2: Agent = agents[rnd_agent_idxs[1]]

                a1.set_fitness(0)
                a2.set_fitness(0)

                excluded_ids = {a1.get_agent_id(), a2.get_agent_id()}
                n_agents = len(agents)

                # Pre-generate opponent indices for this round (z pairs of opponents)
                opponent_idxs = rng.integers(n_agents, size=(z, 2))

                for j in range(z):
                    # resample if opponent index is invalid
                    while opponent_idxs[j, 0] in excluded_ids:
                        opponent_idxs[j, 0] = rng.integers(n_agents)
                    while opponent_idxs[j, 1] in excluded_ids:
                        opponent_idxs[j, 1] = rng.integers(n_agents)

                    az = agents[opponent_idxs[j, 0]]
                    res_z, n_z, ri = model.prisoners_dilemma(a1, az, random_values, ri)
                    if past_convergence:
                        cooperative_acts += n_z
                        games_played += 2
                    a1.add_fitness(res_z[0])

                    ax = agents[opponent_idxs[j, 1]]
                    res_x, n_x, ri = model.prisoners_dilemma(a2, ax, random_values, ri)
                    if past_convergence:
                        cooperative_acts += n_x
                        games_played += 2
                    a2.add_fitness(res_x[0])

                # Normalize fitness
                a1.set_fitness(a1.get_fitness() / z)
                a2.set_fitness(a2.get_fitness() / z)

                # Strategy imitation
                pi = (1 + np.exp(selection_strength * (a1.get_fitness() - a2.get_fitness()))) ** (-1)
                if random_values[ri] < pi:
                    a1.set_strategy(a2.strategy)
                    a1.set_emotion_profile(a2.emotion_profile)
                    if model.gamma_delta != 0:
                        a1.set_gamma(a2.gamma())
                ri += 1
            ri += 1

        if past_convergence:
            cooperation_per_gen[current_gen] = cooperative_acts/games_played if games_played > 0 else 0

        strat_freq = aux.calculate_strategy_frequency(agents)
        allD[current_gen] = strat_freq.get(Strategy.ALWAYS_DEFECT, 0)
        Disc[current_gen] = strat_freq.get(Strategy.DISCRIMINATE, 0)
        pDisc[current_gen] = strat_freq.get(Strategy.PARADOXICALLY_DISC, 0)
        allC[current_gen] = strat_freq.get(Strategy.ALWAYS_COOPERATE, 0)

        ep_freq = aux.calculate_ep_frequencies(agents)
        mean[current_gen] = ep_freq.get(0, 0)
        nice[current_gen] = ep_freq.get(1, 0)

        rep_freq = aux.calculate_reputation_frequencies(agents)
        bad[current_gen] = rep_freq.get(BAD, 0)
        good[current_gen] = rep_freq.get(GOOD, 0)

        avg_gammas[current_gen] = aux.calculate_average_gamma(agents)

    aux.export_results(100 * cooperation_per_gen[gens-1], model, agents)
    return cooperation_per_gen, (allD, Disc, pDisc, allC), (mean, nice), (bad, good), avg_gammas


def read_yaml(filename):
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    return data


def make_model_from_params(simulation_parameters):
    ebsn_list = simulation_parameters["ebsn"]
    eb_sn = aux.make_ebsn_from_list(ebsn_list)

    sn_list = simulation_parameters["sn"]
    sn = aux.make_sn_from_list(sn_list)

    z = int(simulation_parameters["z"])
    mu = float(simulation_parameters["mu"])
    chi = float(simulation_parameters["chi"])
    eps = float(simulation_parameters["eps"])
    alpha = float(simulation_parameters["alpha"])
    benefit = int(simulation_parameters["benefit"])
    cost = int(simulation_parameters["cost"])
    beta = float(simulation_parameters["beta"])
    generations = int(simulation_parameters["generations"])
    min_gamma = float(simulation_parameters["gamma_min"])
    max_gamma = float(simulation_parameters["gamma_max"])
    gamma_delta = float(simulation_parameters["gamma_delta"])
    gamma_gaussian_center = float(simulation_parameters["gamma_gaussian_n"])
    convergence = float(simulation_parameters.get("convergence period", 0))

    model_parameters = Model(
        sn, eb_sn, z, mu, chi, eps, alpha,
        min_gamma, max_gamma, gamma_delta, gamma_gaussian_center, generations, benefit, cost, beta, convergence
    )
    return model_parameters


def plot_time_series(all_results, model):
    cooperation_results = [r[0] for r in all_results]
    strategy_results = [r[1] for r in all_results]
    ep_results = [r[2] for r in all_results]
    rep_results = [r[3] for r in all_results]
    gammas = [r[4] for r in all_results]

    gens = cooperation_results[0].shape[0]
    x = np.arange(gens)

    coop_matrix = np.stack(cooperation_results)
    coop_mean = coop_matrix.mean(axis=0)
    coop_std = coop_matrix.std(axis=0)

    strat_matrix = np.stack(strategy_results)
    strat_mean = strat_matrix.mean(axis=0)
    strat_std = strat_matrix.std(axis=0)

    ep_matrix = np.stack(ep_results)
    ep_mean = ep_matrix.mean(axis=0)
    ep_std = ep_matrix.std(axis=0)

    rep_matrix = np.stack(rep_results)
    rep_mean = rep_matrix.mean(axis=0)
    rep_std = rep_matrix.std(axis=0)

    gammas_matrix = np.stack(gammas)
    gammas_mean = gammas_matrix.mean(axis=0)
    gammas_std = gammas_matrix.std(axis=0)

    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    plt.suptitle(model.ebsn_str)

    axes[0].plot(x, coop_mean, color='blue', label='Mean Cooperation Rate')
    axes[0].fill_between(x, coop_mean - coop_std, coop_mean + coop_std,
                         color='blue', alpha=0.3, label='±1 Std Dev')
    axes[0].set_title("Average Cooperation Rate Across Simulations")
    axes[0].set_ylabel("Cooperation Rate")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(True)

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

    axes[4].plot(x, gammas_mean, color='blue', label='Average Gamma')
    axes[4].fill_between(x, gammas_mean - gammas_std, gammas_mean + gammas_std,
                         color='blue', alpha=0.3, label='±1 Std Dev')
    axes[4].set_title("Average Gammas Across Simulations")
    axes[4].set_ylabel("Gamma Frequency")
    axes[4].set_xlabel("Generation")
    axes[4].legend()
    axes[4].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def expand_parameter(param):
    """
    Expands parameter if it is a list or a range string "start-end-step" or single value.
    Returns a list of values.
    """
    if isinstance(param, list):
        return param
    if isinstance(param, str) and '-' in param:
        parts = param.split('-')
        if len(parts) == 3:
            start, end, step = map(float, parts)
            return list(np.arange(start, end + step, step))
    return [param]


def generate_parameter_sets(base_simulation_params):
    # For now, only sweep gamma_gaussian_n
    gamma_param = base_simulation_params.get('gamma_gaussian_n', 0)
    gamma_values = expand_parameter(gamma_param)

    param_sets = []
    for gamma in gamma_values:
        params_copy = deepcopy(base_simulation_params)
        params_copy['gamma_gaussian_n'] = gamma
        param_sets.append(params_copy)
    return param_sets, gamma_values


def run_experiments(n_runs, n_cores, base_model_params):
    param_sets, param_values = generate_parameter_sets(base_model_params)

    avg_cooperations = []

    for i, sim_params in enumerate(param_sets):
        print(f"Running simulations for gamma_gaussian_n = {param_values[i]}")

        model = make_model_from_params(sim_params)
        all_results = run_simulations_for_model(model, n_runs, n_cores)

        cooperation_runs = [r[0][-1] for r in all_results]  # final cooperation for each run
        avg_coop = np.mean(cooperation_runs)
        avg_cooperations.append(avg_coop)

    return param_values, avg_cooperations


def plot_parameter_sweep(param_values, avg_cooperations, ebsn, param_name='gamma_gaussian_n'):
    title = aux.ebsn_to_GB(ebsn)
    plt.figure(figsize=(8, 5))
    plt.plot(param_values, avg_cooperations, marker='o')
    plt.title(f'Average Cooperation Rate vs {param_name}\nEB Social Norm: {title}')
    plt.xlabel(param_name)
    plt.ylabel('Average Cooperation Rate')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def run_single_value_experiment(n_runs, n_cores, base_sim_params, plots=True):
    print(f"Running single gamma experiment for gamma_gaussian_n = {base_sim_params['gamma_gaussian_n']}")

    model = make_model_from_params(base_sim_params)
    print(model.__str__())
    all_results = run_simulations_for_model(model, n_runs, n_cores)
    if plots:
        plot_time_series(all_results, model)


def run_sweep_experiment(n_runs, n_cores, base_sim_params, plots=True):
    param_sets, param_values = generate_parameter_sets(base_sim_params)

    avg_cooperations = []

    for i, sim_params in enumerate(param_sets):
        print(f"Running simulations for gamma_gaussian_n = {param_values[i]}")

        model = make_model_from_params(sim_params)
        print(model.__str__())
        all_results = run_simulations_for_model(model, n_runs, n_cores)

        cooperation_runs = [r[0][-1] for r in all_results]  # final cooperation for each run
        avg_coop = np.mean(cooperation_runs)
        avg_cooperations.append(avg_coop)

    if plots:
        plot_parameter_sweep(param_values, avg_cooperations, model.ebsn)


def load_norm_variants(csv_path, norm_name):
    df = pd.read_csv(csv_path, dtype={
        "4bit_orig": str,
        "8bit_vector": str
    })

    variants = df[df["norm"] == norm_name]

    if variants.empty:
        raise ValueError(f"No variants found for social norm '{norm_name}' in {csv_path}")

    return variants


def parse_vector_string(vec_str):
    """Turn a string like '00000101' into list of ints."""
    return [int(b) for b in vec_str]


def run_all_variants(norm_name, yaml_file, csv_file, n_runs, n_cores, plots=True):
    # 1. Load base params from yaml
    params = read_yaml(yaml_file)
    base_sim_params = params["simulation"]

    # 2. Load all variants for the given norm
    variants = load_norm_variants(csv_file, norm_name)

    # 3. Loop over variants
    for _, row in variants.iterrows():
        variant_id = row["variant_id"]
        sn = parse_vector_string(row["4bit_orig"])
        ebsn = parse_vector_string(row["8bit_vector"])

        # 4. Update sim params
        sim_params = deepcopy(base_sim_params)
        sim_params["sn"] = sn
        sim_params["ebsn"] = ebsn

        print(f"\n--- Running {variant_id} ---")
        if aux.is_single_value(sim_params["gamma_gaussian_n"]):
            run_single_value_experiment(n_runs, n_cores, sim_params, plots=plots)
        else:
            run_sweep_experiment(n_runs, n_cores, sim_params, plots=plots)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Usage: python SNARE.py <experiment.yaml>")

    config_file = sys.argv[1]

    # Hard-coded path to the master CSV with all norms
    NORM_CSV_PATH = "data/new_norms.csv"

    # Load config
    data = read_yaml(config_file)

    n_runs = data["running"]["runs"]
    n_cores = multiprocessing.cpu_count() - 1 if data["running"]["cores"] == "all" else data["running"]["cores"]
    with_logging = data["running"]["plotting"]

    # Base sim params (used as template)
    base_sim_params = data["simulation"]

    # Detect mode
    norm_name = base_sim_params.get("norm", "")

    start_time = time()

    if norm_name:  # --- Meta-norm mode ---
        print(f"Running all variants of meta-norm: {norm_name}")
        run_all_variants(
            norm_name=norm_name,
            yaml_file=config_file,
            csv_file=NORM_CSV_PATH,
            n_runs=n_runs,
            n_cores=n_cores,
            plots=with_logging
        )
    else:  # --- Manual mode (use ebsn + sn from YAML) ---
        gamma_param = base_sim_params.get('gamma_gaussian_n', 0)
        if aux.is_single_value(gamma_param):
            run_single_value_experiment(n_runs, n_cores, base_sim_params, plots=with_logging)
        else:
            run_sweep_experiment(n_runs, n_cores, base_sim_params, plots=with_logging)

    elapsed = time() - start_time
    print(f"Finished all experiments in {str(timedelta(seconds=int(elapsed)))} (hh:mm:ss)")

