from collections import Counter
import numpy as np
from model import Model
from agent import Agent
import os
import csv
from colorama import Fore, Style, init
from constants import *
from numba import njit
# Initialize colorama (needed for Windows)
init()


def write_to_file(filename: str, data: str):
    with open(filename, "a") as f:
        f.write(data + "\n")


@njit
def invert_binary(action: int):
    return 1 - action


def action_char(act: int):
    return "C" if act == 1 else "D"


def ep_char(ep: int):
    return "N" if ep == 1 else "M"


def rep_char(rep: int):
    return "G" if rep == 1 else "B"


def export_results(acr: float, model: Model, population: list[Agent], filename="outputs/results.csv"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    result_data: dict = {
        "base_social_norm": model.social_norm_str,
        "eb_social_norm": model.ebsn_str,
        "Z": model.population_size,
        "gens": model.generations,
        "mu": model.mutation_rate,
        "chi": model._chi,
        "eps": model.execution_error,
        "alpha": model._alpha,
        "b": model.benefit,
        "c": model.cost,
        "beta": model.beta,
        "generations": model.generations,
        "convergence_period": model.converge,
        "gamma_min": model.min_gamma,
        "gamma_max": model.max_gamma,
        "gamma_delta": model.gamma_delta,
        "gamma_center": model.gamma_normal_center,
        "average_cooperation": round(acr, 3)
    }

    # Add strategy frequencies
    frequencies = calculate_strategy_frequency(population)
    for strat in Strategy:
        result_data[strat.name] = round(frequencies.get(strat, 0), 4)

    ep_frequencies = calculate_ep_frequencies(population)
    for ep in ["Competitive", "Cooperative"]:
        result_data[ep] = round(ep_frequencies.get(ep, 0), 2)

    # Write to CSV
    write_dict_to_csv(result_data, filename)


def write_dict_to_csv(row: dict, filepath: str):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def most_common_evol_trait(population: list[Agent]):
    eps = [0, 1]
    strats = list(Strategy)
    counter_dict: dict = {}
    for ep in eps:
        for strat in strats:
            counter_dict[(ep, strat)] = 0
    for agent in population:
        counter_dict[(agent.emotion_profile, agent.strategy)] += 1
    # print(counter_dict)
    best = max(counter_dict, key=counter_dict.get)

    return f"{best[0]}{best[1].value[0]}{best[1].value[1]}"


def most_common_strats(population: list[Agent]):
    counter: dict = {}

    there_is_ep0 = False
    there_is_ep1 = False

    # initialize
    strat_counts_0: dict = {s: 0 for s in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    strat_counts_1: dict = {s: 0 for s in [(0, 0), (0, 1), (1, 0), (1, 1)]}

    for agent in population:
        if agent.emotion_profile == EmotionProfile.COMPETITIVE:
            there_is_ep0 = True
            strat_counts_0[agent.strategy] += 1
        elif agent.emotion_profile == EmotionProfile.COOPERATIVE:
            there_is_ep1 = True
            strat_counts_1[agent.strategy] += 1

    if there_is_ep0:
        counter["Competitive"] = strat_name(max(strat_counts_0, key=strat_counts_0.get))
    else:
        counter["Competitive"] = None

    if there_is_ep1:
        counter["Cooperative"] = strat_name(max(strat_counts_1, key=strat_counts_1.get))
    else:
        counter["Cooperative"] = None

    return counter


def calculate_strategy_frequency(population: list[Agent]) -> dict:
    total = len(population)
    counts = Counter(agent.strategy for agent in population)
    # Ensure all 4 strategies are present in the output
    strats = list(Strategy)
    return {strat: counts.get(strat, 0) / total for strat in strats}


def calculate_ep_frequencies(population: list[Agent]) -> dict:
    total = len(population)
    counts = Counter(agent.emotion_profile for agent in population)
    eps = list(EmotionProfile)
    return {ep: counts.get(ep, 0) / total for ep in eps}


@njit
def calculate_average_consensus(image_matrix: np.ndarray) -> float:
    z = image_matrix.shape[0]
    # Count number of 1s in each column (opinions about each focal agent)
    total_good = np.sum(image_matrix == 1, axis=0)
    total_bad = z - total_good  # Since only 0 or 1, and it's a square matrix
    consensus_per_agent = np.abs(total_good - total_bad) / z
    return float(np.mean(consensus_per_agent))


def calculate_average_gamma(population: list[Agent]) -> float:
    return sum(a.gamma() for a in population) / len(population)


def make_strat_str(frequencies: dict):
    builder: str = ""
    for key in frequencies:
        builder += str(frequencies.get(key)) + "\t"
    return builder


def make_ebsn_from_list(l: list):
    # ((bdm, bdn), (bcm, bcn), (gdm, gdn), (gcm, gcn))
    sn = []
    entry = []
    for i in np.arange(0, len(l) - 1, 2):
        entry.append((l[i], l[i + 1]))
        if len(entry) == 2:
            sn.append(entry)
            entry = []
    return sn


def make_sn_from_list(l: list):
    sn = [[l[0], l[1]], [l[2], l[3]]]
    return sn


def print_ebnorm(sn):
    """
    Prints the social norm as a colored table.
    Rows: Cooperate, Defect
    Columns: Good, Bad
    Cell entries: (nice, mean)
    Each 'G' is green, each 'B' is red.
    """
    def rep_char(val):
        return "G" if val == 1 else "B"

    def color_char(ch):
        if ch == "G":
            return Fore.GREEN + ch + Style.RESET_ALL
        elif ch == "B":
            return Fore.RED + ch + Style.RESET_ALL
        else:
            return ch

    # Build readable table values
    readable = [["", ""], ["", ""]]
    for i in range(len(sn)):
        for j in range(len(sn[i])):
            nice = rep_char(sn[i][j][1])
            mean = rep_char(sn[i][j][0])
            readable[i][j] = f"({color_char(nice)},{color_char(mean)})"

    # Header
    print(Fore.CYAN + "2nd Order Emotion-Based Social Norm:" + Style.RESET_ALL)
    header = f"{'':<12}{Fore.YELLOW}Good{Style.RESET_ALL:<8}{Fore.YELLOW}Bad{Style.RESET_ALL}"
    print(header)
    print("-" * 28)

    # Rows: i=1 -> Cooperate, i=0 -> Defect
    row_labels = {1: "Cooperate", 0: "Defect"}
    for i in [1, 0]:
        row_str = f"{row_labels[i]:<12}"
        for j in [1, 0]:  # Good, Bad
            row_str += f"{readable[i][j]:<8}"
        print(row_str)


def print_sn(sn):
    def colored_rep_char(c):
        # Example coloring based on char (adjust as needed)
        if c == 1:  # Good
            return Fore.GREEN + "G" + Style.RESET_ALL
        elif c == 0:  # Bad
            return Fore.RED + "B" + Style.RESET_ALL
        else:
            return c  # default no color

    print(Fore.CYAN + "2nd Order Social norm:" + Style.RESET_ALL)
    print(Fore.YELLOW + "   G B" + Style.RESET_ALL)
    print(Fore.YELLOW + "----------" + Style.RESET_ALL)
    print(f"{Fore.GREEN}C{Style.RESET_ALL}|", colored_rep_char(sn[1][1]), colored_rep_char(sn[1][0]))
    print(f"{Fore.RED}D{Style.RESET_ALL}|", colored_rep_char(sn[0][1]), colored_rep_char(sn[0][0]))


def strat_name(strat):
    d: dict = {(0, 0): "AllD", (0, 1): "Disc", (1, 0): "pDisc", (1, 1): "AllC"}
    return d[strat]


def action(a: int):
    return "C" if a else "D"


def reputation(r: int):
    return "G" if r else "B"


def emotion(e: int):
    return "Coop" if e else "Comp"


def calculate_reputation_frequencies(population: list[Agent]) -> dict:
    total = len(population)
    counts = Counter(agent.reputation() for agent in population)
    return {rep: counts.get(rep, 0) / total for rep in (0, 1)}


@njit
def consume_random(r, idx):
    val = r[idx]
    idx += 1
    return val, idx