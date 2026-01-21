from collections import Counter
import numpy as np
from model import Model
import os
import csv
from colorama import Fore, Style, init
from constants import *
from itertools import chain

# Initialize colorama (needed for Windows)
init()


def write_to_file(filename: str, data: str):
    with open(filename, "a") as f:
        f.write(data + "\n")


def invert_binary(action: int):
    return 1 - action


def action_char(act: int):
    return "C" if act == 1 else "D"


def ep_char(ep: int):
    return "N" if ep == 1 else "M"


def rep_char(rep: int):
    return "G" if rep == 1 else "B"


def export_results(acr: float, model: Model, filename="outputs/observability_results.csv"):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    result_data: dict = {
        "base_social_norm": model.social_norm,
        "eb_social_norm": model.ebsn,
        "Z": model.population_size,
        "gens": model.generations,
        "mu": model.mutation_rate,
        "chi": model._chi,
        "eps": model.execution_error,
        "alpha": model._alpha,
        "q": model.observability,
        "b": model.benefit,
        "c": model.cost,
        "beta": model.beta,
        "generations": model.generations,
        "convergence_period": model.converge,
        "gamma_min": model.min_gamma,
        "gamma_max": model.max_gamma,
        "gamma_delta": model.gamma_delta,
        "gamma_center": model.gamma_normal_center,
        "average_cooperation": round(acr, 3),
        "average_consensus": round(calculate_average_consensus(model.image_matrix), 3)
    }

    # Add strategy frequencies
    frequencies = calculate_strategy_frequency(model)
    for strat in Strategy:
        result_data[strat.name] = round(frequencies.get(strat, 0), 4)

    ep_frequencies = calculate_ep_frequencies(model)
    for ep in list(EmotionProfile):
        result_data[ep.name] = round(ep_frequencies.get(ep, 0), 2)

    # Add reputation frequencies
    rep_frequencies = calculate_reputation_frequencies(model.image_matrix)
    result_data["Good_Rep_Freq"] = round(rep_frequencies.get(GOOD), 4)
    result_data["Bad_Rep_Freq"] = round(rep_frequencies.get(BAD), 4)

    # Write to CSV
    write_dict_to_csv(result_data, filename)


def calculate_ep_frequencies(model: Model) -> dict:
    total = model.population_size
    counts = Counter(model.emotion_profiles)
    eps = list(EmotionProfile)
    return {ep: counts.get(ep, 0) / total for ep in eps}


def calculate_average_consensus(image_matrix: np.ndarray) -> float:
    z = image_matrix.shape[0]
    # Count number of 1s in each column (opinions about each focal agent)
    total_good = np.sum(image_matrix == 1, axis=0)
    total_bad = np.sum(image_matrix == 0, axis=0)  # Since only 0 or 1, and it's a square matrix
    consensus_per_agent = np.abs(total_good - total_bad) / z
    return float(np.mean(consensus_per_agent))


def write_dict_to_csv(row: dict, filepath: str):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def calculate_strategy_frequency(model: Model) -> dict:
    total = model.population_size
    counts = Counter(model.strategies)
    strats = list(Strategy)
    return {strat: counts.get(strat, 0) / total for strat in strats}


def calculate_average_gamma(model: Model) -> float:
    return np.mean(model.gammas)


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


def format_ebnorm(sn) -> str:
    def rep_char(val):
        return "G" if val == 1 else "B"

    def color_char(ch):
        if ch == "G":
            return Fore.GREEN + ch + Style.RESET_ALL
        elif ch == "B":
            return Fore.RED + ch + Style.RESET_ALL
        else:
            return ch

    readable = [["", ""], ["", ""]]
    for i in range(len(sn)):
        for j in range(len(sn[i])):
            nice = rep_char(sn[i][j][1])
            mean = rep_char(sn[i][j][0])
            readable[i][j] = f"({color_char(nice)},{color_char(mean)})"

    lines = [
        Fore.CYAN + "2nd Order Emotion-Based Social Norm:" + Style.RESET_ALL,
        f"{'':<12}{Fore.YELLOW}Good{Style.RESET_ALL:<8}{Fore.YELLOW}Bad{Style.RESET_ALL}",
        "-" * 28
    ]

    row_labels = {1: "Cooperate", 0: "Defect"}
    for i in [1, 0]:
        row_str = f"{row_labels[i]:<12}"
        for j in [1, 0]:
            row_str += f"{readable[i][j]:<8}"
        lines.append(row_str)
    return "\n".join(lines)


def format_sn(sn) -> str:
    def colored_rep_char(c):
        if c == 1:
            return Fore.GREEN + "G" + Style.RESET_ALL
        elif c == 0:
            return Fore.RED + "B" + Style.RESET_ALL
        else:
            return c

    lines = [
        Fore.CYAN + "2nd Order Social norm:" + Style.RESET_ALL,
        Fore.YELLOW + "   G B" + Style.RESET_ALL,
        Fore.YELLOW + "----------" + Style.RESET_ALL,
        f"{Fore.GREEN}C{Style.RESET_ALL}| {colored_rep_char(sn[1][1])} {colored_rep_char(sn[1][0])}",
        f"{Fore.RED}D{Style.RESET_ALL}| {colored_rep_char(sn[0][1])} {colored_rep_char(sn[0][0])}"
    ]
    return "\n".join(lines)


def strat_name(strat):
    d: dict = {(0, 0): "AllD", (0, 1): "Disc", (1, 0): "pDisc", (1, 1): "AllC"}
    return d[strat]


def action(a: int):
    return "C" if a else "D"


def reputation(r: int):
    return "G" if r else "B"


def emotion(e: int):
    return "Coop" if e else "Comp"


def calculate_reputation_frequencies(image_matrix: np.ndarray) -> dict:
    """
    Calculates the frequency of GOOD and BAD reputations from the image matrix.
    The overall reputation of an agent is the mode of the column in the image matrix.
    This implementation uses vectorized NumPy operations for performance.
    """
    z = image_matrix.shape[0]

    # Count the number of GOOD (1) opinions for each agent (column-wise sum)
    g_index = np.sum(image_matrix)/(z*z)
    b_index = 1 - g_index

    return {
        GOOD: g_index,
        BAD: b_index
    }


def ebsn_to_GB(ebsn):
    # Flatten two levels: outer list and tuple
    seq = list(chain.from_iterable(chain.from_iterable(ebsn)))
    return ''.join('G' if int(b) == 1 else 'B' for b in seq)


def is_single_value(param):
    return not isinstance(param, (list, tuple))
