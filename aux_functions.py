import random as rand
from collections import Counter
import numpy as np
from model import Model
from agent import Agent


def write_to_file(filename: str, data: str):
    with open(filename, "a") as f:
        f.write(data + "\n")


def get_random_agent_pair(agents):
    return rand.sample(agents, 2)


def invert_binary(action: int):
    return 1 - action


def action_char(act: int):
    return "C" if act == 1 else "D"


def ep_char(ep: int):
    return "N" if ep == 1 else "M"


def rep_char(rep: int):
    return "G" if rep == 1 else "B"


def export_results(acr: float, model: Model, population: list[Agent]):
    # most_popular_per_ep = most_common_strats(population)
    winner_et = most_common_evol_trait(population)
    builder: str = model.generate_mp_string() + "\t" + str(round(acr, 3)) \
                   + "\t" + str(calculate_average_consensus(model.image_matrix))
    builder += make_strat_str(calculate_strategy_frequency(population))
    builder += make_strat_str(calculate_ep_frequencies(population))
    builder += str(winner_et)
    write_to_file("outputs/results.txt", builder)


def most_common_evol_trait(population: list[Agent]):
    eps = [0, 1]
    strats = [(0, 0), (0, 1), (1, 0), (1, 1)]
    counter_dict: dict = {}
    for ep in eps:
        for strat in strats:
            counter_dict[(ep, strat)] = 0
    for agent in population:
        counter_dict[(agent.emotion_profile(), agent.strategy())] += 1
    # print(counter_dict)
    best = max(counter_dict, key=counter_dict.get)

    string: str = str(best[0]) + str(best[1][0]) + str(best[1][1])
    return string


def most_common_strats(population: list[Agent]):
    counter: dict = {}

    there_is_ep0 = False
    there_is_ep1 = False

    # initialize
    strat_counts_0: dict = {s: 0 for s in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    strat_counts_1: dict = {s: 0 for s in [(0, 0), (0, 1), (1, 0), (1, 1)]}

    for agent in population:
        if agent.emotion_profile() == 0:
            there_is_ep0 = True
            strat_counts_0[agent.strategy()] += 1
        elif agent.emotion_profile() == 1:
            there_is_ep1 = True
            strat_counts_1[agent.strategy()] += 1

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
    counts = Counter(agent.strategy() for agent in population)
    # Ensure all 4 strategies are present in the output
    strats = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return {strat: counts.get(strat, 0) / total for strat in strats}


def calculate_ep_frequencies(population: list[Agent]) -> dict:
    total = len(population)
    counts = Counter(agent.emotion_profile() for agent in population)
    return {ep: counts.get(ep, 0) / total for ep in (0, 1)}


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
    # (bdm, bdn, bcm, bcn, gdm, gdn, gcm, gcn)
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
    # prints the chosen social norm in a readable fashion
    readable = [["", ""], ["", ""]]
    for i in range(len(sn)):
        for j in range(len(sn[i])):
            part = rep_char(sn[i][j][0]) + "," + rep_char(sn[i][j][1])
            readable[i][j] = part
    print("2nd Order Emotion Based Social norm:")
    # socialNorm[action][reputation]
    print("\tGOOD C nice: [", readable[1][1][2], "]")
    print("\tGOOD C mean: [", readable[1][1][0], "]")
    print("\tGOOD D nice: [", readable[1][0][2], "]")
    print("\tGOOD D mean: [", readable[1][0][0], "]")
    print("\tBAD  C nice: [", readable[0][1][2], "]")
    print("\tBAD  C mean: [", readable[0][1][0], "]")
    print("\tBAD  D nice: [", readable[0][0][2], "]")
    print("\tBAD  D mean: [", readable[0][0][0], "]")


def print_sn(sn):
    print("2nd Order Social norm:")
    # socialNorm[action][reputation]
    print("\t   G B")
    print("\t----------")
    print("\tC|", rep_char(sn[1][1]), rep_char(sn[1][0]))
    print("\tD|", rep_char(sn[0][1]), rep_char(sn[0][0]))


def strat_name(strat):
    d: dict = {(0, 0): "AllD", (0, 1): "Disc", (1, 0): "pDisc", (1, 1): "AllC"}
    return d[strat]


def action(a: int):
    return "C" if a else "D"


def reputation(r: int):
    return "G" if r else "B"


def emotion(e: int):
    return "Coop" if e else "Comp"
