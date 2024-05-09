import random as rand
from collections import defaultdict

import numpy as np
from ModelParameters import ModelParameters as MP
from agent import Agent


def print_sn(sn):
    print("2nd Order Social norm:")
    # socialNorm[action][reputation]
    print("\t   G B")
    print("\t----------")
    print("\tC|", rep_char(sn[1][1]), rep_char(sn[1][0]))
    print("\tD|", rep_char(sn[0][1]), rep_char(sn[0][0]))


def get_random_agent_pair(agents):
    return rand.sample(agents, 2)


def get_random_agent(agents) -> Agent:
    return rand.choice(agents)


def invert_binary(action: int):
    return 1 - action


def print_population(agents):
    for ag in agents:
        print_agent(ag)


def prisoners_dilemma(agent1: Agent, agent2: Agent, EBSN, SN, eps: float, chi: float, alpha: float, gamma: float):
    # Payoff matrix of the prisoner's dilemma (pd)
    # pd = ( [ [D,D],[D,C] ],[ [C,D],[C,C] ] )
    # (T)emptation; (R)eward; (P)unishment; (S)ucker's payoff
    T: int = 5
    R: int = 4
    P: int = 0
    S: int = -1
    pd = np.array([
        [(P, P), (T, S)],
        [(S, T), (R, R)]
    ])

    cooperative_acts = 0

    # Get agent's reputations locally so that the changed reputations aren't used in the wrong place
    a1_rep: int = agent1.get_reputation()
    a2_rep: int = agent2.get_reputation()
    # get_trait()[0]: action rule

    # Check for rep assessment error
    if rand.random() > chi:
        a1_action: int = agent1.strategy()[a2_rep]
    else:
        a1_action: int = agent1.strategy()[invert_binary(a2_rep)]

    if rand.random() > chi:
        a2_action: int = agent2.strategy()[a1_rep]
    else:
        a2_action: int = agent2.strategy()[invert_binary(a1_rep)]

    # Execution error for action of agent 1
    if rand.random() < eps:
        a1_action = invert_binary(a1_action)
        # print("Agent " + str(agent1.get_agent_id()) + " tried to cooperate but failed!")
    # Execution error for action of agent 2
    if rand.random() < eps:
        a2_action = invert_binary(a2_action)

    # AGENT 1 new rep --------------------------
    # Probability of using EB Norm
    if rand.random() < gamma:
        # Look at Emotion Based Social Norm
        # DONOR FOCAL
        #new_rep: int = EBSN[a1_action][a1_rep][agent1.emotion_profile()]
        # RECIPIENT FOCAL (common in IR)
        new_rep: int = EBSN[a1_action][a2_rep][agent1.emotion_profile()]
    else:
        # Look at simple social norm
        # DONOR FOCAL
        #new_rep: int = SN[a1_action][a1_rep]
        # RECIPIENT FOCAL (common in IR)
        new_rep: int = SN[a1_action][a2_rep]

    # Assignment error
    if rand.random() < alpha:
        # REPUTATION ASSIGNMENT ERROR ag1
        new_rep = invert_binary(new_rep)

    agent1.set_reputation(new_rep)

    # AGENT 2 new rep --------------------------
    if rand.random() < gamma:
        # Look at Emotion Based Social Norm
        # DONOR FOCAL
        #new_rep: int = EBSN[a2_action][a2_rep][agent2.emotion_profile()]
        # RECIPIENT FOCAL (common in IR)
        new_rep: int = EBSN[a2_action][a1_rep][agent2.emotion_profile()]
    else:
        # Look at simple social norm
        # DONOR FOCAL
        #new_rep: int = SN[a2_action][a2_rep]
        # RECIPIENT FOCAL (common in IR)
        new_rep: int = SN[a2_action][a1_rep]

    if rand.random() < alpha:
        # REPUTATION ASSIGNMENT ERROR ag2
        new_rep = invert_binary(new_rep)

    agent2.set_reputation(new_rep)

    # Count coop acts
    # coop = 1, def = 0
    cooperative_acts += a1_action
    cooperative_acts += a2_action

    return pd[a1_action][a2_action], cooperative_acts


def action_char(act: int):
    return "C" if act == 1 else "D"


def ep_char(ep: int):
    return "M" if ep == 0 else "N"


def rep_char(rep: int):
    return "G" if rep == 1 else "B"


def print_agent(ag: Agent):
    print("Agent: " + str(ag.get_agent_id()) + ", Strat: " + strat_name(ag.strategy()) + ", Rep: " + str(ag.get_reputation()))


def export_results(acr: float, mp: MP, population: list[Agent]):
    #most_popular_per_ep = most_common_strats(population)
    winner_et = most_common_evol_trait(population)
    builder: str = mp.generate_mp_string() + "\t" + str(round(acr,3)) + "\t"
    rep_freqs = reputation_frequencies(population)
    builder += str(rep_freqs[0]) + "\t" + str(rep_freqs[1]) + "\t"
    builder += make_strat_str(calculate_strategy_frequency(population))
    builder += make_strat_str(calculate_ep_frequencies(population))
    builder += str(winner_et)
    builder += "\n"
    f = open("outputs/results_with_ets.txt", "a")

    if " " not in builder:
        f.write(builder)
    else:
        print("Exportation error")
    f.close()

def most_common_evol_trait(population: list[Agent]):
    eps = [0, 1]
    strats = [(0,0), (0,1), (1,0), (1,1)]
    counter_dict: dict = {}
    for ep in eps:
        for strat in strats:
            counter_dict[(ep, strat)] = 0
    for agent in population:
        counter_dict[(agent.emotion_profile(), agent.strategy())] += 1
    #print(counter_dict)
    best = max(counter_dict, key=counter_dict.get)

    string: str = str(best[0]) + str(best[1][0]) + str(best[1][1])
    print(best)
    print(string)
    return string



def most_common_strats(population: list[Agent]):
    counter: dict = {}

    there_is_ep0 = False
    there_is_ep1 = False

    # initialize
    strat_counts_0: dict = {s: 0 for s in [(0,0), (0,1), (1,0), (1,1)]}
    strat_counts_1: dict = {s: 0 for s in [(0,0), (0,1), (1,0), (1,1)]}

    for agent in population:
        if agent.emotion_profile() == 0:
            there_is_ep0 = True
            strat_counts_0[agent.strategy()] += 1
        elif agent.emotion_profile() == 1:
            there_is_ep1 = True
            strat_counts_1[agent.strategy()] += 1

    print("COMPETITIVE")
    print(strat_counts_0)
    print("COOPERATIVE")
    print(strat_counts_1)

    if there_is_ep0:
        counter["Competitive"] = strat_name(max(strat_counts_0, key=strat_counts_0.get))
    else:
        counter["Competitive"] = None

    if there_is_ep1:
        counter["Cooperative"] = strat_name(max(strat_counts_1, key=strat_counts_1.get))
    else:
        counter["Cooperative"] = "NA"

    print(counter)
    return counter


def reputation_frequencies(population: list[Agent]):
    # Only works binarily, for now
    # 0: B, 1: G
    rep_freqs: dict = {0: 0, 1: 0}
    for agent in population:
        rep_freqs[agent.get_reputation()] += 1
    rep_freqs[0] /= len(population)
    rep_freqs[1] /= len(population)
    return [rep_freqs[0], rep_freqs[1]]


def calculate_strategy_frequency(population: list[Agent]):
    strategy_freqs: dict = {}
    strats = [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ]

    for strat in strats:
        strategy_freqs[strat] = 0

    for agent in population:
        strategy_freqs[agent.strategy()] += 1

    for strat in strats:
        strategy_freqs[strat] = strategy_freqs[strat] / len(population)

    return strategy_freqs


def calculate_ep_frequencies(population: list[Agent]):
    ep_freqs = {0: 0, 1: 0}

    for agent in population:
        ep_freqs[agent.emotion_profile()] += 1

    for ep in ep_freqs:
        ep_freqs[ep] = ep_freqs[ep] / len(population)

    return ep_freqs


def make_strat_str(frequencies: dict):
    builder: str = ""
    for key in frequencies:
        builder += str(frequencies.get(key)) + "\t"
    return builder


def make_ebsn_from_list(l: list):
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
    print("2nd Order Emotion Based Social norm: (m,n)")
    # socialNorm[action][reputation]
    print("\t    G   B")
    print("\t----------")
    print("\tC|", readable[1][1], readable[1][0])
    print("\tD|", readable[0][1], readable[0][0])


def strat_name(strat):
    d: dict = {(0, 0): "AllD", (0, 1): "pDisc", (1, 0): "Disc", (1, 1): "AllC"}
    return d[strat]
