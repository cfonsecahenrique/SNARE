import random as rand
import numpy as np
from ModelParameters import ModelParameters as MP
from agent import Agent


def print_ebnorm(sn):
    readable = [["", ""], ["", ""]]
    for i in range(len(sn)):
        for j in range(len(sn[i])):
            part = rep_char(sn[i][j][0]) + "," + rep_char(sn[i][j][1])
            readable[i][j] = part
    print("2nd Order Emotion Based Social norm:")
    # socialNorm[action][reputation]
    print("\t    G   B")
    print("\t----------")
    print("\tC|", readable[1][1], readable[1][0])
    print("\tD|", readable[0][1], readable[0][0])


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


def prisoners_dilemma(agent1: Agent, agent2: Agent, EBSN, SN, eps: float, chi: float, gamma: float):
    # pd = ( [ [D,D],[D,C] ],[ [C,D],[C,C] ] )
    T = 5
    R = 4
    P = 0
    S = -1
    pd = np.array([
        [(P, P), (T, S)],
        [(S, T), (R, R)]
    ])

    cooperative_acts = 0

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

    # print("------------------")
    # print("a2rep = " + str(agent2.get_reputation()))
    # print("a1_trait = " + str(a1.get_trait()[0]))
    # print("This means a1_action: " + str(a1_action))

    # Execution error for action of agent 1
    if rand.random() < eps:
        a1_action = invert_binary(a1_action)
        # print("Agent " + str(agent1.get_agent_id()) + " tried to cooperate but failed!")
        if rand.random() < gamma:
            # Look at Emotion Based Social Norm
            agent1.set_reputation(EBSN[a1_action][a1_rep][agent1.emotion_profile()])
        else:
            # Look at simple social norm
            agent1.set_reputation(SN[a1_action][a1_rep])
    else:
        # Look at simple social norm
        agent1.set_reputation(SN[a1_action][a1_rep])

    # Execution error for action of agent 2
    if rand.random() < eps:
        a2_action = invert_binary(a2_action)
        # print("Agent " + str(agent2.get_agent_id()) + " tried to cooperate but failed!")
        if rand.random() < gamma:
            agent2.set_reputation(EBSN[a2_action][a2_rep][agent2.emotion_profile()])
        else:
            # Look at simple social norm
            agent1.set_reputation(SN[a1_action][a1_rep])
    else:
        # Look at simple social norm
        agent1.set_reputation(SN[a1_action][a1_rep])

    # Count coop acts
    # coop = 1, def = 0
    cooperative_acts += a1_action
    cooperative_acts += a2_action

    # Reputation stuff
    # print("----------------------------")
    # print_norm(SN)
    # print("Agent A", agent1.get_agent_id(), "action was", action_char(a1_action), "while Agent", agent2.get_agent_id(), "reputation was", rep_char(a2_rep))
    # print("Agent B", agent2.get_agent_id(), "action was", action_char(a2_action), "while Agent", agent1.get_agent_id(), "reputation was", rep_char(a1_rep))
    # print("Additionally, A showed the", ep_char(agent1.get_trait()[1]), "emotion profile and B showed", ep_char(agent2.get_trait()[1]))

    # print("Agent A", agent1.get_agent_id(), "new rep =", rep_char(agent1.get_reputation()), "payoff:", pd[a1_action][a2_action][0])
    # print("Agent B", agent2.get_agent_id(), "new rep =", rep_char(agent2.get_reputation()), "payoff:", pd[a1_action][a2_action][1])
    return pd[a1_action][a2_action], cooperative_acts


def action_char(act: int):
    return "C" if act == 1 else "D"


def ep_char(ep: int):
    return "M" if ep == 0 else "N"


def rep_char(rep: int):
    return "G" if rep == 1 else "B"


def print_agent(ag: Agent):
    if ag.strategy()[0] == (0, 0):
        strat = "AllD"
    elif ag.strategy()[0] == (0, 1):
        strat = "pDisc"
    elif ag.strategy()[0] == (1, 0):
        strat = "Disc"
    else:
        strat = "AllC"
    print("Agent: " + str(ag.get_agent_id()) + ", Strat: " + strat + ", Rep: " + str(ag.get_reputation()))


def export_results(acr: float, mp: MP, population: list[Agent]):
    builder: str = mp.generate_mp_string() + "\t" + str(acr) + "\t"
    rep_freqs = reputation_frequencies(population)
    builder += str(rep_freqs[0]) + "\t" + str(rep_freqs[1]) + "\t"
    builder += make_strat_str(calculate_strategy_frequency(population))
    builder += make_strat_str(calculate_ep_frequencies(population))
    builder += "\n"
    f = open("outputs/results.txt", "a")
    f.write(builder)
    f.close()


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
