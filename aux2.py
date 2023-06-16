import random as rand

import numpy as np

from agent import Agent


def print_norm(sn):
    readable = np.chararray((2,2))
    for i in range(len(sn)):
        for j in range(len(sn[i])):
            readable[i][j] = "G" if sn[i][j][0]==1 else "B"
    print("2nd Order Social norm:")
    # socialNorm[action][reputation]
    print("\t   G B")
    print("\t-------")
    print("\tC|", readable[1][1].decode(), readable[1][0].decode())
    print("\tD|", readable[0][1].decode(), readable[0][0].decode())

def get_random_agent_pair(agents):
    return rand.sample(agents, 2)


def get_random_agent(agents) -> Agent:
    return rand.choice(agents)


def invert_action(action: int):
    return 1 - action


def print_population(agents):
    for ag in agents:
        print_agent(ag)

def prisoners_dilemma(agent1: Agent, agent2: Agent, SN, eps: float):
    # pd = ( [ [D,D],[D,C] ],[ [C,D],[C,C] ] )
    T = 4
    R = 2
    P = 1
    S = 0
    pd = np.array([
        [P, T],
        [S, R]
    ])

    cooperative_acts = 0

    # get_trait()[0]: action rule
    a1_action: int = agent1.get_trait()[0][agent2.get_reputation()]
    #print("Agent's " + str(agent1.get_agent_id()) + " strat is " + str(agent1.get_trait()))
    #print("His action is " + str(a1_action))
    a2_action: int = agent2.get_trait()[0][agent1.get_reputation()]
    #print("Agent's " + str(agent2.get_agent_id()) + " strat is " + str(agent2.get_trait()))
    #print("His action is " + str(a2_action))

    # print("------------------")
    # print("a2rep = " + str(agent2.get_reputation()))
    # print("a1_trait = " + str(a1.get_trait()[0]))
    # print("This means a1_action: " + str(a1_action))

    if a1_action == 1:
        if rand.random() < eps:
            a1_action = invert_action(a1_action)
            #print("Agent " + str(agent1.get_agent_id()) + " (strat " + str(agent1.get_trait()) + ") tried to cooperate but failed!")
        else:
            cooperative_acts += 1
    if a2_action == 1:
        if rand.random() < eps:
            a2_action = invert_action(a2_action)
            #print("Agent " + str(agent2.get_agent_id()) + " (strat " + str(agent2.get_trait()) + ") tried to cooperate but failed!")
        else:
            cooperative_acts += 1

    # Reputation stuff
    print("----------------------------")
    print_norm(SN)
    action = "C" if a1_action==1 else "D"
    reputation = "G" if agent2.get_reputation()==1 else "B"
    print("Agent", agent1.get_agent_id(), "action was", action)
    print("Agent", agent2.get_agent_id(), "reputation was", reputation)
    agent1.set_reputation(SN[a1_action][agent2.get_reputation()][agent1.get_trait()[1]])
    agent2.set_reputation(SN[a2_action][agent2.get_reputation()][agent1.get_trait()[1]])
    newrep = "G" if agent1.get_reputation()==1 else "B"
    print("Agent", agent1.get_agent_id(), "new rep = ", newrep)

    return pd[a1_action][a2_action], cooperative_acts


def print_agent(ag: Agent):
    if ag.get_trait()[0] == (0,0):
        strat = "AllD"
    elif ag.get_trait()[0] == (0,1):
        strat = "pDisc"
    elif ag.get_trait()[0] == (1,0):
        strat = "Disc"
    else:
        strat = "AllC"
    print("Agent: " + str(ag.get_agent_id()) + ", Strat: " + strat + ", Rep: " + str(ag.get_reputation()))

