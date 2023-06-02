import numpy as np
import pandas as pd
import random as rand
from agent import Agent
from tqdm import tqdm


def get_random_agent_pair():
    return rand.sample(agents, 2)


def get_random_agent() -> Agent:
    return rand.choice(agents)


def invert_action(action: int):
    return 1 - action


def print_population():
    for ag in agents:
        print_agent(ag)


def print_agent(ag: Agent):
    print("Agent: " + str(ag.get_agent_id()) + ", Trait: " + str(ag.get_trait()) + ", Rep: " + str(ag.get_reputation()))


def prisoners_dilemma(agent1: Agent, agent2: Agent):
    global games_played
    global cooperative_acts
    # increment # of played games, 2 because it's 2DG=PD

    if current_gen > converge: games_played += 2


    # get_trait()[0]: action rule
    a1_action: int = agent1.get_trait()[0][agent2.get_reputation()]
    print("Agent's " + str(agent1.get_agent_id()) + " strat is " + str(agent1.get_trait()))
    print("His action is " + str(a1_action))
    a2_action: int = agent2.get_trait()[0][agent1.get_reputation()]
    print("Agent's " + str(agent2.get_agent_id()) + " strat is " + str(agent2.get_trait()))
    print("His action is " + str(a2_action))

    # print("------------------")
    # print("a2rep = " + str(agent2.get_reputation()))
    # print("a1_trait = " + str(a1.get_trait()[0]))
    # print("This means a1_action: " + str(a1_action))

    if a1_action == 1:
        if rand.random() < eps:
            a1_action = invert_action(a1_action)
            print("Agent " + str(agent1.get_agent_id()) + " (strat " + str(agent1.get_trait()) + ") tried to cooperate but failed!")
        else:
            if current_gen > converge: cooperative_acts += 1
    if a2_action == 1:
        if rand.random() < eps:
            a2_action = invert_action(a2_action)
            print("Agent " + str(agent2.get_agent_id()) + " (strat " + str(agent2.get_trait()) + ") tried to cooperate but failed!")
        else:
            if current_gen > converge: cooperative_acts += 1

    # Reputation stuff
    a1.set_reputation(socialNorm[a1_action][a2.get_reputation()][a1.get_trait()[1]])
    a2.set_reputation(socialNorm[a2_action][a1.get_reputation()][a2.get_trait()[1]])
    # print("A1 new rep = " + str(a1.get_reputation()))

    return PD[a1_action][a2_action]


Z = 10
G = 1000 * Z
mu = 0.1 / Z
# Eps will be an array for the experiments
eps = 0.1 / Z
b = 5
c = 1
converge = 200

# pd = [ [ [D,D],[D,C] ],[ [C,D],[C,C] ] ]
PD = [[(0, 0), (b, -c)], [(-c, b), (b, b)]]
print("Dilemma: " + str(pd))

print("Z", Z, ", G:", G, ", mu:", mu, ", eps:", eps)

current_gen = 0
games_played = 0
cooperative_acts = 0
number_mutations = 0

# social norm - new reputation : socialNorm[action][rec_reputation][emotion profile]
# [ [ [DBM,DBN],[DGM,DGN] ],[ [CBM,CBN],[CGM,CGN] ] ]
#     ....0.... ....1....     ....0.... ....1....
#   ...........0...........  ...........1..........
# ...................................................
# B = 0, G = 1
# D = 0, C = 1
# M = 0, N = 1

# Normal SJ
# socialNorm = [[(1, 1), (0, 0)], [(0, 0), (1, 1)]]
# All G
# socialNorm = [[(1, 1), (1, 1)], [(1, 1), (1, 1)]]
# All B
socialNorm = [[(0, 0), (0, 0)], [(0, 0), (0, 0)]]

# Initialization
agents = []
for i in range(Z):
    a = Agent(i)
    agents.append(a)

print_population()

for current_gen in tqdm(range(G)):
    if rand.random() < mu:
        # Trait Exploration
        a1 = get_random_agent()
        a1.trait_mutation()
        number_mutations += 1
        # print("Agent " + str(a1.get_agent_id()) + " randomly explored. New trait: " + str(a1.get_trait()))
    else:
        # Exploration through Social Learning
        a1, a2 = get_random_agent_pair()
        print("Selected agents " + str(a1.get_agent_id()) + " and " + str(a2.get_agent_id()))
        a1.set_fitness(0)
        a2.set_fitness(0)
        # Make a new list without a1 and a2
        aux_list = [a for a in agents if
                    a.get_agent_id() != a1.get_agent_id() and a.get_agent_id() != a2.get_agent_id()]

        # Each agent plays Z games
        for i in range(Z):
            az = rand.choice(aux_list)
            print("Agent " + str(a1.get_agent_id()) + " will play with Agent ", str(az.get_agent_id()))
            res1, resZ = prisoners_dilemma(a1, az)
            a1.add_fitness(res1)
            ax = rand.choice(aux_list)
            print("Agent " + str(a2.get_agent_id()) + " will play with Agent ", str(ax.get_agent_id()))
            res2, resX = prisoners_dilemma(a2, ax)
            a2.add_fitness(res2)

        a1.set_fitness(a1.get_fitness() / Z)
        a2.set_fitness(a2.get_fitness() / Z)

        if rand.random() < (1 + np.exp(a2.get_fitness() - a1.get_fitness())) ** (-1):
            a1.set_trait(a2.get_trait())

print("Final ACR:" + str(cooperative_acts / games_played))
print("#Cooperative acts: " + str(cooperative_acts) + ", #Played Games: " + str(games_played))
print("#Mutations: " + str(number_mutations))
print_population()
