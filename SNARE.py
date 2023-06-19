import numpy as np
import pandas as pd
import random as rand
from agent import Agent
from tqdm import tqdm
import aux_functions as aux

converge = 200

# social norm - new reputation : socialNorm[action][rec_reputation][emotion profile]
# [ [ [DBM,DBN],[DGM,DGN] ],[ [CBM,CBN],[CGM,CGN] ] ]
# R   ....0.... ....1....     ....0.... ....1....
# A  ..........0...........  ...........1..........

# B = 0, G = 1
# D = 0, C = 1
# M = 0, N = 1


def main(social_norm, runs: int = 50, z: int = 50, g: int = 5000, mu: float = 0.1, chi: float = 0.1, eps: float = 0.1):

    mu = mu / z
    chi = chi / z
    eps = eps / z
    g = g * z

    all_games = np.zeros(runs)

    for r in range(runs):
        all_games[r] = simulation(z, g, mu, eps, chi, r, social_norm)

    print([round(a, 3) for a in all_games])
    print("Final Average of ", runs, " runs: ", all_games.mean())

    export_results(all_games)


def simulation(Z: int, gens: int, mu: float, eps: float, chi: float, i: int, sn):

    print("Run", i, "Z", Z, ", Gens:", gens, ", mu:", mu, ", eps:", eps, ", chi:", chi)
    games_played = 0
    cooperative_acts = 0
    number_mutations = 0
    aux.print_norm(sn)
    # Initialization
    agents = []
    for i in range(Z):
        a = Agent(i)
        agents.append(a)

    #aux.print_population(agents)

    for current_gen in tqdm(range(gens)):
        if rand.random() < mu:
            # Trait Exploration
            a1 = aux.get_random_agent(agents)
            a1.trait_mutation()
            number_mutations += 1
            # print("Agent " + str(a1.get_agent_id()) + " randomly explored. New trait: " + str(a1.get_trait()))
        else:
            # Exploration through Social Learning
            a1, a2 = aux.get_random_agent_pair(agents)
            # print("Selected agents " + str(a1.get_agent_id()) + " and " + str(a2.get_agent_id()))
            a1.set_fitness(0)
            a2.set_fitness(0)
            # Make a new list without a1 and a2
            aux_list = [a for a in agents if
                        a.get_agent_id() != a1.get_agent_id() and a.get_agent_id() != a2.get_agent_id()]

            # Each agent plays Z games
            for i in range(Z):
                # increment # of played games, 4 because it's 2*2DG=2PD
                if current_gen > converge: games_played += 4
                #print("----------------- GAME", i, "------------------------")
                az = rand.choice(aux_list)
                # print("Agent " + str(a1.get_agent_id()) + " will play with Agent ", str(az.get_agent_id()))
                res, n = aux.prisoners_dilemma(a1, az, sn, eps, chi)
                res1 = res[0]
                if current_gen > converge: cooperative_acts += n
                # print("Res1:", res1, "resZ:", resZ)
                a1.add_fitness(res1)

                ax = rand.choice(aux_list)
                # print("Agent " + str(a2.get_agent_id()) + " will play with Agent ", str(ax.get_agent_id()))
                res, n = aux.prisoners_dilemma(a2, ax, sn, eps, chi)
                res2 = res[0]
                if current_gen > converge: cooperative_acts += n
                # print("Res2:", res1, "resX:", resX)
                a2.add_fitness(res2)

            a1.set_fitness(a1.get_fitness() / Z)
            a2.set_fitness(a2.get_fitness() / Z)
            #print("f(a1):", a1.get_fitness(), "f(a2):", a2.get_fitness())

            # Calculate Probability of imitation
            p = (1 + np.exp(a1.get_fitness() - a2.get_fitness())) ** (-1)
            #print("Probability of a1 imitating a2:", p)
            if rand.random() < p:
                a1.set_trait(a2.get_trait())

    acr = 100 * cooperative_acts / games_played
    print("Final ACR: " + str(round(acr, 3)))
    print("#Cooperative acts: " + str(cooperative_acts) + ", #Played Games: " + str(games_played))
    print("#Mutations: " + str(number_mutations))
    #aux.print_population(agents)
    return acr


def read_args():
    f = open("args.txt", "r")
    lines = f.readlines()

    for line in lines:
        if line[0] == "#":
            continue
        else:
            args = line.split(" ")
            sn_list = [int(a) for a in args[0][1:-1].split(",")]
            sn = make_sn_from_list(sn_list)
            z: int = int(args[1])
            chi: float = float(args[2])
            eps: float = float(args[3])
            main(sn, z=z, chi=chi, eps=eps, runs=5)


def make_sn_from_list(l: list):
    sn = []
    entry = []
    for i in np.arange(0, len(l)-1, 2):
        entry.append((l[i], l[i+1]))
        if len(entry) == 2:
            sn.append(entry)
            entry = []
        #i += 1
    return sn


def export_results(all_runs: np.ndarray):
    builder: str = ""

    pass


read_args()
