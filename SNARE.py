import sys
import numpy as np
import random as rand
from agent import Agent
from tqdm import tqdm
import aux_functions as aux
from ModelParameters import ModelParameters as MP
import multiprocessing

# social norm - new reputation : socialNorm[action][rec_reputation][emotion profile]
# [ [ [DBM,DBN],[DGM,DGN] ],[ [CBM,CBN],[CGM,CGN] ] ]
# R   ....0.... ....1....     ....0.... ....1....
# A  ..........0...........  ...........1..........
# ----------------------------------------------------
# B = 0, G = 1
# D = 0, C = 1
# M = 0, N = 1
# -------------


def main(mp: MP):

    # list with ACR results of each run
    all_games = np.zeros(mp.runs)
    for r in range(mp.runs):
        all_games[r] = simulation(mp, r)

    print([round(a, 3) for a in all_games])
    print("Final Average of ", mp.runs, " runs: ", all_games.mean())


def simulation(model_parameters: MP, run: int):

    # Population Size
    z = model_parameters.z
    # Strategy Exploration Probability
    mu: float = model_parameters.mu / model_parameters.z
    # Reputation Assessment Error Probability
    chi: float = model_parameters.chi / model_parameters.z
    # (cooperation) Execution Error
    eps: float = model_parameters.eps / model_parameters.z
    # Judge assignment error
    alpha: float = model_parameters.alpha / model_parameters.z
    # Number of Generations to run
    gens: int = model_parameters.gens * model_parameters.z
    # Social Norm
    social_norm = model_parameters.social_norm
    # EB Social norm
    eb_social_norm = model_parameters.ebsn
    # Converging period
    converge: int = model_parameters.converge
    # Probability of looking at emotion
    gamma: float = model_parameters.gamma
    # benefit to cost ratio
    benefit: int = model_parameters.benefit
    cost: int = model_parameters.cost
    bc_ratio: float = benefit/cost
    # selection strength beta
    selection_strength: float = model_parameters.selection_strength

    print("--------------------------------------------NEW RUN--------------------------------------------------------")
    print("Run:", run, ", Z:", z, ", Gens:", gens, ", mu:", mu, ", eps:", eps, ", chi:", chi, ", alpha:", alpha,
          " gamma:", gamma, ", pdx:", model_parameters.paradoxical_strats, ", b/c ratio:", bc_ratio, ", beta:", selection_strength)

    games_played = 0
    cooperative_acts = 0
    number_mutations = 0
    aux.print_sn(social_norm)
    aux.print_ebnorm(eb_social_norm)
    # Initialization
    agents: list[Agent] = []
    for i in range(z):
        a = Agent(i, MP)
        agents.append(a)

    for current_gen in tqdm(range(gens)):
        if rand.random() < mu:
            # Trait Exploration
            a1 = aux.get_random_agent(agents)
            a1.trait_mutation(model_parameters)
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
            # increment # of played games, 4 because it's each PD has 2 donation games
            for i in range(z):
                # (each prisoner's dilemma can have at most 2 cooperative acts)
                if current_gen > converge: games_played += 4

                az = rand.choice(aux_list)
                res, n = aux.prisoners_dilemma(a1, az, eb_social_norm, social_norm, eps, chi, alpha, gamma, benefit, cost)
                if current_gen > converge: cooperative_acts += n
                a1.add_fitness(res[0])

                ax = rand.choice(aux_list)
                res, n = aux.prisoners_dilemma(a2, ax, eb_social_norm, social_norm, eps, chi, alpha, gamma, benefit, cost)
                if current_gen > converge: cooperative_acts += n
                a2.add_fitness(res[0])

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

    acr = 100 * cooperative_acts / games_played
    print("\nFinal ACR: " + str(round(acr, 3)))
    #print("#Cooperative acts: " + str(cooperative_acts) + ", #Played Games: " + str(games_played))
    #print("#Mutations: " + str(number_mutations))
    #aux.print_population(agents)
    aux.export_results(acr, model_parameters, agents)
    return acr


# Read lines from args.txt and run each one
def read_args(process_id):
    file_name: str = str(sys.argv[1])
    f = open(file_name, "r")
    lines = f.readlines()

    for line in lines:
        if line[0] == "#":
            continue
        else:
            print("Current Instruction:", line)
            args = line.split(" ")
            # [1:-1] is because of the ()
            ebsn_list: list = [int(a) for a in args[0][1:-1].split(",")]
            eb_sn: list = aux.make_ebsn_from_list(ebsn_list)
            sn_list: list = [int(b) for b in args[1][1:-1].split(",")]
            sn: list = aux.make_sn_from_list(sn_list)
            pdx: bool = args[2] == "true"
            z: int = int(args[3])
            mu: float = float(args[4])
            chi: float = float(args[5])
            eps: float = float(args[6])
            alpha: float = float(args[7])
            gamma: float = float(args[8])
            ben: int = int(args[9])
            cost: int = int(args[10])
            beta: float = float(args[11])
            model_parameters: MP = MP(args[1], sn, args[0], eb_sn, z, mu, chi, eps, alpha, gamma,
                                      runs=1, gens=5000, pdx_strats=pdx, b=ben, c=cost, beta=beta)
            main(model_parameters)
    f.close()


if __name__ == '__main__':
    num_simulations: int = 50
    num_cores = 48
    with multiprocessing.Pool(num_cores) as pool:
        pool.map(read_args, range(num_simulations))

