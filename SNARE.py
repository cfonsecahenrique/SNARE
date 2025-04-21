import sys
import numpy as np
import random as rand
import yaml
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
BAD, DEFECT, MEAN = 0, 0, 0
GOOD, COOPERATE, NICE = 1, 1, 1
# -------------


def main(mp: MP, lock: multiprocessing.Lock):
    acr: float = simulation(mp, lock)
    print(round(acr, 3))


# Read lines from args.txt and run each one
def simulation(model_parameters: MP, lock: multiprocessing.Lock):

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
    # benefit to cost ratio
    benefit: int = model_parameters.benefit
    cost: int = model_parameters.cost
    bc_ratio: float = benefit/cost
    # selection strength beta
    selection_strength: float = model_parameters.selection_strength

    min_gamma: float = model_parameters.min_gamma
    max_gamma: float = model_parameters.max_gamma

    print("\n--------------------------------------------NEW RUN------------------------------------------------------")
    print("Z:", z, ", Gens:", gens, ", mu:", mu, ", eps:", eps, ", chi:", chi, ", alpha:", alpha,
          ", b/c ratio:", bc_ratio, ", beta:", selection_strength, ", min:_gamma:", min_gamma, ", max_gamma:", max_gamma)

    games_played = 0
    cooperative_acts = 0
    number_mutations = 0
    aux.print_sn(social_norm)
    aux.print_ebnorm(eb_social_norm)
    # Initialization
    agents: list[Agent] = []
    a1: Agent
    a2: Agent

    for i in range(z):
        a = Agent(i, min_gamma, max_gamma)
        agents.append(a)

    for current_gen in tqdm(range(gens)):
        if rand.random() < mu:
            # Trait Exploration
            a1 = aux.get_random_agent(agents)
            a1.trait_mutation(min_gamma, max_gamma)
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
                res, n = aux.prisoners_dilemma(a1, az, eb_social_norm, social_norm, eps, chi, alpha, benefit, cost)
                if current_gen > converge: cooperative_acts += n
                a1.add_fitness(res[0])

                ax = rand.choice(aux_list)
                res, n = aux.prisoners_dilemma(a2, ax, eb_social_norm, social_norm, eps, chi, alpha, benefit, cost)
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
                a1.set_gamma(a2.gamma())

    acr = 100 * cooperative_acts / games_played
    print("\nFinal ACR: " + str(round(acr, 3)))
    #print("#Cooperative acts: " + str(cooperative_acts) + ", #Played Games: " + str(games_played))
    #print("#Mutations: " + str(number_mutations))
    #aux.print_population(agents)

    aux.export_results(acr, model_parameters, agents, lock)
    return acr


def read_args(mp_args: tuple):
    process_id, lock = mp_args  # Unpack the arguments
    # Reads the arguments of the .yaml file
    # Executes on __main__ function for each different instruction/experiment
    file_name: str = str(sys.argv[1])

    # Open and parse the YAML file
    with open(file_name, "r") as f:
        data = yaml.safe_load(f)

    # Loop through each set of parameters in the YAML file
    for entry in data.get("instructions", []):
        print("\nCurrent Instruction:", entry)
        ebsn_list: list = entry["ebsn"]
        eb_sn: list[list] = aux.make_ebsn_from_list(ebsn_list)
        sn_list: list = entry["sn"]
        sn: list[list] = aux.make_sn_from_list(sn_list)
        z: int = int(entry["z"])
        mu: float = float(entry["mu"])
        chi: float = float(entry["chi"])
        eps: float = float(entry["eps"])
        alpha: float = float(entry["alpha"])
        benefit: int = int(entry["benefit"])
        cost: int = int(entry["cost"])
        beta: float = float(entry["beta"])
        generations: int = int(entry["generations"])
        min_gamma: float = float(entry["gamma_min"])
        max_gamma: float = float(entry["gamma_max"])

        model_parameters: MP = MP(
            str(entry["sn"]), sn, str(entry["ebsn"]), eb_sn, z, mu, chi, eps, alpha,
            min_gamma=min_gamma, max_gamma=max_gamma,
            gens=generations, b=benefit, c=cost, beta=beta
        )

        main(model_parameters, lock)


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    if len(sys.argv) == 1:
        raise ValueError("No experiment configuration passed!")
    elif len(sys.argv) == 4:
        num_simulations: int = int(sys.argv[2])
        num_cores: int = int(sys.argv[3])
    else:
        print("No number of runs or cores specified. Running default values")
        num_simulations: int = 50
        num_cores = 48
    print("Running", num_simulations, "independent parallel simulations over", num_cores, "cpu core(s).")

    args_list = [(i, lock) for i in range(num_simulations)]
    with multiprocessing.Pool(num_cores) as pool:
        pool.map(read_args, args_list)

