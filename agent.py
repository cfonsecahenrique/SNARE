import random as rand
import numpy as np
ALWAYS_COOPERATE = (1, 1)
DISCRIMINATE = (0, 1)
PARADOXICALLY_DISC = (1, 0)
ALWAYS_DEFECT = (0, 0)


class Agent:
    def __init__(self, agent_id: int, min_gamma, max_gamma):
        self._agent_id: int = agent_id
        self._reputation: int = rand.randint(0, 1)
        self._fitness_score: int = 0
        self._strategy: tuple = (-1, -1)
        self._emotion_profile: int = -1
        self._gamma: float = -1
        self.initialise_traits(min_gamma, max_gamma)

    def initialise_traits(self, min_gamma, max_gamma):
        # strategy
        self._strategy = (rand.randint(0, 1), rand.randint(0, 1))
        # emotion profile
        self._emotion_profile = rand.randint(0, 1)
        # gamma
        steps = np.arange(min_gamma, max_gamma + .01, 0.1)
        #print(min_gamma, max_gamma, steps)
        self._gamma = rand.choice(steps)

    def trait_mutation(self, min_gamma, max_gamma):
        # emotion profile
        self._emotion_profile = 1 - self._emotion_profile
        # strategy
        self._strategy = rand.choice(list({ALWAYS_COOPERATE, DISCRIMINATE, PARADOXICALLY_DISC, ALWAYS_DEFECT}
                                          - {self._strategy}))
        # GAMMA; mutate gamma; TODO: Change to delta_gamma, tbd
        #._gamma += rand.choice((-1, 1)) * 0.1
        # clamp gamma
        #self._gamma = max(min_gamma, min(max_gamma, self._gamma))

    def gamma(self):
        return self._gamma

    def set_gamma(self, new_gamma: float):
        self._gamma = new_gamma

    def strategy(self):
        return self._strategy

    def emotion_profile(self):
        return self._emotion_profile

    def set_strategy(self, strat):
        self._strategy = strat

    def set_emotion_profile(self, ep):
        self._emotion_profile = ep

    # Getter for agent_id
    def get_agent_id(self):
        return self._agent_id
        # Setter for agent_id

    def set_agent_id(self, agent_id: int):
        self._agent_id = agent_id

    # Getter for reputation
    def get_reputation(self):
        return self._reputation

    # Setter for reputation
    def set_reputation(self, reputation: int):
        self._reputation = reputation

    # Getter for fitness_score
    def get_fitness(self):
        return self._fitness_score

    # Setter for fitness_score
    def set_fitness(self, fitness_score):
        self._fitness_score = fitness_score

    # Increment Fitness method
    def add_fitness(self, f):
        self._fitness_score += f
