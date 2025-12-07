import random as rand
from constants import *


class Agent:
    def __init__(self, agent_id: int, min_gamma, max_gamma, gamma_normal_center, gamma_delta):
        self._agent_id: int = agent_id
        self._fitness_score: int = 0
        self._strategy: Strategy = None
        self._emotion_profile: EmotionProfile = None
        #self._reputation: int = rand.randint(0, 1)
        self._gamma: float = -1
        self.initialise_traits(min_gamma, max_gamma, gamma_normal_center, gamma_delta)

    def initialise_traits(self, min_gamma, max_gamma, center, delta):
        # strategy
        self._strategy: Strategy = rand.choice(list(Strategy))
        # emotion profile
        self._emotion_profile = rand.choice(list(EmotionProfile))
        # gamma
        if center == 0 or center == 1 or delta == 0:
            # homogeneous population
            self._gamma = center
        else:
            self._gamma = round(sample_normal_bounded(center, min_bound=min_gamma, max_bound=max_gamma, std=delta), 1)

    def trait_mutation(self, min_gamma: float, max_gamma: float, gamma_delta: float):
        # emotion profile
        self._emotion_profile = self.emotion_profile.mutate()
        # strategy
        self._strategy: Strategy = rand.choice(list(set(Strategy) - {self.strategy}))
        self._gamma += rand.choice((-gamma_delta, gamma_delta))
        # clamp gamma
        self._gamma = max(min_gamma, min(max_gamma, self._gamma))

    def gamma(self):
        return self._gamma

    def set_gamma(self, new_gamma: float):
        self._gamma = new_gamma

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def emotion_profile(self) -> EmotionProfile:
        return self._emotion_profile

    def get_trait(self):
        return self._strategy, self._emotion_profile

    def set_strategy(self, strat):
        self._strategy = strat

    def set_emotion_profile(self, ep: EmotionProfile):
        self._emotion_profile = ep

    # Getter for agent_id
    def get_agent_id(self):
        return self._agent_id

    # Setter for agent_id
    def set_agent_id(self, agent_id: int):
        self._agent_id = agent_id

    # Getter for fitness_score
    def get_fitness(self):
        return self._fitness_score

    # Setter for fitness_score
    def set_fitness(self, fitness_score):
        self._fitness_score = fitness_score

    # Increment Fitness method
    def add_fitness(self, f):
        self._fitness_score += f

    def reputation(self):
        return self._reputation

    def set_reputation(self, new_rep: int):
        self._reputation = new_rep


def sample_normal_bounded(m: float, min_bound=0.0, max_bound=1.0, std: float = 0.1) -> float:
    """
    Sample from a normal distribution centered at m, repeat until result is in [0, 1].
    """
    while True:
        sample = rand.gauss(m, std)
        if min_bound <= sample <= max_bound:
            return sample