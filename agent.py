import random as rand

class Agent:
    def __init__(self, agent_id: int, mp):
        self._agent_id: int = agent_id
        self._reputation: int = rand.randint(0, 1)
        self._fitness_score: int = 0
        self._strategy: tuple = (-1, -1)
        self._emotion_profile: int = -1
        self.trait_mutation(mp)

    def trait_mutation(self, mp):
        self._emotion_profile = rand.randint(0,1)
        self._strategy = (rand.randint(0, 1), rand.randint(0, 1))
        #if not mp.paradoxical_strats:
        #    if (self._strategy == (0, 0) and self._emotion_profile == 1) or \
        #            (self._strategy == (1, 1) and self._emotion_profile == 0):
        #        self.trait_mutation(mp)

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

    def add_fitness(self, f):
        self._fitness_score += f
