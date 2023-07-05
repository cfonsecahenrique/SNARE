import random as rand

class Agent:
    def __init__(self, agent_id: int, mp):
        self._agent_id = agent_id
        self._reputation = rand.randint(0, 1)
        self._trait = None
        self.trait_mutation(mp)
        self._fitness_score = 0

    def trait_mutation(self, mp):
        self._trait = ((rand.randint(0, 1), rand.randint(0, 1)), rand.randint(0, 1))
        if not mp.paradoxical_strats:
            if self._trait == ((0, 0), 1) or self._trait == ((1, 1), 0):
                self.trait_mutation(mp)


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

        # Getter for trait

    def get_trait(self):
        return self._trait

        # Setter for trait

    def set_trait(self, trait):
        self._trait = trait

        # Getter for fitness_score

    def get_fitness(self):
        return self._fitness_score

        # Setter for fitness_score

    def set_fitness(self, fitness_score):
        self._fitness_score = fitness_score

    def add_fitness(self, f):
        self._fitness_score += f
