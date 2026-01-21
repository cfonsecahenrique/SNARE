import numpy as np
import aux_functions as aux
from constants import *
import random as rand


def sample_normal_bounded(m: float, min_bound=0.0, max_bound=1.0, std: float = 0.1) -> float:
    """
    Sample from a normal distribution centered at m, repeat until result is in [0, 1].
    """
    while True:
        sample = rand.gauss(m, std)
        if min_bound <= sample <= max_bound:
            return sample


class Model:

    def __init__(self, sn_list: list, ebsn_list: list, z: int,
                 mu: float, chi: float, eps: float, alpha: float, min_gamma: float, max_gamma: float,
                 gamma_delta: float, gamma_normal_center: float, gens: int, b: int, c: int, beta: float, convergence: float, q: float):
        self._social_norm: list = sn_list
        self._eb_social_norm: list = ebsn_list
        self._z: int = z
        self._mu: float = mu/self._z
        self._gens: int = gens
        self._chi: float = chi/self._z
        self._eps: float = eps/self._z
        self._alpha: float = alpha/self._z
        self._converge: int = int(convergence * gens)
        self._b: int = b
        self._c: int = c
        self._min_gamma: float = min_gamma
        self._max_gamma: float = max_gamma
        self._gamma_delta: float = gamma_delta
        self._gamma_normal_center: float = gamma_normal_center
        self._beta: float = beta
        self._q: float = q

        # Agent properties as numpy arrays
        self.strategies = np.random.choice(list(Strategy), size=z)
        self.emotion_profiles = np.random.choice(list(EmotionProfile), size=z)

        #self.strategy_values = np.array([s.value for s in self.strategies])
        #self.emotion_values = np.array([e.value for e in self.emotion_profiles])

        if gamma_normal_center == 0 or gamma_normal_center == 1 or gamma_delta == 0:
            self.gammas = np.full(z, gamma_normal_center)
        else:
            self.gammas = np.array([
                round(sample_normal_bounded(gamma_normal_center, min_bound=min_gamma, max_bound=max_gamma, std=gamma_delta), 1)
                for _ in range(z)
            ])

        # Initialize image_matrix
        assigned_reputations = np.random.randint(0, 2, size=z)
        self._image_matrix = np.tile(assigned_reputations, (z, 1))
        
        self.sn_arr = np.array(self._social_norm)
        self.ebsn_arr = np.array(self._eb_social_norm)

    def __str__(self) -> str:
        # Box drawing characters
        top_left = "╔"
        top_right = "╗"
        bottom_left = "╚"
        bottom_right = "╝"
        horizontal = "═"
        vertical = "║"

        width = 80

        def center(text, width):
            return text.center(width)

        s = f"{top_left}{horizontal * (width - 2)}{top_right}\n"
        s += f"{vertical}{center(f'Model Parameters', width - 2)}{vertical}\n"
        s += f"{vertical}{horizontal * (width - 2)}{vertical}\n"

        # Parameters
        params = {
            "Population size (Z)": self._z,
            "Generations (G)": self._gens,
            "Convergence time (gens)": self._converge,
            "Strategy exploration (μ)": f"{self._mu:.6f}",
            "Reputation error (χ)": f"{self._chi:.6f}",
            "Execution error (ε)": f"{self._eps:.6f}",
            "Judge assignment error (α)": f"{self._alpha:.6f}",
            "Private-Public assessment (q)": f"{self._q:.6f}",
            "Selection strength (β)": f"{self._beta:.6f}",
            "Benefit (b)": self._b,
            "Cost (c)": self._c,
            "Gamma range": f"[{self._min_gamma:.3f}, {self._max_gamma:.3f}]",
        }

        for name, value in params.items():
            s += f"{vertical} {name:<30} {str(value):<45} {vertical}\n"

        s += f"{vertical}{horizontal * (width - 2)}{vertical}\n"

        # Social Norms
        sn_str = aux.format_sn(self._social_norm)
        ebsn_str = aux.format_ebnorm(self._eb_social_norm)

        sn_lines = sn_str.split('\n')
        ebsn_lines = ebsn_str.split('\n')

        max_lines = max(len(sn_lines), len(ebsn_lines))

        # Define column widths
        sn_width = 38
        ebsn_width = 38

        for i in range(max_lines):
            sn_line = sn_lines[i] if i < len(sn_lines) else ''
            ebsn_line = ebsn_lines[i] if i < len(ebsn_lines) else ''
            s += f"{vertical} {sn_line.ljust(sn_width)} {ebsn_line.ljust(ebsn_width)} {vertical}\n"

        s += f"{bottom_left}{horizontal * (width - 2)}{bottom_right}"
        return s

    def run_generation(self, rng):
        total_generation_cooperation, total_generation_games = 0, 0
        shuffled_indices = np.random.permutation(self.population_size)
        for agent_idx in shuffled_indices:
            cooperation, total_games = self.single_agent_evolution(agent_idx, rng)
            total_generation_cooperation += cooperation
            total_generation_games += total_games
        return total_generation_cooperation, total_generation_games


    def single_agent_evolution(self, focal_agent_idx, rng):
        if rng.random() < self._mu:
            self._mutate_agent(focal_agent_idx)
            return 0, 0  # No games played in mutation

        # Social learning
        role_model_idx = rng.choice(np.delete(np.arange(self.population_size), focal_agent_idx))

        # Play z games for both agents
        focal_fitness, focal_coop, focal_games = self._play_games(focal_agent_idx, rng)
        role_model_fitness, role_model_coop, role_model_games = self._play_games(role_model_idx, rng)

        pi: float = (1 + np.exp(self.selection_strength * (focal_fitness - role_model_fitness))) ** (-1)
        # Compare fitness and imitate
        if rng.random() < pi:
            self.strategies[focal_agent_idx] = self.strategies[role_model_idx]
            self.emotion_profiles[focal_agent_idx] = self.emotion_profiles[role_model_idx]
            self.gammas[focal_agent_idx] = self.gammas[role_model_idx]

        return focal_coop + role_model_coop, focal_games + role_model_games


    def _play_games(self, agent_idx, rng):
        num_opponents = self.population_size - 1
        opponents = np.delete(np.arange(self.population_size), agent_idx)

        # Get reputations
        agent_reps_for_opponents = self.image_matrix[agent_idx, opponents]
        opponent_reps_for_agent = self.image_matrix[opponents, agent_idx]

        # Apply reputation assessment error
        error_mask_agent = rng.random(num_opponents) < self.reputation_assessment_error
        error_mask_opponent = rng.random(num_opponents) < self.reputation_assessment_error
        agent_reps_for_opponents = np.where(error_mask_agent, 1 - agent_reps_for_opponents, agent_reps_for_opponents)
        opponent_reps_for_agent = np.where(error_mask_opponent, 1 - opponent_reps_for_agent, opponent_reps_for_agent)

        # Determine actions
        agent_actions = np.zeros(len(agent_reps_for_opponents))
        for i in range(len(agent_reps_for_opponents)):
            agent_actions[i] = self.strategies[agent_idx].value[agent_reps_for_opponents[i]]

        opponent_actions = np.zeros(len(opponent_reps_for_agent))
        for i in range(len(opponent_reps_for_agent)):
            opponent_actions[i] = self.strategies[opponents[i]].value[opponent_reps_for_agent[i]]


        # Apply execution errors
        error_mask_agent_action = rng.random(num_opponents) < self.execution_error
        error_mask_opponent_action = rng.random(num_opponents) < self.execution_error
        agent_actions = np.where(error_mask_agent_action, 1 - agent_actions, agent_actions)
        opponent_actions = np.where(error_mask_opponent_action, 1 - opponent_actions, opponent_actions)

        # Calculate payoffs
        fitness = np.sum(np.where(agent_actions == COOPERATE, -self.cost, 0)) + \
                  np.sum(np.where(opponent_actions == COOPERATE, self.benefit, 0))

        cooperative_acts = np.sum(agent_actions) + np.sum(opponent_actions)
        games_played = num_opponents * 2

        # Observer logic
        num_observers = int(self.observability * self.population_size)
        if num_observers > 0:
            # Generate random observers for each game (column-wise)
            # Shape: (num_observers, num_opponents)
            rand_matrix = rng.random((self.population_size, num_opponents))
            observer_indices = np.argsort(rand_matrix, axis=0)[:num_observers]
            
            # Vectorized judging
            use_ebsn_mask = rng.random((num_observers, num_opponents)) < self.gammas[observer_indices]
            
            # Broadcast actions to match (num_observers, num_opponents)
            agent_actions_broad = np.broadcast_to(agent_actions, (num_observers, num_opponents))
            opponent_actions_broad = np.broadcast_to(opponent_actions, (num_observers, num_opponents))
            
            # Get reputations from the perspective of the observers
            opponents_broad = np.broadcast_to(opponents, (num_observers, num_opponents))
            
            # Observers' view of the opponent (recipient of agent's action)
            observer_reps_of_opponents = self.image_matrix[observer_indices, opponents_broad]
            
            # Observers' view of the agent (recipient of opponent's action)
            observer_reps_of_agent = self.image_matrix[observer_indices, agent_idx]

            # Apply reputation assessment error for observers
            error_mask_obs_opp = rng.random((num_observers, num_opponents)) < self.reputation_assessment_error
            error_mask_obs_agent = rng.random((num_observers, num_opponents)) < self.reputation_assessment_error

            # Wrap these in .astype(int) because the math (1 - x) forces them to float64
            observer_reps_of_opponents = np.where(error_mask_obs_opp, 1 - observer_reps_of_opponents,
                                                  observer_reps_of_opponents).astype(int)
            observer_reps_of_agent = np.where(error_mask_obs_agent, 1 - observer_reps_of_agent,
                                              observer_reps_of_agent).astype(int)

            # Create an array of the actual integer values stored inside the profile objects
            # We need these to match the (num_observers, num_opponents) shape
            opponent_emotion_values_raw = np.array([ep.value for ep in self.emotion_profiles[opponents]])
            # Broadcast opponent emotions across all observers
            opp_emotion_broad = np.broadcast_to(opponent_emotion_values_raw, (num_observers, num_opponents))

            # Get the agent's specific emotion value
            agent_emotion_val = self.emotion_profiles[agent_idx].value

            # Ensure actions are also integers (just in case)
            agent_actions_idx = agent_actions_broad.astype(int)
            opp_actions_idx = opponent_actions_broad.astype(int)

            # Assess agent actions (agent acts on opponent)
            new_reps_agent = np.where(use_ebsn_mask,
                                      self.ebsn_arr[
                                          agent_actions_idx,
                                          observer_reps_of_opponents,
                                          agent_emotion_val  # Scalar is okay here as it broadcasts
                                      ],
                                      self.sn_arr[agent_actions_idx, observer_reps_of_opponents])

            # Assess opponent actions (opponent acts on agent)
            new_reps_opponent = np.where(use_ebsn_mask,
                                         self.ebsn_arr[
                                             opp_actions_idx,
                                             observer_reps_of_agent,
                                             opp_emotion_broad  # Use the broadcasted version
                                         ],
                                         self.sn_arr[opp_actions_idx, observer_reps_of_agent])

            # Apply assignment error
            error_mask_agent = rng.random((num_observers, num_opponents)) < self.reputation_assignment_error
            error_mask_opponent = rng.random((num_observers, num_opponents)) < self.reputation_assignment_error
            
            new_reps_agent = np.where(error_mask_agent, 1 - new_reps_agent, new_reps_agent)
            new_reps_opponent = np.where(error_mask_opponent, 1 - new_reps_opponent, new_reps_opponent)

            # Update image matrix
            # Note: If an observer sees multiple games, the last update in the batch effectively wins (or random depending on implementation)
            # This is acceptable for simultaneous batch updates
            self.image_matrix[observer_indices, agent_idx] = new_reps_agent
            self.image_matrix[observer_indices, opponents_broad] = new_reps_opponent

        return fitness, cooperative_acts, games_played

    def _mutate_agent(self, agent_idx):
        # Mutate emotion profile (flip)
        self.emotion_profiles[agent_idx] = EmotionProfile(1 - self.emotion_profiles[agent_idx].value)

        # Mutate strategy
        current_strategy = self.strategies[agent_idx]
        possible_strategies = list(set(Strategy) - {current_strategy})
        self.strategies[agent_idx] = rand.choice(possible_strategies)

        # Mutate gamma
        if self.gamma_normal_center == 0 or self.gamma_normal_center == 1 or self.gamma_delta == 0:
            return
        else:
            self.gammas[agent_idx] += rand.choice([-self._gamma_delta, self._gamma_delta])
            self.gammas[agent_idx] = np.clip(self.gammas[agent_idx], self._min_gamma, self._max_gamma)

    @property
    def social_norm_str(self):
        return str(self._social_norm)

    @property
    def ebsn_str(self):
        return str(self._eb_social_norm)

    @property
    def gamma_normal_center(self):
        return self._gamma_normal_center

    @property
    def gamma_delta(self):
        return self._gamma_delta

    @property
    def min_gamma(self):
        return self._min_gamma

    @property
    def max_gamma(self):
        return self._max_gamma

    def generate_mp_string(self) -> str:
        # for results exporting
        builder: str = self.ebsn_str.replace("[", "(").replace("]", ")").replace(" ", "") + "\t" + \
                       self.social_norm_str.replace("[", "(").replace("]", ")").replace(" ", "") + "\t" + \
                       str(self._z) + "\t" + str(self._gens) + "\t" + str(self._mu) + "\t" + str(self._chi) \
                       + "\t" + str(self._eps) + "\t" + str(self._alpha) + "\t" + str(self._q) + "\t" + str(self._b) + "\t" \
                       + str(self._c) + "\t" + str(self._beta)
        return builder

    @property
    def ebsn(self):
        # emotion-based social norm
        return self._eb_social_norm

    @property
    def beta(self):
        return self._beta

    @property
    def execution_error(self):
        return self._eps

    @property
    def social_norm(self):
        return self._social_norm

    @property
    def mutation_rate(self):
        return self._mu

    @property
    def generations(self):
        return self._gens

    @property
    def reputation_assessment_error(self):
        return self._chi

    @property
    def population_size(self):
        return self._z

    @property
    def converge(self):
        return self._converge

    @property
    def reputation_assignment_error(self):
        return self._alpha

    @property
    def benefit(self):
        return self._b

    @property
    def cost(self):
        return self._c

    @property
    def selection_strength(self):
        return self._beta

    @property
    def observability(self):
        return self._q

    @property
    def image_matrix(self):
        return self._image_matrix
