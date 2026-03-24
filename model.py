from agent import Agent
import numpy as np
import aux_functions as aux
from constants import *


class Model:

    def __init__(self, sn_list: list, ebsn_list: list, z: int,
                 mu: float, chi: float, eps: float, alpha: float, min_gamma: float, max_gamma: float,
                 gamma_delta: float, gamma_normal_center: float, gens: int, b: int, c: int, beta: float, convergence: float, q: float, consensus_thresh: float, non_consensus_strategy: str = "emotion"):
        self._social_norm: list = sn_list
        self._eb_social_norm: list = ebsn_list
        self._mu: float = mu
        self._z: int = z
        self._gens: int = gens
        self._chi: float = chi
        self._eps: float = eps
        self._alpha: float = alpha
        self._converge: int = int(convergence * gens)
        self._b: int = b
        self._c: int = c
        self._min_gamma: float = min_gamma
        self._max_gamma: float = max_gamma
        self._gamma_delta: float = gamma_delta
        self._gamma_normal_center: float = gamma_normal_center
        self._beta: float = beta
        self._q: float = q
        self._consensus_thresh: float = consensus_thresh
        self._non_consensus_strategy: str = non_consensus_strategy
        
        # Initialize image_matrix as requested
        # 1. Every agent is assigned a random reputation
        assigned_reputations = np.random.randint(0, 2, size=z)
        # 2. The image matrix is filled in a perfectly consensual manner:
        #    a column of the matrix is filled with only the assigned reputation
        self._image_matrix = np.zeros((z, z), dtype=int)
        for j in range(z):
            self._image_matrix[:, j] = assigned_reputations[j]

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
            "Consensus threshold": f"{self._consensus_thresh:.6f}",
            "Non-consensus strategy": self._non_consensus_strategy,
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
        builder: str = self._ebsn_str.replace("[", "(").replace("]", ")").replace(" ", "") + "\t" + \
                       self._social_norm_str.replace("[", "(").replace("]", ")").replace(" ", "") + "\t" + \
                       str(self._z) + "\t" + str(self._gens) + "\t" + str(self._mu) + "\t" + str(self._chi) \
                       + "\t" + str(self._eps) + "\t" + str(self._alpha) + "\t" + str(self._q) + "\t" + str(self._b) + "\t" \
                       + str(self._c) + "\t" + str(self._beta) + "\t" + str(self._consensus_thresh) + "\t" + str(self._non_consensus_strategy)
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
    def consensus_thresh(self):
        return self._consensus_thresh

    @property
    def non_consensus_strategy(self):
        return self._non_consensus_strategy

    @property
    def image_matrix(self):
        return self._image_matrix

    def prisoners_dilemma(self, agent1: Agent, agent2: Agent, random_vals, ri: int):
        """
        Play a single Prisoner's Dilemma between agent1 and agent2.
        """
        # Payoff matrix
        T = self.benefit
        R = self.benefit - self.cost
        P = 0
        S = -self.cost
        pd = np.array([[(P, P), (T, S)],
                       [(S, T), (R, R)]], dtype=np.int32)

        cooperative_acts = 0

        # Reputations are now derived from the image matrix.
        # a1 perceives a2's reputation as:
        a2_rep_by_a1 = self.image_matrix[agent1.get_agent_id(), agent2.get_agent_id()]
        # a2 perceives a1's reputation as:
        a1_rep_by_a2 = self.image_matrix[agent2.get_agent_id(), agent1.get_agent_id()]

        # Rep assessment error
        a1_action = agent1.strategy.value[aux.invert_binary(a2_rep_by_a1)] \
            if random_vals[ri] < self.reputation_assessment_error else agent1.strategy.value[a2_rep_by_a1]
        ri += 1
        a2_action = agent2.strategy.value[aux.invert_binary(a1_rep_by_a2)] \
            if random_vals[ri] < self.reputation_assessment_error else agent2.strategy.value[a1_rep_by_a2]
        ri += 1

        # Execution errors
        if random_vals[ri] < self.execution_error:
            a1_action = aux.invert_binary(a1_action)
        ri += 1
        if random_vals[ri] < self.execution_error:
            a2_action = aux.invert_binary(a2_action)
        ri += 1

        cooperative_acts += a1_action
        cooperative_acts += a2_action

        # Optimization for q=1 (Full Observability)
        if self.observability == 1:
            z = self.population_size
            
            # --- Update Agent 1 Reputation ---
            opinions_on_a2 = self.image_matrix[:, agent2.get_agent_id()]
            
            if a2_action == COOPERATE:
                # Gamma check
                gamma_rands = random_vals[ri:ri+z]
                ri += z
                use_ebsn = gamma_rands < agent1.gamma()
                
                # Lookup tables
                sn_vals = np.array(self.social_norm[a1_action])[opinions_on_a2]
                ebsn_vals = np.array([self.ebsn[a1_action][0][agent1.emotion_profile.value], 
                                      self.ebsn[a1_action][1][agent1.emotion_profile.value]])[opinions_on_a2]
                
                new_rep_1 = np.where(use_ebsn, ebsn_vals, sn_vals)
            else:
                new_rep_1 = np.array(self.social_norm[a1_action])[opinions_on_a2]
            
            # Assignment error
            err_rands = random_vals[ri:ri+z]
            ri += z
            flips = err_rands < self.reputation_assignment_error
            new_rep_1 = np.where(flips, 1 - new_rep_1, new_rep_1)
            
            self.image_matrix[:, agent1.get_agent_id()] = new_rep_1

            # --- Update Agent 2 Reputation ---
            opinions_on_a1 = self.image_matrix[:, agent1.get_agent_id()]
            
            if a1_action == COOPERATE:
                # Gamma check
                gamma_rands = random_vals[ri:ri+z]
                ri += z
                use_ebsn = gamma_rands < agent2.gamma()
                
                # Lookup tables
                sn_vals = np.array(self.social_norm[a2_action])[opinions_on_a1]
                ebsn_vals = np.array([self.ebsn[a2_action][0][agent2.emotion_profile.value], 
                                      self.ebsn[a2_action][1][agent2.emotion_profile.value]])[opinions_on_a1]
                
                new_rep_2 = np.where(use_ebsn, ebsn_vals, sn_vals)
            else:
                new_rep_2 = np.array(self.social_norm[a2_action])[opinions_on_a1]
            
            # Assignment error
            err_rands = random_vals[ri:ri+z]
            ri += z
            flips = err_rands < self.reputation_assignment_error
            new_rep_2 = np.where(flips, 1 - new_rep_2, new_rep_2)
            
            self.image_matrix[:, agent2.get_agent_id()] = new_rep_2
            
            return pd[a1_action, a2_action], cooperative_acts, ri

        # Sample subsection of population to observe and update image matrix
        num_observers = int(self.observability * self.population_size)
        # Ensure num_observers is within valid range [0, population_size]
        num_observers = max(0, min(num_observers, self.population_size))
        if num_observers > 0:
            observer_ids = np.random.choice(self.population_size, num_observers, replace=False)
            is_a1_rep_consensual = self.is_consensual(agent1.get_agent_id(), self.consensus_thresh)
            is_a2_rep_consensual = self.is_consensual(agent2.get_agent_id(), self.consensus_thresh)
            
            for observer_id in observer_ids:
                observer_opinion_on_a1 = self.image_matrix[observer_id, agent1.get_agent_id()]
                observer_opinion_on_a2 = self.image_matrix[observer_id, agent2.get_agent_id()]

                # --- Update Agent 1 Reputation ---
                if random_vals[ri] < agent1.gamma():
                    # Use EB Social Norm if opponent cooperated
                    if a2_action == COOPERATE:
                        if is_a2_rep_consensual:
                            new_rep_1 = self.ebsn[a1_action][observer_opinion_on_a2][agent1.emotion_profile.value]
                        else:
                            # Non-consensual fallback
                            if self.non_consensus_strategy == "action":
                                new_rep_1 = a1_action  # Image Scoring: cooperate->GOOD, defect->BAD
                            else:  # "emotion" (default)
                                new_rep_1 = agent1.emotion_profile.value
                    else:
                        # Fallback to base social norm if opponent defected
                        new_rep_1 = self.social_norm[a1_action][observer_opinion_on_a2]
                else:
                    # Use Base Social Norm
                    new_rep_1 = self.social_norm[a1_action][observer_opinion_on_a2]
                ri += 1

                # --- Update Agent 2 Reputation ---
                if random_vals[ri] < agent2.gamma():
                    # Use EB Social Norm if opponent cooperated
                    if a1_action == COOPERATE:
                        if is_a1_rep_consensual:
                            new_rep_2 = self.ebsn[a2_action][observer_opinion_on_a1][agent2.emotion_profile.value]
                        else:
                            # Non-consensual fallback
                            if self.non_consensus_strategy == "action":
                                new_rep_2 = a2_action  # Image Scoring: cooperate->GOOD, defect->BAD
                            else:  # "emotion" (default)
                                new_rep_2 = agent2.emotion_profile.value
                    else:
                        # Fallback to base social norm if opponent defected
                        new_rep_2 = self.social_norm[a2_action][observer_opinion_on_a1]
                else:
                    # Use Base Social Norm
                    new_rep_2 = self.social_norm[a2_action][observer_opinion_on_a1]
                ri += 1

                # Apply assignment errors for this specific observer's perception
                if random_vals[ri] < self.reputation_assignment_error:
                    new_rep_1 = 1 - new_rep_1
                ri += 1

                if random_vals[ri] < self.reputation_assignment_error:
                    new_rep_2 = 1 - new_rep_2
                ri += 1

                self.image_matrix[observer_id, agent1.get_agent_id()] = new_rep_1
                self.image_matrix[observer_id, agent2.get_agent_id()] = new_rep_2
        # If num_observers is 0, no updates to image_matrix or consumption of random_vals for assignment errors.

        # Remove the direct setting of agent reputations, as they are now distributed in the image matrix.
        # The agent's internal _reputation field is no longer the source of truth for interactions.
        # agent1.set_reputation(new_rep_1)
        # agent2.set_reputation(new_rep_2)

        return pd[a1_action, a2_action], cooperative_acts, ri

    def is_consensual(self, agent_id: int, threshold: float) -> bool:
        """
        Checks if the reputation information regarding a given agent is consensual or not, through a threshold.
        """
        col = self.image_matrix[:, agent_id]
        good = np.sum(col)
        # Consensus is defined as the absolute difference between good and bad opinions divided by total population
        # |good - bad| / z = |good - (z - good)| / z = |2*good - z| / z
        consensus = abs(2 * good - self.population_size) / self.population_size
        return consensus >= threshold
