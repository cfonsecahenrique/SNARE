from agent import Agent
import numpy as np
import aux_functions as aux
from constants import *


class Model:

    def __init__(self, sn_list: list, ebsn_list: list, z: int,
                 mu: float, chi: float, eps: float, alpha: float, min_gamma: float, max_gamma: float,
                 gamma_delta: float, gamma_normal_center: float, gens: int, b: int, c: int, beta: float, convergence: float):
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

    def __str__(self) -> str:
        s = (
            "\nModel Parameters:\n"
            "---------------------------------------------\n"
            f"{'Population size (Z):':<30} {self._z:<20} {'Generations (G):':<30} {self._gens}\n"
            f"{'Convergence time: (gens)':<30} {self._converge:<20} {'Strategy exploration (μ):':<30} {self._mu:.6f}\n"
            f"{'Reputation error (χ):':<30} {self._chi:.6f} {'Execution error (ε):':<30} {self._eps:.6f}\n"
            f"{'Judge assignment error (α):':<30} {self._alpha:.6f} {'Selection strength (β):':<30} {self._beta:.6f}\n"
            f"{'Benefit (b):':<30} {self._b:<20} {'Cost (c):':<30} {self._c}\n"
            f"{'Gamma range:':<30} [{self._min_gamma:.3f}, {self._max_gamma:.3f}]  {'':<30} \n"
            f"{'Social Norm (string):':<30} {self.social_norm_str:<20} {'EB Social Norm (string):':<30} {self.ebsn_str}"
        )
        print(s)
        print("\nSocial Norm Details:")
        aux.print_sn(self._social_norm)
        print("\nEB Social Norm Details:")
        aux.print_ebnorm(self._eb_social_norm)
        return ""

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
                       + "\t" + str(self._eps) + "\t" + str(self._alpha) + "\t" + str(self._b) + "\t" \
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

        a1_rep = agent1.reputation()
        a2_rep = agent2.reputation()

        # Rep assessment error
        a1_action = agent1.strategy.value[aux.invert_binary(a2_rep)] \
            if random_vals[ri] < self.reputation_assessment_error else agent1.strategy.value[a2_rep]
        ri += 1
        a2_action = agent2.strategy.value[aux.invert_binary(a1_rep)] \
            if random_vals[ri] < self.reputation_assessment_error else agent2.strategy.value[a1_rep]
        ri += 1

        # Execution errors
        if random_vals[ri] < self.execution_error:
            a1_action = aux.invert_binary(a1_action)
        ri += 1
        if random_vals[ri] < self.execution_error:
            a2_action = aux.invert_binary(a2_action)
        ri += 1

        # Social norm updates
        new_rep_1 = self.ebsn[a1_action][a2_rep][agent1.emotion_profile.value] \
            if random_vals[ri] < agent1.gamma() else self.social_norm[a1_action][a2_rep]
        ri += 1
        new_rep_2 = self.ebsn[a2_action][a1_rep][agent2.emotion_profile.value] \
            if random_vals[ri] < agent2.gamma() else self.social_norm[a2_action][a1_rep]
        ri += 1

        # Assignment errors
        if random_vals[ri] < self.reputation_assignment_error:
            new_rep_1 = aux.invert_binary(new_rep_1)
        ri += 1
        if random_vals[ri] < self.reputation_assignment_error:
            new_rep_2 = aux.invert_binary(new_rep_2)
        ri += 1

        agent1.set_reputation(new_rep_1)
        agent2.set_reputation(new_rep_2)

        cooperative_acts += a1_action
        cooperative_acts += a2_action

        return pd[a1_action, a2_action], cooperative_acts, ri


