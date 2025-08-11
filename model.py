from agent import Agent
import numpy as np
import random as rand
import aux_functions as aux


class Model:

    def __init__(self, sn_str: str, sn_list: list, ebsn_str: str, ebsn_list: list, z: int,
                 mu: float, chi: float, eps: float, alpha: float, min_gamma: float, max_gamma: float,
                 gamma_delta: float, gamma_normal_center: float, gens: int, b: int, c: int, beta: float, convergence: float):
        self._social_norm: list = sn_list
        self._social_norm_str: str = sn_str
        self._eb_social_norm: list = ebsn_list
        self._ebsn_str: str = ebsn_str
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
        # self.image_matrix = np.ones((z, z))

    def __str__(self) -> str:
        s = (
            "\nModel Parameters:\n"
            "---------------------------------------------\n"
            f"{'Population size (Z):':<30} {self._z:<20} {'Generations (G):':<30} {self._gens}\n"
            f"{'Convergence time:':<30} {self._converge:<20} {'Strategy exploration (μ):':<30} {self._mu:.6f}\n"
            f"{'Reputation error (χ):':<30} {self._chi:.6f} {'Execution error (ε):':<30} {self._eps:.6f}\n"
            f"{'Judge assignment error (α):':<30} {self._alpha:.6f} {'Selection strength (β):':<30} {self._beta:.6f}\n"
            f"{'Benefit (b):':<30} {self._b:<20} {'Cost (c):':<30} {self._c}\n"
            f"{'Gamma range:':<30} [{self._min_gamma:.3f}, {self._max_gamma:.3f}]  {'':<30} \n"
            f"{'Social Norm (string):':<30} {self._social_norm_str:<20} {'EB Social Norm (string):':<30} {self._ebsn_str}"
        )
        print(s)
        print("\nSocial Norm Details:")
        aux.print_sn(self._social_norm)
        print("\nEB Social Norm Details:")
        aux.print_ebnorm(self._eb_social_norm)
        return ""
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
    def eps(self):
        return self._eps

    @property
    def social_norm(self):
        return self._social_norm

    @property
    def social_norm_str(self):
        return self._social_norm_str

    @property
    def mu(self):
        return self._mu

    @property
    def gens(self):
        return self._gens

    @property
    def chi(self):
        return self._chi

    @property
    def z(self):
        return self._z

    @property
    def converge(self):
        return self._converge

    @property
    def alpha(self):
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

    """
    def get_opinion(self, observer: Agent, observed: Agent) -> int:
        return int(self.image_matrix[observer.get_agent_id(), observed.get_agent_id()])

    def set_opinion(self, new_opinion: int, judge: Agent, judged: Agent):
        self.image_matrix[judge.get_agent_id(), judged.get_agent_id()] = new_opinion
    """

    def prisoners_dilemma(self, agent1: Agent, agent2: Agent):
        # Payoff matrix of the prisoner's dilemma (pd)
        # also a DG with b>c
        # pd = ( [ [D,D],[D,C] ],[ [C,D],[C,C] ] )
        # (T)emptation; (R)eward; (P)unishment; (S)ucker's payoff
        T: int = self.benefit
        R: int = self.benefit - self.cost
        P: int = 0
        S: int = -self.cost
        pd = np.array([
            [(P, P), (T, S)],
            [(S, T), (R, R)]
        ])

        cooperative_acts: int = 0
        # Get agent's reputations locally so that the changed reputations aren't used in the wrong place
        a1_rep: int = agent1.reputation()
        a2_rep: int = agent2.reputation()

        # Check for rep assessment error
        if rand.random() < self.chi:
            a1_action: int = agent1.strategy.value[aux.invert_binary(a2_rep)]
        else:
            a1_action: int = agent1.strategy.value[a2_rep]

        if rand.random() > self.chi:
            a2_action: int = agent2.strategy.value[a1_rep]
        else:
            a2_action: int = agent2.strategy.value[aux.invert_binary(a1_rep)]

        # Execution error for action of agent 1
        if rand.random() < self.eps:
            a1_action = aux.invert_binary(a1_action)
            # print("Agent " + str(agent1.get_agent_id()) + " tried to cooperate but failed!")

        # Execution error for action of agent 2
        if rand.random() < self.eps:
            a2_action = aux.invert_binary(a2_action)

        # DONOR FOCAL would be
        # gamma: new_rep: int = EBSN[a1_rep][self.get_opinion(observer, agent1)][agent1.emotion_profile()]
        # 1-gamma: new_rep: int = SN[a1_action][self.get_opinion(observer, agent1)]
        if rand.random() < agent1.gamma():
            # Look at Emotion Based Social Norm
            # RECIPIENT FOCAL (common in IR)
            new_rep_1: int = self.ebsn[a1_action][agent2.reputation()][agent1.emotion_profile()]
            # DONOR FOCAL
            # new_rep_1: int = self.ebsn[a1_action][agent1.reputation()][agent1.emotion_profile()]
        else:
            # Look at simple social norm
            # RECIPIENT FOCAL (common in IR)
            new_rep_1: int = self.social_norm[a1_action][agent2.reputation()]
            # DONOR FOCAL
            # new_rep_1: int = self.social_norm[a1_action][agent1.reputation()]

        if rand.random() < agent2.gamma():
            new_rep_2: int = self.ebsn[a2_action][agent1.reputation()][agent2.emotion_profile()]
            # new_rep_2: int = self.ebsn[a2_action][agent2.reputation()][agent2.emotion_profile()]
        else:
            new_rep_2: int = self.social_norm[a2_action][agent1.reputation()]
            # new_rep_2: int = self.social_norm[a2_action][agent2.reputation()]

        # Assignment error
        if rand.random() < self.alpha:
            new_rep_1 = aux.invert_binary(new_rep_1)
        if rand.random() < self.alpha:
            new_rep_2 = aux.invert_binary(new_rep_2)

        # if image matrix
        # self.set_opinion(new_rep_1, judge=observer, judged=agent1)
        # self.set_opinion(new_rep_2, judge=observer, judged=agent2)
        agent1.set_reputation(new_rep_1)
        agent2.set_reputation(new_rep_2)

        # Count coop acts; coop = 1, def = 0
        cooperative_acts += a1_action
        cooperative_acts += a2_action

        return pd[a1_action, a2_action], cooperative_acts


