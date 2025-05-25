from agent import Agent
import numpy as np


class ModelParameters:

    def __init__(self, sn_str: str, sn_list: list, ebsn_str: str, ebsn_list: list, z: int,
                 mu: float, chi: float, eps: float, alpha: float, min_gamma: float, max_gamma: float,
                 gens: int, b: int, c: int, beta: float, convergence: float):
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
        self._converge: int = int(convergence * z)
        self._b: int = b
        self._c: int = c
        self._min_gamma: float = min_gamma
        self._max_gamma: float = max_gamma
        self._beta: float = beta

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
    def gamma(self):
        return self._gamma

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
    def runs(self):
        return self._runs

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

    def __str__(self) -> str:
        return (
            "\nModel Parameters:\n"
            "---------------------------------------------\n"
            f"{'Population size (Z):':<30} {self._z:<20} {'Generations (G):':<30} {self._gens}\n"
            f"{'Convergence time:':<30} {self._converge:<20} {'Strategy exploration (μ):':<30} {self._mu:.6f}\n"
            f"{'Reputation error (χ):':<30} {self._chi:.6f} {'Execution error (ε):':<30} {self._eps:.6f}\n"
            f"{'Judge assignment error (α):':<30} {self._alpha:.6f} {'Selection strength (β):':<30} {self._beta:.6f}\n"
            f"{'Benefit (b):':<30} {self._b:<20} {'Cost (c):':<30} {self._c}\n"
            f"{'Gamma range:':<30} [{self._min_gamma:.3f}, {self._max_gamma:.3f}]  {'':<30} \n"
            f"{'Social Norm (string):':<30} {self._social_norm_str:<20} {'EB Social Norm (string):':<30} {self._ebsn_str}\n"
        )
