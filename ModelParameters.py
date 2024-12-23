from agent import Agent
import numpy as np

class ModelParameters:

    def __init__(self, sn_str: str, sn_list: list, ebsn_str: str, ebsn_list: list, z: int,
                 mu: float, chi: float, eps: float, alpha: float, gamma: float,
                 runs: int = 50, gens: int = 1000, pdx_strats: bool = True, b: int = 5, c: int = 1):
        self._social_norm = sn_list
        self._social_norm_str = sn_str
        self._eb_social_norm = ebsn_list
        self._ebsn_str = ebsn_str
        self._mu = mu
        self._z = z
        self._gens = gens
        self._chi = chi
        self._eps = eps
        self._alpha = alpha
        self._gamma = gamma
        self._runs = runs
        self._converge = z*10
        self._paradoxical_strats = pdx_strats
        self._b = b
        self._c = c

    @property
    def paradoxical_strats(self):
        return self._paradoxical_strats

    def generate_mp_string(self) -> str:
        # for results exporting
        builder: str = self._ebsn_str + "\t" + self._social_norm_str + "\t" + str(self._z) + "\t" + str(self._gens) \
                       + "\t" + str(self._mu) + "\t" + str(self._chi) + "\t" + str(self._eps) + "\t" + str(self._alpha) \
                       + "\t" + str(self._gamma) + "\t" + str(self._paradoxical_strats) + "\t" + str(self._b) + "\t" + str(self._c)
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
