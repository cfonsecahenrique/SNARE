from agent import Agent
import numpy as np

class ModelParameters:

    def __init__(self, sn_str: str, sn_list: list, z: int, mu: float, chi: float, eps: float, runs: int = 50, gens: int = 5000, pdx_strats: bool = True):
        self._social_norm = sn_list
        self._social_norm_str = sn_str
        self._mu = mu
        self._z = z
        self._gens = gens
        self._chi = chi
        self._eps = eps
        self._runs = runs
        self._converge = z*10
        self._paradoxical_strats = pdx_strats

    @property
    def paradoxical_strats(self):
        return self._paradoxical_strats

    def generate_mp_string(self) -> str:
        builder: str = self._social_norm_str + "\t" + str(self.z) + "\t" + str(self._gens) + "\t" + str(self.mu) + "\t" \
                       + str(self.chi) + "\t" + str(self.eps) + "\t" + str(self._paradoxical_strats)
        return builder

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