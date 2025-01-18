from sympy import Symbol, Function
from modulus.sym.geometry.primitives import VectorField
from modulus.eq.pdes import PDE
from modulus.key import Key
from typing import List
from scipy.constants import G


class HamiltonianBodies(PDE):
    def __init__(self, masses: List[float]) -> None:
        super().__init__()
        assert len(masses) == 3, "3 masses only"
        q1, q2, q3, q4, q5, q6 = [Symbol(f"q{i}") for i in range(1, 7)]
        m1, m2, m3 = masses
        t = Symbol("t")
        # self.equations = dict(
        #    "x": 1
        # )
