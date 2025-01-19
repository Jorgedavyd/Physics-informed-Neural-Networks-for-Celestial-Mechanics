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
        m1, m2, m3 = masses
        t = Symbol("t")
        q1, q2, q3, q4, q5, q6 = [Symbol(f"q{i}") for i in range(1, 7)]
        p1, p2, p3, p4, p5, p6 = q1.diff(t) * m1, q2.diff(t) * m1, q3.diff(t) * m2, q4.diff(t) * m2, q5.diff(t) * m3, q6.diff(t) * m3
        H = Function()(*[f"q{i}" for i in range(1, 7)])
        self.equations = dict(
            "first": (p1 / 2 * m1) - ...,
            "second": (p1 / 2 * m1) - ...,
        )
        # self.equations = dict(
        #    "x": 1
        # )
