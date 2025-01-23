from sympy import Symbol, Function, Number
from modulus.sym.geometry.primitives import VectorField
from modulus.eq.pdes import PDE
from modulus.key import Key
from typing import List, Dict
from scipy.constants import G
from itertools import chain


def generalPhase(mass: List[Number], dimensions: int = 2) -> Dict[str, List[Symbol]]:
    N: int = len(mass)
    t: Symbol = Symbol("t")
    q: List[Symbol] = [Symbol(f"q{i}") for i in range(1, dimensions * N)]
    new_mass: List[Number] = list(
        chain.from_iterable(map(lambda mass: [mass] * dimensions))
    )
    return dict(q=q, p=[qi.diff(t) * mi for qi, mi in zip(q, new_mass)])


def createMasses(masses: List[float]) -> List[Number]:
    return list(map(lambda x: Number(x), masses))


class Hamiltonian(PDE):
    def __init__(self, masses: List[float], dimensions: int = 2) -> None:
        super().__init__()
        assert dimensions <= 3, "Not valid dimension value"
        t = Symbol("t")
        mass: List[Number] = createMasses(masses)
        input: Dict[str, List[Symbol]] = generalPhase(mass)
        q: List[Symbol] = input["q"]
        p: List[Symbol] = input["p"]
        H = Function()(*q, *p)
        self.equations = dict()
