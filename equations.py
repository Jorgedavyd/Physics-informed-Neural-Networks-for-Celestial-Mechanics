from sympy import Symbol, Function, Number, diff
from modulus.eq.pdes import PDE
from typing import List, Dict
from scipy.constants import G


def generalPhase(mass: List[Number], dimensions: int = 2) -> Dict[str, List[Symbol]]:
    N: int = len(mass)
    q: List[Symbol] = [Symbol(f"q{i}") for i in range(dimensions * N)]
    return {
        "q": q,
        "p": [Symbol(f"p{i}") for i in range(dimensions * N)],
    }


def createMasses(masses: List[float]) -> List[Number]:
    return [Number(m) for m in masses]


class FreeBodies(PDE):
    def __init__(self, masses: List[float], dimensions: int = 2) -> None:
        super().__init__()
        assert dimensions <= 3, "Invalid dimension value (must be ≤ 3)"
        t = Symbol("t")
        mass: List[Number] = createMasses(masses)
        input: Dict[str, List[Symbol]] = generalPhase(mass, dimensions)
        q: List[Symbol] = input["q"]
        p: List[Symbol] = input["p"]
        kinetic_energy = sum([pi**2 / (2 * mi) for pi, mi in zip(p, mass)])
        potential_energy = sum(
            [
                G * mi * mj / abs(qi - qj)
                for i, qi in enumerate(q)
                for j, qj in enumerate(q)
                if i != j
            ]
        )
        H = kinetic_energy + potential_energy
        self.equations = {}
        for i, (qi, pi) in enumerate(zip(q, p)):
            self.equations[f"q{i}_dot"] = diff(H, pi) - qi.diff(t)
            self.equations[f"p{i}_dot"] = -diff(H, qi) - pi.diff(t)


class Euler(PDE):
    def __init__(self, masses: List[float], dimensions: int = 2) -> None:
        super().__init__()
        assert dimensions <= 3, "Invalid dimension value (must be ≤ 3)"
        t = Symbol("t")
        mass: List[Number] = createMasses(masses)
        input: Dict[str, List[Symbol]] = generalPhase(mass, dimensions)
        q: List[Symbol] = input["q"]
        p: List[Symbol] = input["p"]
        kinetic_energy = sum([pi**2 / (2 * mi) for pi, mi in zip(p, mass)])
        potential_energy = sum(
            [
                G * mi * mj / abs(qi - qj)
                for i, qi in enumerate(q)
                for j, qj in enumerate(q)
                if i != j
            ]
        )
        H = kinetic_energy + potential_energy
        self.equations = {}
        for i, (qi, pi) in enumerate(zip(q, p)):
            self.equations[f"q{i}_dot"] = diff(H, pi) - qi.diff(t)
            self.equations[f"p{i}_dot"] = -diff(H, qi) - pi.diff(t)


class Lagrange(PDE):
    def __init__(self, masses: List[float], dimensions: int = 2) -> None:
        super().__init__()
        assert dimensions <= 3, "Invalid dimension value (must be ≤ 3)"
        t = Symbol("t")
        mass: List[Number] = createMasses(masses)
        input: Dict[str, List[Symbol]] = generalPhase(mass, dimensions)
        q: List[Symbol] = input["q"]
        p: List[Symbol] = input["p"]
        kinetic_energy = sum([pi**2 / (2 * mi) for pi, mi in zip(p, mass)])
        potential_energy = sum(
            [
                G * mi * mj / abs(qi - qj)
                for i, qi in enumerate(q)
                for j, qj in enumerate(q)
                if i != j
            ]
        )
        H = kinetic_energy + potential_energy
        self.equations = {}
        for i, (qi, pi) in enumerate(zip(q, p)):
            self.equations[f"q{i}_dot"] = diff(H, pi) - qi.diff(t)
            self.equations[f"p{i}_dot"] = -diff(H, qi) - pi.diff(t)
