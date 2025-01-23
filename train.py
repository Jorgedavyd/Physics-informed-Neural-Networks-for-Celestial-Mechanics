from typing import List, Dict
import modulus.sym
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from equations import Hamiltonian, generalPhase
from modulus.sym.key import Key
from sympy import Symbol
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.hydra import instantiate_arch, ModulusConfig
from itertools import chain


def make_geometry():
    return Rectangle(point_1=(-1, 1), point_2=(1, -1))


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    t: Symbol = Symbol("t")
    input: Dict[str, List[Symbol]] = generalPhase(
        cfg.custom.masses, cfg.custom.dimensions
    )
    eq = Hamiltonian(cfg.custom.masses, cfg.custom.dimensions)
    input_keys: List[Key] = list(
        map(lambda key: Key(key), chain.from_iterable(input.values()))
    )
    h_net = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("H")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = eq.make_nodes() + [h_net.make_node(name="model")]
    geo = make_geometry()
    domain = Domain()

    initial = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"q": cfg.custom.q0, "p": cfg.custom.p0},
        batch_size=cfg.batch_size.boundary,
        parameterization={t: 0},
    )

    boundary = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo, outvar={"H": 0}, batch_size=cfg.batch_size.boundary
    )

    q_dot = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"q_dot": 0},  ## revisar
        batch_size=cfg.batch_size.q_dot,
        bounds={},
    )

    p_dot = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p_dot": 0},  ## revisar
        batch_size=cfg.batch_size.p_dot,
        bounds={},
    )

    domain.add_constraint(initial, "initial")
    domain.add_constraint(boundary, "boundary")
    domain.add_constraint(p_dot, "p_dot_hamiltonian")
    domain.add_constraint(q_dot, "q_dot_hamiltonian")

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
