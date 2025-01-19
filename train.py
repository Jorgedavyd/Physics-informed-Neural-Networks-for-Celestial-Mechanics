from typing import List
import modulus.sym
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from equations import HamiltonianBodies
from modulus.sym.key import Key
from modulus.sym.geometry.primitives_2d import Rectangle
from modulus.sym.hydra import instantiate_arch, ModulusConfig

def make_geometry():
    return Rectangle()


@modulus.sym.main(version_base="1.3", config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    m: List[float] = None
    eq = HamiltonianBodies(*m)
    h_net = instantiate_arch(
        input_keys=[Key(f"q{i}") for i in range(1, 7)] + [Key(f"p{i}") for i in range(1, 7)] + [Key("t")],
        output_keys=[Key("H")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = eq.make_nodes() + [h_net.make_node(name="model")]
    geo = make_geometry()
    domain = Domain()

    boundary = PointwiseBoundaryConstraint(
        nodes=nodes, geometry=geo, outvar={"H": 0}, batch_size=cfg.batch_size.boundary
    )

    q_dot = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"q_dot": 0},
        batch_size=cfg.batch_size.q_dot,
        bounds={
        },
    )

    p_dot = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p_dot": 0},
        batch_size=cfg.batch_size.p_dot,
        bounds={
        },
    )

    domain.add_constraint(boundary, "boundary")
    domain.add_constraint(p_dot, "p_dot_hamiltonian")
    domain.add_constraint(q_dot, "q_dot_hamiltonian")

    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()
