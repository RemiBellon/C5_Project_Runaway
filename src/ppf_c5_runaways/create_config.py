
import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ppf_c5_runaways.constants import *
import numpy as np
from ppf_c5_runaways.fields import custom_toroidal


def new_config(name: str, kind: str ='tokamak',  b_field: list=[5.3, 6.2, 4, 1.6, "0", "b0 * r * q0 * (1 + r**2/lamb**2) * 1 / r0*(1 + (r/r0)*cos(theta))","b0 * 1 / (1 + (r/r0)*cos(theta))"], e_field: list=None,
                  steps: int = 1000, pusher: str='boris',
                  dt: float = 1e-11, save_every: int = 5, seed: int = 42, 
                  q: float = -1.602176634e-19, m: float = 9.1093837139e-31, 
                  x0: list = [[0, 0, 0],[0, 0, 0]], u0: list = [[4.0e8, 0, 4.0e8],[4.0e8, 0, 4.0e8]]) -> None:

    """Create a YAML configuration file for the simulation in complex cases.
    Most arguments have default values except for name, b_field and e_field.
    Default values are set for a single electron with initial velocity in x and z.

    x0 and u0 are lists of lists, one row per particle. 
    !! due to conflict between YAML notation and list of list notation, this function only
    works for 2 or more particles. (for now?)
        """

    if kind == 'tokamak':
        b0, r0, lamb, q0, b_r, b_theta, b_phi= b_field
        field_dict = CommentedMap({
            'kind': 'tokamak', 'b0': b0, 'r0': r0, 'lamb': lamb, 'q0': q0,
            'b_r': b_r, 'b_theta': b_theta, 'b_phi': b_phi,
        })
        field_dict.fa.set_block_style()
    elif kind == 'uniform':
        field_dict = CommentedMap({
            'kind': 'uniform', 'b_field': b_field, 'e_field': e_field,
        })
        field_dict.fa.set_block_style()

    def flow_list(value):
        sequence = CommentedSeq(
            flow_list(item) if isinstance(item, (list, tuple)) else item
            for item in value
        )
        sequence.fa.set_flow_style()
        return sequence

    d = {'name':name, 'pusher': pusher, 
          'n_steps': steps, 'dt': dt, 
          'save_every': save_every, 'seed': seed, 
          'q': q, 'm': m, 
          'field': field_dict,
          'initial': {'x0': flow_list(x0), 'u0': flow_list(u0)}}
    with open(f'configs/{name}.yaml', 'w') as yaml_file:
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = None  # this makes the leaf nodes flow style
        yaml.width = 2048  # to prevent line wrapping
        yaml.dump(d, yaml_file)

    return name


def u_from_theta(energy, theta, m=m_e,
                 b_field = [5.3, 6.2, 4, 1.6, "0", "b0 * r * q0 * (1 + r**2/lamb**2) * 1 / r0*(1 + (r/r0)*cos(theta))","b0 * 1 / (1 + (r/r0)*cos(theta))"],
                  x0=[[1.0, 7.2, 0.0]],
                  alpha = 0): ###is the energy given in the paper total or kinetic energy?
    """
    total energy in MeV
    theta in degrees
    m in kg
    tokamak b field parameters: b0, r0, lamb, q0, b_r, b_theta, b_phi
    initial position x0 in m
    alpha in degrees, angle of the perpendicular velocity in the plane perpendicular to B

    gives proper u (u=gamma * v) in scientific units in cartesian coordinates

    """

    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim != 2 or x0.shape[1] != 3 or x0.shape[0] == 0:
        raise ValueError(f"x0 must have shape (n_particles, 3), got {x0.shape}")
    if not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain only finite values")

    n_particles = x0.shape[0]

    def per_particle_values(value, name):
        values = np.asarray(value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(n_particles, values.item())
        if values.shape != (n_particles,):
            raise ValueError(
                f"{name} must be a scalar or have one value per particle "
                f"({n_particles}), got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values

    energy = per_particle_values(energy, "energy") * MeV_inJ
    theta = np.deg2rad(per_particle_values(theta, "theta"))
    alpha = np.deg2rad(per_particle_values(alpha, "alpha"))
    rest_energy = m * c**2
    if np.any(energy <= rest_energy):
        raise ValueError("energy must be greater than the particle rest energy")
    u_proper = c * np.sqrt((energy / rest_energy) ** 2 - 1)
    u_parallel = u_proper * np.cos(theta)
    u_perp = u_proper * np.sin(theta)

    field = custom_toroidal(
        b0=b_field[0], r0=b_field[1], q0=b_field[3], lamb=b_field[2],
        b_r=b_field[4],
        b_theta=b_field[5],
        b_phi=b_field[6],
    )

    _, B = field(x0, 0.0)
    b_magnitude = np.linalg.norm(B, axis=1)
    if np.any(b_magnitude == 0.0):
        raise ValueError("magnetic field must be non-zero at every initial position")
    b_hat = B / b_magnitude[:, np.newaxis]

    reference = np.tile([0.0, 0.0, 1.0], (n_particles, 1))
    use_x_reference = np.abs(b_hat[:, 2]) > 0.9
    reference[use_x_reference] = [1.0, 0.0, 0.0]

    e1 = np.cross(reference, b_hat)
    e1 /= np.linalg.norm(e1, axis=1)[:, np.newaxis]
    e2 = np.cross(b_hat, e1)

    u0_xyz = (
        u_parallel[:, np.newaxis] * b_hat
        + u_perp[:, np.newaxis]
        * (
            np.cos(alpha)[:, np.newaxis] * e1
            + np.sin(alpha)[:, np.newaxis] * e2
        )
    )
    return u0_xyz.tolist()


#DO THIS IN MAIN/
##how to create a new config file for a tokamak with custom parameters
"""
x_init = [[1, 7.2, 0], [1, 7.2, 0]]  # initial positions for two particles

b_field_tokamak = [5.3, 6.2, 4, 1.6, "0", "b0 * r * q0 * (1 + r**2/lamb**2) * 1 / r0*(1 + (r/r0)*cos(theta))","b0 * 1 / (1 + (r/r0)*cos(theta))"]
theta = [57, 80]  # angle in degrees
energies = [40, 40]  # energy in MeV
u_init = u_from_theta(
    energy=energies, theta=theta, m=m_e,
    x0=x_init, alpha=0, b_field=b_field_tokamak
)
print(f"Initial velocity u0: {u_init}")

tokamak_theta57_80= new_config(name="tokamak_theta57_80", kind="tokamak", b_field=b_field_tokamak, e_field=None, steps=10000, pusher='boris', dt=1e-6, save_every=5, seed=42, q=-1.602176634e-19, m=9.1093837139e-31, x0=x_init, u0=u_init)
"""