
import ruamel.yaml


def new_config(name: str, b_field: list, e_field: list, 
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

    d = {'name':name, 'pusher': pusher, 
          'n_steps': steps, 'dt': dt, 
          'save_every': save_every, 'seed': seed, 
          'q': q, 'm': m, 
          'field': {'kind': 'uniform', 'b_field': b_field, 'e_field': e_field},
          'initial': {'x0': x0, 'u0': u0}}
    with open(f'configs/{name}.yaml', 'w') as yaml_file:
        yaml = ruamel.yaml.YAML()
        yaml.default_flow_style = None  # this makes the leaf nodes flow style
        yaml.width = 2048  # to prevent line wrapping
        yaml.dump(d, yaml_file)

    return name
