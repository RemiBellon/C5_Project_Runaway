# ppf_c5_runaways
Relativistic electrons (*runaways*) integration in tokamak plasma.

The code will test **classic Boris**, **relativistic Vay** and **Higuera-Cary** pushers to compare with **RK4** (as seen in dedicated literature).

Further objectives will consist in add synchrotron radiations effects then if time allows it collisons (Monte-carlo).

The code mostly consists in tests, comparison and validation of numerical and physical results in different physical cases.

> For now the project is named `ppf_c5_runaways`. To replace it :
> `src/ppf_c5_runaways/`, `packages = ["src/ppf_c5_runaways"]` et `name = "ppf_c5_runaways"` dans `pyproject.toml`.

---

## 1. Installation

Project managed with [**uv**](https://docs.astral.sh/uv/). No `pip`, `venv` or `conda`.

### 1.1 Install uv
```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 Install Project

```bash
git clone https://github.com/RemiBellon/C5_Project_Runaway.git
cd ppf_c5_runaways

uv python install 3.14          # install python version
uv sync --dev --python 3.14     # create .venv/ and install full kernel + dev tools (mostly manage by Claude)
```

`uv sync` read `pyproject.toml`, write `uv.lock` and install `ppf_c5_runaways` (that is writable)
(rmk for commit: don't forget to commit 'uv.lock' for github project)

### 1.3 Verification

```bash
uv run python -c "import ppf_c5_runaways; print(ppf_c5_runaways.__version__)"
uv run pytest -m "not slow"
```

### 1.4 Extras optional

| Extra | Commande | Purposes|
| --- | --- | --- |
| `viz` | `uv sync --dev --extra viz` | `plotly` : interactives 3D orbites for notebook. |
| `plasma` | `uv sync --dev --extra plasma` | `plasmapy` : reference plasma formulary, it pulls `astropy`, so heavy package. |

### 1.5 Plateform notes

- **macOS (Apple Silicon)** : main targeted plateform.
- **Linux x86-64** : idem.
- `h5netcdf` + `h5py` sont préférés à `netCDF4` parce que leurs wheels sont fiables sur
  macOS arm64.
- Le fichier `pyproject.toml` constraints to use `uv` to these two plateformes (`[tool.uv] environments`).

---

## 2. Use
Single simulation or a group of simulations are describe by a `.yaml` ;
The user decide what simulation run and save and plot in `main.py`.

### 2.1 Config files
... (add an example of config file main fields)

### 2.2 Example of main

```python
# main.py
from ppf_c5_runaways import run, plotting

result_dir = run.pusher("configs/uniform_b.yaml")

plotting.gamma_vs_time(result_dir)
plotting.orbit_xy(result_dir)
plotting.energy_error(result_dir, reference="analytic")
```

(In the terminal:)
```bash
uv run python main.py
```

### 2.3 Results foled

Every main.py run create a new file (which name is based on date time).
```
results/
└── 20260920_143205_uniform_b_relativistic/
    ├── config.yaml        exact copy of sued config file
    ├── metadata.json      versions, plateforme, seed RNG, hash git, time
    ├── trajectory.nc      simulation data xarray (netCDF) : x, u, gamma, t
    └── figures/           generated figures from the data
```
---

## 3. Code architecture
```
src/ppf_c5_runaways/
├── constants.py     Physical constants SI
├── config.py        pydantic scheme for .yaml (ensure config file will be readable by the code)
├── fields.py        Analytics E(x,t), B(x,t)
├── pushers.py       boris / vay / higuera_cary / rk4
├── forces.py        collisions, radiations (???)
├── integrate.py     pure time loop
├── diagnostics.py   energy, invariants, errors
├── io.py            folder horodates, xarray -> netCDF
├── run.py           gather all process to run simulations from config files
└── plotting.py      generates figure from results folder
```

**Dependencies** : `fields`, `pushers` and `integrate` don't read config or run files.

**Convention** : variable = relativistic velocity `u = gamma * v` in (m/s)
with `gamma = sqrt(1 + |u|^2 / c^2)`, never directly use `v`. Positions live at integer steps
, and velocities live at half-step (*leapfrog*).

---

## 4. Tests (Managed by Claude)
```bash
uv run pytest -m "not slow"            # quick computation
uv run pytest                          # full synthaxes tests
```

Before every commit:
```bash
uv run ruff check --fix . && uv run ruff format . && uv run pytest -m "not slow"
```

---

## 5. Still in development
1. Classic and relativistic pusher implementation
2. Comparison and validation vs literature for physical invariants, space-phase volume preservation, etc..
3. Configuration tokamak : non-sumetrical fields, trapped ortbites, conservation of toroidal canonical moment
4. Collisions and synchrotron radiations ; comparison to literature.

---

## 6. Main References
- J. P. Boris, *Relativistic plasma simulation — optimization of a hybrid code*, Proc. 4th
  Conf. Num. Sim. Plasmas (1970).
- J.-L. Vay, *Simulation of beams or plasmas crossing at relativistic velocity*,
  Phys. Plasmas **15**, 056701 (2008).
- A. V. Higuera & J. R. Cary, *Structure-preserving second-order integration of relativistic
  charged particle trajectories*, Phys. Plasmas **24**, 052104 (2017).
- S. Zenitani & T. Umeda, *On the Boris solver in particle-in-cell simulation*,
  Phys. Plasmas **25**, 112110 (2018).
- B. Ripperda *et al.*, *A comprehensive comparison of relativistic particle integrators*,
  ApJS **235**, 21 (2018).
- B. N. Breizman *et al.*, *Physics of runaway electrons in tokamaks*, Nucl. Fusion **59**,
  083001 (2019).

---

## 7. Licence

MIT. see `LICENSE`.
