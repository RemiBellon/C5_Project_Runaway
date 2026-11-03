"""Run a simulation from a .yaml file: read it, compute, write the results."""

import shutil
import warnings
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ppf_c5_runaways import fields
from ppf_c5_runaways.config import load_config
from ppf_c5_runaways.diagnostics import half_step_position, step_jacobian_determinant
from ppf_c5_runaways.integrate import integrate
from ppf_c5_runaways.io import (
    CONFIG_FILENAME,
    create_result_dir,
    list_results,
    write_result,
)
from ppf_c5_runaways.pushers import get_pusher
from ppf_c5_runaways.radiation import landau_lifshitz_parameter, quantum_parameter

# chi and eta must be << 1 for the classical Landau-Lifshitz force (roadmap D5);
# above this value run warns. The quantum correction to the radiated power is
# about 6 chi at first order, so 1e-2 already means a few percent.
VALIDITY_WARNING_THRESHOLD: Final[float] = 1.0e-2


def _radiation_validity(
    u: NDArray[np.float64],
    e_field: NDArray[np.float64],
    b_field: NDArray[np.float64],
    q: float,
    m: float,
    rr_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``chi`` and ``eta`` at each saved state; warn if one is not small.

    Roadmap D5: the classical force of Landau and Lifshitz is valid while both the
    quantum parameter ``chi`` and its own small parameter ``eta`` (which grows with
    ``rr_scale``) stay well below 1. A run outside this domain is still written,
    since the equations solved are those asked for, but a ``RuntimeWarning`` says
    that its physics is not that of a real electron.

    Returns
    -------
    chi, eta : ndarray, shape (n_saved, n_part)
        See ``radiation.quantum_parameter`` and
        ``radiation.landau_lifshitz_parameter``.
    """
    chi = quantum_parameter(u, e_field, b_field, q, m)
    eta = landau_lifshitz_parameter(u, e_field, b_field, q, m, rr_scale)
    if max(chi.max(), eta.max()) >= VALIDITY_WARNING_THRESHOLD:
        warnings.warn(
            "The Landau-Lifshitz force is used outside its domain of validity: "
            f"max chi = {chi.max():.3e}, max eta = {eta.max():.3e} (both should be "
            f"<< 1; this warning starts at {VALIDITY_WARNING_THRESHOLD}). With "
            "rr_scale > 1, eta grows as rr_scale.",
            RuntimeWarning,
            stacklevel=3,
        )
    return chi, eta


def pusher(
    config_path: str | Path,
    result_dir: str | Path | None = None,
    *,
    base_dir: str | Path = "results",
    phase_space_volume: bool = False,
) -> Path:
    """
    Run the simulation described by a .yaml file and save its trajectories.

    The pusher is that of the ``pusher:`` key, with the radiation reaction force
    multiplied by ``rr_scale:`` (absent or ``0``: no radiation). Both are recorded
    with the case in the results folder.

    With ``phase_space_volume=True``, the Jacobian determinant of every step is also
    measured (``diagnostics.step_jacobian_determinant``) and stored, for
    ``plotting.phase_space_volume``. It needs ``save_every: 1`` (the volume change of
    the run is the product over all the steps) and costs about 12 times the run.

    For the tokamak field of Carbajal et al. (``kind: carbajal``), the poloidal flux
    ``Psi`` is also stored at the saved positions ``x^n`` and at ``x^{n+1/2}``
    (``io.write_result``), so that ``P_zeta`` can be computed from the results
    folder alone, as it is and as a staggered code would measure it.

    With radiation (``rr_scale > 0``), the quantum parameter ``chi`` and the
    validity parameter ``eta`` of the Landau-Lifshitz force are stored at every
    saved state, and a ``RuntimeWarning`` is raised if either reaches
    ``VALIDITY_WARNING_THRESHOLD`` (roadmap D5).
    """
    config = load_config(config_path)
    # Fails first if the pusher does not exist. With rr_scale > 0 (key of the .yaml)
    # the pusher includes the radiation reaction force.
    push = get_pusher(config.pusher, config.rr_scale)
    if phase_space_volume and config.save_every != 1:
        raise ValueError(
            "phase_space_volume=True needs save_every: 1 in the configuration, so "
            f"that every step is measured; got save_every: {config.save_every}"
        )

    # Every check that can fail is done before the computation, which can be long.
    if result_dir is not None:
        result_dir = Path(result_dir)
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Results folder not found: {result_dir}")
        stored = load_config(result_dir / CONFIG_FILENAME)
        ignored = {"pusher", "name"}
        new_case = config.model_dump(exclude=ignored)
        stored_case = stored.model_dump(exclude=ignored)
        different = [key for key in new_case if new_case[key] != stored_case[key]]
        if different:
            raise ValueError(
                f"Cannot add this run to {result_dir}: it does not describe the same "
                f"case as the one stored there (different: {different}). A results "
                "folder holds one physical case, so that pushers are compared "
                "fairly. Only 'pusher' and 'name' may differ."
            )
        if config.pusher in list_results(result_dir):
            raise FileExistsError(
                f"{result_dir} already holds a result for {config.pusher!r}. Results "
                "are never overwritten: run again without result_dir to make a new "
                "folder."
            )
    if config.field.kind == "uniform_cartesian":
        field = fields.uniform(
            b_field=config.field.b_field, e_field=config.field.e_field
        )
    elif config.field.kind == "mirror":
        field = fields.magnetic_mirror(b0=config.field.b0, length=config.field.length)
    elif config.field.kind == "carbajal":
        field = fields.carbajal(
            b0=config.field.b0,
            r0=config.field.r0,
            q0=config.field.q0,
            lam=config.field.lam,
            e0=config.field.e0,
        )
    elif config.field.kind == "higuera_cary_shear":
        field = fields.higuera_cary_shear(a=config.field.a, b=config.field.b)
    else:
        raise ValueError(
            f"The field kind {config.field.kind!r} has no field function yet. "
            "Available: 'uniform_cartesian', 'mirror', 'carbajal', "
            "'higuera_cary_shear'."
        )
    t, x, u = integrate(
        x0=np.array(config.initial.x0),
        u0=np.array(config.initial.u0),
        field=field,
        push=push,
        dt=config.dt,
        n_steps=config.n_steps,
        save_every=config.save_every,
        q=config.q,
        m=config.m,
    )
    # Fields at each saved position and time. B is stored for the diagnostics that
    # need it (magnetic moment, gyration phase): plotting never evaluates a field.
    e_field = np.empty_like(x)
    b_field = np.empty_like(x)
    for k in range(t.size):
        e_field[k], b_field[k] = field(x[k], t[k])
    # Validity of the radiation force at each saved state (roadmap D5).
    chi = None
    eta = None
    if config.rr_scale > 0.0:
        chi, eta = _radiation_validity(
            u, e_field, b_field, config.q, config.m, config.rr_scale
        )
    # Determinant of the step that starts at each saved state (x^n, u^n).
    step_determinant = None
    if phase_space_volume:
        step_determinant = step_jacobian_determinant(
            push, field, t, x, u, config.dt, config.q, config.m
        )

    # Poloidal flux at x^n, for P_zeta, and at x^{n+1/2}, for the error of a
    # diagnostic that pairs x^{n+1/2} with u^n (roadmap C5(c)).
    poloidal_flux = None
    poloidal_flux_half_step = None
    if config.field.kind == "carbajal":
        profile = {
            "b0": config.field.b0,
            "q0": config.field.q0,
            "lam": config.field.lam,
        }
        r = fields.cartesian_to_toroidal(x, config.field.r0)[0]
        poloidal_flux = fields.carbajal_poloidal_flux(r, **profile)
        x_half = half_step_position(x, u, config.dt)
        r_half = fields.cartesian_to_toroidal(x_half, config.field.r0)[0]
        poloidal_flux_half_step = fields.carbajal_poloidal_flux(r_half, **profile)

    created = result_dir is None
    if result_dir is None:
        result_dir = create_result_dir(base_dir, config)
    try:
        write_result(
            result_dir,
            config.pusher,
            t,
            x,
            u,
            config.dt,
            b_field=b_field,
            step_jacobian_determinant=step_determinant,
            poloidal_flux=poloidal_flux,
            poloidal_flux_half_step=poloidal_flux_half_step,
            quantum_parameter=chi,
            landau_lifshitz_parameter=eta,
        )
    except BaseException:
        if created:
            shutil.rmtree(result_dir)  # do not leave an empty results folder
        raise
    return result_dir
