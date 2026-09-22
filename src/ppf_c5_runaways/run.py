"""Run a simulation from a .yaml file: read it, compute, write the results."""

import shutil
from pathlib import Path

import numpy as np

from ppf_c5_runaways import fields
from ppf_c5_runaways.config import load_config
from ppf_c5_runaways.integrate import integrate
from ppf_c5_runaways.io import (
    CONFIG_FILENAME,
    create_result_dir,
    list_results,
    write_result,
)
from ppf_c5_runaways.pushers import get_pusher


def pusher(
    config_path: str | Path,
    result_dir: str | Path | None = None,
    *,
    base_dir: str | Path = "results",
) -> Path:
    """
    Run the simulation described by a .yaml file and save its trajectories.
    """
    config = load_config(config_path)
    push = get_pusher(config.pusher)  # fails first if the pusher does not exist

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

    field = fields.uniform(b_field=config.field.b_field, e_field=config.field.e_field)
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

    created = result_dir is None
    if result_dir is None:
        result_dir = create_result_dir(base_dir, config)
    try:
        write_result(result_dir, config.pusher, t, x, u, config.dt)
    except BaseException:
        if created:
            shutil.rmtree(result_dir)  # do not leave an empty results folder
        raise
    return result_dir
