"""
Calculate quickly some estimations for the simulation, such as the Larmor radius and the gyrofrequency.
"""

import numpy as np


def larmor_radius(b_field: float, e_field: float, q: float, m: float) -> float:
    """
    Calculate the Larmor radius of a particle in a magnetic field.

    Parameters
    ----------
    b_field : float
        Magnetic field strength [T].
    e_field : float
        Electric field strength [V/m].
    q : float
        Particle charge [C].
    m : float
        Particle mass [kg].

    Returns
    -------
    float
        Larmor radius [m].
    """
    return m * e_field / (q * b_field**2)

def v_perpendicular(v: np.array, b_field: np.array) -> float:
    """
    Calculate the perpendicular velocity of a particle in crossed electric and magnetic fields.

    Parameters
    ----------
    v : np.array
        Particle velocity vector [m/s].
    b_field : np.array
        Magnetic field vector [T].

    Returns
    -------
    float
        Perpendicular velocity [m/s].
    """
    return np.linalg.norm(v - np.dot(v, b_field) / np.linalg.norm(b_field)**2 * b_field)


def get_mu(B:np.array, v:np.array, m:float=9.109e-31) -> float:
    """
    Calculate the magnetic moment of a particle in a magnetic field.

    Parameters
    ----------
    B : np.array
        Magnetic field vector [T].
    v : np.array
        Particle velocity vector [m/s].
    m : float
        Particle mass [kg].

    Returns
    -------
    float
        Magnetic moment [A·m²].
    """
    v_perp = v_perpendicular(B, v)
    return 0.5 * m * v_perp**2 / np.linalg.norm(B)


def magnetic_energy(B:np.array, v:np.array, m:float=9.109e-31) -> float:
    """
    Calculate the magnetic energy of a particle in a magnetic field.

    Parameters
    ----------
    B : np.array
        Magnetic field vector [T].
    m : float
        Particle mass [kg].
    v : np.array
        Particle velocity vector [m/s].

    Returns
    -------
    float
        Magnetic energy [J].
    """
    mu = get_mu(B, v, m)
    return mu * np.linalg.norm(B)

def electric_energy(e_field:np.array, v:np.array, q:float=-1.602e-19) -> float:
    """
    Calculate the electric energy of a particle in an electric field.

    Parameters
    ----------
    q : float
        Particle charge [C].
    e_field : np.array
        Electric field vector [V/m].
    v : np.array
        Particle velocity vector [m/s].

    Returns
    -------
    float
        Electric energy [J].
    """
    return q * np.dot(e_field, v)


def kinetic_energy(v:np.array, m:float=9.109e-31) -> float:
    """
    Calculate the kinetic energy of a particle.

    Parameters
    ----------
    m : float
        Particle mass [kg].
    v : np.array
        Particle velocity vector [m/s].

    Returns
    -------
    float
        Kinetic energy [J].
    """
    return 0.5 * m * np.linalg.norm(v)**2

def tot_energy(B:np.array, e_field:np.array, v:np.array, m:float=9.109e-31, q:float=-1.602e-19) -> float:
    """
    Calculate the total energy of a particle in electric and magnetic fields.

    Parameters
    ----------
    B : np.array
        Magnetic field vector [T].
    e_field : np.array
        Electric field vector [V/m].
    m : float
        Particle mass [kg].
    q : float
        Particle charge [C].
    v : np.array
        Particle velocity vector [m/s].

    Returns
    -------
    float
        Total energy [J].
    """
    return kinetic_energy(v, m) + magnetic_energy(B, v, m) + electric_energy(e_field, v, q)

#print("Larmor radius:", f"{larmor_radius(b_field=0.1, e_field=10**8, q=1.602e-19, m=9.109e-31)*1000:.2f}", "mm")