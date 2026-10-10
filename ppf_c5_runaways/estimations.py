"""
Calculate quickly some estimations for the simulation, such as the Larmor radius and the gyrofrequency.
"""

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

def v_perpendicular(e_field: float, b_field: float) -> float:
    """
    Calculate the perpendicular velocity of a particle in crossed electric and magnetic fields.

    Parameters
    ----------
    e_field : float
        Electric field strength [V/m].
    b_field : float
        Magnetic field strength [T].

    Returns
    -------
    float
        Perpendicular velocity [m/s].
    """
    return e_field / b_field

print("Larmor radius:", f"{larmor_radius(b_field=0.1, e_field=10**8, q=1.602e-19, m=9.109e-31)*1000:.2f}", "mm")