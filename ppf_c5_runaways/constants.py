"""Physical constants in SI units."""

import math
from typing import Final

c: Final[float] = 299_792_458.0
e: Final[float] = 1.602176634e-19
m_e: Final[float] = 9.1093837139e-31
m_p: Final[float] = 1.67262192595e-27
eps_0: Final[float] = 8.8541878188e-12
mu_0: Final[float] = 1.25663706127e-6
h: Final[float] = 6.62607015e-34  # Planck constant [J s], exact since the 2019 SI
hbar: Final[float] = h / (2.0 * math.pi)  # reduced Planck constant [J s]
