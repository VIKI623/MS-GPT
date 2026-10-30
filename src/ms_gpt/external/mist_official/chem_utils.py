"""Feature constants for MIST official checkpoints."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

VALID_ELEMENTS = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]
NORM_VEC = np.asarray(
    [81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6]
)

ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]

instrument_to_type = defaultdict(lambda: "unknown")
instrument_to_type.update(
    {
        "Orbitrap": "orbitrap",
        "QTOF": "qtof",
        "Unknown": "unknown",
        "Thermo Finnigan Velos Orbitrap": "orbitrap",
        "Thermo Finnigan Elite Orbitrap": "orbitrap",
        "Orbitrap Fusion Lumos": "orbitrap",
        "Q-ToF (LCMS)": "qtof",
        "Unknown (LCMS)": "unknown",
        "ion trap": "iontrap",
        "FTICR (LCMS)": "fticr",
        "Bruker Q-ToF (LCMS)": "qtof",
        "Orbitrap (LCMS)": "orbitrap",
    }
)
instruments = sorted(set(instrument_to_type.values()))
max_instr_idx = len(instruments) + 1
