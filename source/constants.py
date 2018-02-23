################################################################################
# Dictionaries with constants from the periodic table are defined here
################################################################################


# Dictionary to convert atomic symbols to atomic numbers
SymbolToNumber = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
    "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38,
    "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59,
    "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
    "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87,
    "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94,
    "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101,
    "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107,
    "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113,
    "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118}

# Invert the above: atomic numbers to atomic symbols
NumberToSymbol = {v: k for k, v in SymbolToNumber.items()}

# Dictionary to convert atomic symbols to atomic masses
# Specifically, to atomic mass units ("amu", more precisely "u" or "Da", i.e. Dalton)
SymbolToMass = {
    "H": 1.00794, "He": 4.002602, "Li": 6.941, "Be": 9.012182, "B": 10.811,
    "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984032, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.3050, "Al": 26.9815386, "Si": 28.0855,
    "P": 30.973762, "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.0983,
    "Ca": 40.078, "Sc": 44.955912, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.938045, "Fe": 55.845, "Co": 58.933195, "Ni": 58.6934, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.64, "As": 74.92160, "Se": 78.96,
    "Br": 79.904, "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90585,
    "Zr": 91.224, "Nb": 92.90638, "Mo": 95.96, "Tc": 98.0, "Ru": 101.07,
    "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.411, "In": 114.818,
    "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293,
    "Cs": 132.9054519, "Ba": 137.327, "La": 138.90547, "Ce": 140.116,
    "Pr": 140.90765, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.92535, "Dy": 162.500, "Ho": 164.93032, "Er": 167.259,
    "Tm": 168.93421, "Yb": 173.054, "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788,
    "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.966569, "Hg": 200.59, "Tl": 204.3833, "Pb": 207.2, "Bi": 208.98040,
    "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0,
    "Th": 232.03806, "Pa": 231.03588, "U": 238.02891, "Np": 237.0, "Pu": 244.0,
    "Am": 243.0, "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0,
    "Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 267.0, "Db": 268.0, "Sg": 271.0,
    "Bh": 272.0, "Hs": 270.0, "Mt": 276.0, "Ds": 281.0, "Rg": 280.0, "Cn": 285.0,
    "Nh": 284.0, "Fl": 289.0, "Mc": 288.0, "Lv": 293.0, "Ts": 294.0, "Og": 291.0}

# Dictionary to convert atomic symbols to atomic masses
# Specifically, to masses in atomic units (a.u. or m_e)
SymbolToAUMass = {
    "I": 231332.6972092981,
    "Er": 304894.5053119877,
    "Ba": 250331.80714328878,
    "W": 335119.8193015373,
    "Sn": 216395.0921958523,
    "He": 7296.297100609073,
    "Mc": 524991.8840232961,
    "Pa": 421152.64554923656,
    "La": 253209.1819320883,
    "Sc": 81949.61437106077,
    "Os": 346768.07672830415,
    "Fl": 526814.772509488,
    "Tc": 178643.071646816,
    "Fr": 406504.132420816,
    "Ne": 36785.3427848087,
    "Bh": 495825.66824422404,
    "Hf": 325367.3659004101,
    "Ta": 329847.80705285165,
    "Ts": 535929.2149404481,
    "F": 34631.970449313245,
    "Tl": 372567.9643399254,
    "Cr": 94783.09201688785,
    "Mo": 174924.3791349843,
    "Db": 488534.114299456,
    "Lu": 318944.9651858584,
    "Th": 422979.5079323285,
    "Pm": 264318.83049784,
    "Eu": 277013.4259156811,
    "Md": 470305.22943753604,
    "Rf": 486711.225813264,
    "Fe": 101799.20751139225,
    "Pb": 377702.4943389824,
    "Kr": 152754.40936591724,
    "Gd": 286649.214453692,
    "Be": 16428.20280326679,
    "Ca": 73057.72474960299,
    "Zn": 119180.44922723295,
    "Nb": 169357.9703957787,
    "Ra": 411972.797879392,
    "Np": 432024.571227504,
    "Am": 442961.902144656,
    "Cm": 450253.456089424,
    "Pu": 444784.79063084803,
    "N": 25532.652159545487,
    "Si": 51196.73457894542,
    "Yb": 315458.1440894704,
    "Te": 232600.5708380992,
    "C": 21894.166741106255,
    "Nh": 517700.33007852803,
    "Ce": 255415.84313127832,
    "V": 92860.67381934977,
    "Tm": 307948.22633294144,
    "Mn": 100145.92968439798,
    "Lv": 534106.3264542561,
    "Hg": 365653.2014452533,
    "Fm": 468482.340951344,
    "Sr": 159721.48916014304,
    "U": 433900.1594198318,
    "Ni": 106991.52307546153,
    "No": 472128.117923728,
    "Re": 339434.5963483537,
    "Bi": 380947.9649997987,
    "Mg": 44305.30465689656,
    "In": 209300.41020759306,
    "B": 19707.247424221714,
    "Cn": 519523.21856472,
    "Es": 459367.89852038404,
    "K": 71271.84089968068,
    "Cf": 457545.010034192,
    "Al": 49184.336053685016,
    "Rg": 510408.77613376,
    "Nd": 262937.0810253065,
    "Ac": 413795.686365584,
    "Au": 359048.0907948421,
    "Y": 162065.45032011304,
    "Ho": 300649.5813519621,
    "Ti": 87256.20316855246,
    "Br": 145656.08160068557,
    "Pd": 193991.79270055264,
    "Cd": 204912.71762132892,
    "Lr": 477596.783382304,
    "Sb": 221954.90207873794,
    "Ge": 132414.6196369869,
    "Mt": 503117.222188992,
    "P": 56461.7141238513,
    "Ar": 72820.74924639802,
    "Ru": 184239.33929942542,
    "As": 136573.72200708254,
    "Cs": 242271.8180206547,
    "Rh": 187585.25111583088,
    "Rn": 404681.243934624,
    "Tb": 289703.1906790338,
    "Xe": 239332.49801760627,
    "Ir": 350390.1561503677,
    "Li": 12652.668982658672,
    "Ga": 127097.25392276482,
    "S": 58450.91930974648,
    "Ag": 196631.6998062559,
    "Cl": 64626.86550096498,
    "H": 1837.3622207723647,
    #    "H": 1837.471,
    "At": 382806.58210032,
    "Ds": 512231.66461995203,
    "Sm": 274089.5127838292,
    "Co": 107428.64262000794,
    "Zr": 166291.17926437902,
    "Po": 380983.693614128,
    "Pr": 256858.93280137217,
    "Rb": 155798.26856016062,
    "Cu": 115837.27174355683,
    "Se": 143935.2748697203,
    "O": 29165.122045980286,
    #    "O": 29166.208,
    "Dy": 296219.3790062,
    "Na": 41907.785720722546,
    "Bk": 450253.456089424,
    "Pt": 355616.37744028016,
    "Hs": 492179.89127184,
    "Sg": 494002.77975803}

# Define dictionary to convert atomic symbols to covalent radii (in Angstrom)
SymbolToRadius = {
    "H": 0.37, "He": 0.32, "Li": 1.34, "Be": 0.90, "B": 0.82, "C": 0.77,
    "N": 0.75, "O": 0.73, "F": 0.71, "Ne": 0.69, "Na": 1.54, "Mg": 1.30,
    "Al": 1.18, "Si": 1.11, "P": 1.06, "S": 1.02, "Cl": 0.99, "Ar": 0.97,
    "K": 1.96, "Ca": 1.74, "Sc": 1.44, "Ti": 1.36, "V": 1.25, "Cr": 1.27,
    "Mn": 1.39, "Fe": 1.25, "Co": 1.26, "Ni": 1.21, "Cu": 1.38, "Zn": 1.31,
    "Ga": 1.26, "Ge": 1.22, "As": 1.19, "Se": 1.16, "Br": 1.14, "Kr": 1.10,
    "Rb": 2.11, "Sr": 1.92, "Y": 1.62, "Zr": 1.48, "Nb": 1.37, "Mo": 1.45,
    "Tc": 1.56, "Ru": 1.26, "Rh": 1.35, "Pd": 1.31, "Ag": 1.53, "Cd": 1.48,
    "In": 1.44, "Sn": 1.41, "Sb": 1.38, "Te": 1.35, "I": 1.33, "Xe": 1.30,
    "Cs": 2.25, "Ba": 1.98, "La": 1.69, "Ce": 1.70, "Pr": 1.70, "Nd": 1.70,
    "Pm": 1.70, "Sm": 1.70, "Eu": 1.70, "Gd": 1.70, "Tb": 1.70, "Dy": 1.70,
    "Ho": 1.70, "Er": 1.70, "Tm": 1.70, "Yb": 1.70, "Lu": 1.60, "Hf": 1.50,
    "Ta": 1.38, "W": 1.46, "Re": 1.59, "Os": 1.28, "Ir": 1.37, "Pt": 1.28,
    "Au": 1.44, "Hg": 1.49, "Tl": 1.48, "Pb": 1.47, "Bi": 1.46, "Po": 1.50,
    "At": 1.50, "Rn": 1.45, "Fr": 1.50, "Ra": 1.50, "Ac": 1.50, "Th": 1.50,
    "Pa": 1.50, "U": 1.50, "Np": 1.50, "Pu": 1.50, "Am": 1.50, "Cm": 1.50,
    "Bk": 1.50, "Cf": 1.50, "Es": 1.50, "Fm": 1.50, "Md": 1.50, "No": 1.50,
    "Lr": 1.50, "Rf": 1.50, "Db": 1.50, "Sg": 1.50, "Bh": 1.50, "Hs": 1.50,
    "Mt": 1.50, "Ds": 1.50, "Rg": 1.50, "Cn": 1.50, "Nh": 1.50, "Fl": 1.50,
    "Mc": 1.50, "Lv": 1.50, "Uus": 1.50, "Ts": 1.50}

# Define dictionary to convert atomic symbols to van der Waals radii (in Angstrom)
SymbolToVdWRadius = {
    "H": 1.10, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70,
    "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
    "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Sc": 2.15, "Ti": 2.11, "V": 2.07, "Cr": 2.06,
    "Mn": 2.05, "Fe": 2.04, "Co": 2.00, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02,
    "Rb": 3.03, "Sr": 2.49, "Y": 2.32, "Zr": 2.23, "Nb": 2.18, "Mo": 2.17,
    "Tc": 2.16, "Ru": 2.13, "Rh": 2.10, "Pd": 2.10, "Ag": 2.11, "Cd": 2.18,
    "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16,
    "Cs": 3.43, "Ba": 2.68, "La": 2.43, "Ce": 2.42, "Pr": 2.40, "Nd": 2.39,
    "Pm": 2.38, "Sm": 2.36, "Eu": 2.35, "Gd": 2.34, "Tb": 2.33, "Dy": 2.31,
    "Ho": 2.30, "Er": 2.29, "Tm": 2.27, "Yb": 2.26, "Lu": 2.24, "Hf": 2.23,
    "Ta": 2.22, "W": 2.18, "Re": 2.16, "Os": 2.16, "Ir": 2.13, "Pt": 2.13,
    "Au": 2.14, "Hg": 2.23, "Tl": 1.96, "Pb": 2.02, "Bi": 2.07, "Po": 1.97,
    "At": 2.02, "Rn": 2.20, "Fr": 3.48, "Ra": 2.83, "Ac": 2.47, "Th": 2.45,
    "Pa": 2.43, "U": 2.41, "Np": 2.39, "Pu": 2.43, "Am": 2.44, "Cm": 2.45,
    "Bk": 2.44, "Cf": 2.45, "Es": 2.45, "Fm": 2.45, "Md": 2.46, "No": 2.46,
    "Lr": 2.46, "Rf": "?", "Db": "?", "Sg": "?", "Bh": "?", "Hs": "?",
    "Mt": "?", "Ds": "?", "Rg": "?", "Cn": "?", "Nh": "?", "Fl": "?",
    "Mc": "?", "Lv": "?", "Uus": "?", "Ts": "?"}

# Define dictionary to convert atomic symbols to (Pauling) electronegativity
SymbolToEN = {
    "H": 2.20, "He": 0.00, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Ne": 0.00, "Na": 0.93, "Mg": 1.31,
    "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "Ar": 0.00,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66,
    "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16,
    "Tc": 1.90, "Ru": 2.00, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69,
    "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14,
    "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.10, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30,
    "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 1.87, "Bi": 2.02, "Po": 2.00,
    "At": 2.20, "Rn": 2.20, "Fr": 0.70, "Ra": 0.90, "Ac": 1.10, "Th": 1.30,
    "Pa": 1.50, "U": 1.38, "Np": 1.36, "Pu": 1.28, "Am": 1.13, "Cm": 1.28,
    "Bk": 1.30, "Cf": 1.30, "Es": 1.30, "Fm": 1.30, "Md": 1.30, "No": 1.30,
    "Lr": 1.30, "Rf": 1.30, "Db": 1.30, "Sg": 1.30, "Bh": 1.30, "Hs": 1.30,
    "Mt": 1.30, "Ds": 1.30, "Rg": 1.30, "Cn": 1.30, "Nh": 1.30, "Fl": 1.30,
    "Mc": 1.30, "Lv": 1.30, "Uus": 1.30, "Ts": 1.30}
