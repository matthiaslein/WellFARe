################################################################################
# Methods for unit conversions are defined below
################################################################################

# Conversion of length in Angstroms to
# atomic units (Bohrs)
def ang_to_bohr(ang):
    return ang * 1.889725989


def bohr_to_ang(bohr):
    return bohr / 1.889725989
