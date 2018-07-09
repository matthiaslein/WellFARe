###############################################################################
# Methods for unit conversions are defined below
###############################################################################

from wellfare.constants import symbol_to_number


# Conversion of length in Angstroms to
# atomic units (Bohrs)
def ang_to_bohr(ang):
    return ang * 1.889725989


def bohr_to_ang(bohr):
    return bohr / 1.889725989


# Test if the argument is (can be converted to)
# an integer number
def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# Test if the argument is (can be converted to)
# a floating point number
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Test if the argument is (can be converted to)
# an atomic symbol
def is_atom_symbol(s):
    if s in symbol_to_number:
        return True
    else:
        return False
