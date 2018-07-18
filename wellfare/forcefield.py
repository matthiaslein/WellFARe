###########################################################
# Force-Field class and class methods to be defined below #
###########################################################

import math
import numpy as np

from wellfare.messages import *

##############################################################################
# Potential Functions are to be defined below here
##############################################################################


def pot_harmonic(a, a0, k):
    """"
    Harmonic potential (for stretches and bends)
    """

    u = 0.5 * k * (a - a0) ** 2

    return u


def pot_morse(r, r0, d, b):
    """
    Morse oscillator potential
    """

    u = d * (1 - math.exp(-b * (r - r0))) ** 2

    return u


##############################################################################
# Classes for Force Field Terms defined below
##############################################################################


class FFStretch:
    """ A stretching potential"""

    def __init__(self, a, b, r0, typ, arg):
        """ (FFStretch, int, int, number, int, [number]) -> NoneType

    A stretch potential between atoms number a and b with equilibrium
    distance r0, of type typ with arguments [arg] where arg[2] and arg[3]
    are the atomic symbols of atoms number a and b respectively
    """

        self.atom1 = a
        self.atom2 = b
        self.r0 = r0
        self.typ = typ
        self.k_str = 0.0

        if typ == 1:
            # Type 1 is the harmonic potential
            self.typ = typ
            self.k_str = arg[0]
        elif typ == 2:
            # Type 1 is the Morse potential
            self.d = arg[0]
            self.b = arg[1]
        else:
            # We'll also default to the harmonic potential
            self.typ = 1
            self.k_str = arg[0]

    def __str__(self):
        """ (FFStretch) -> str

    Return a string representation of the stretching potential in this format:

    (atom1, atom2, r0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, '.format(self.atom1, self.atom2, self.r0,
                                           self.typ)
        r = ''

        if self.typ == 1:
            r = '{0})'.format(self.k_str)
        elif self.typ == 2:
            r = '{0}, {1})'.format(self.d, self.b)

        return s + r

    def __repr__(self):
        """ (FFStretch) -> str

    Return a string representation of the stretching potential in this format:

    (atom1, atom2, r0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, '.format(self.atom1, self.atom2, self.r0,
                                           self.typ)
        r = ''

        if self.typ == 1:
            r = '{0})'.format(self.k_str)
        elif self.typ == 2:
            r = '{0}, {1})'.format(self.d, self.b)

        return s + r

    def set_k(self, k):
        """ (FFStretch) -> NoneType
    Set the force constant k_str equal to k
    """
        self.k_str = k

    def energy(self, r, replace_k = None):
        """ Returns the energy of this stretching potential at distance r"""

        energy = 0.0
        if self.typ == 1:
            # print("Using Harmonic potential for stretch")
            # print("With r = " + str(r) + ", r0 = " + str(
            #     self.r0) + ", k = " + str(self.k))
            if replace_k is None:
                energy = pot_harmonic(r, self.r0, self.k_str)
            else:
                energy = pot_harmonic(r, self.r0, replace_k)
        elif self.typ == 2:
            # print("Using Morse potential for stretch")
            # print("With r = " + str(r) + ", r0 = " + str(self.r0) +
            #       ", D = " + str(self.D) + ", b = " + str(self.b))
            if replace_k is None:
                energy = pot_morse(r, self.r0, self.d, self.b)
            else:
                energy = pot_morse(r, self.r0, replace_k, self.b)

        return energy


class ForceField:
    """A force-field for a given molecule"""

    def __init__(self, molecule, parametrize_bond_stretches=False,
                 verbosity=0):
        """ (Molecule, str, int) -> NoneType

    The force field is composed of a collection of potential energy terms that
    are commonly used (stretching, bending, torsion, ...) and some other ones
    that catch different (e.g. electrostatic, dispersion, h-bond, ...) inter-
    actions.
    """

        # The following lists are inherited from the molecule that is used
        # to initialize the force field
        self.atoms = []
        for atom in molecule.atoms:
            self.atoms.append(atom)
        self.bonds = []
        for bond in molecule.bonds:
            self.bonds.append(bond)
        self.angles = []
        for angle in molecule.angles:
            self.angles.append(angle)
        self.dihedrals = []
        for dihedral in molecule.dihedrals:
            self.dihedrals.append(dihedral)
        self.hessian = molecule.H_QM
        self.energy_baseline = molecule.qm_energy

        # These are (initially empty) lists for the possible terms in the
        # force field
        self.stretches = []

        if parametrize_bond_stretches is True:
            if verbosity >= 2:
                print("\nAdding Force Field bond stretching terms to"
                      " WellFARe molecule: ", molecule.name)
            for i in molecule.bonds:
                # Extracting the force constant, fc, for a<->b from the Hessian
                a = np.array([molecule.atoms[i[0]].coord[0],
                              molecule.atoms[i[0]].coord[1],
                              molecule.atoms[i[0]].coord[2]])
                b = np.array([molecule.atoms[i[1]].coord[0],
                              molecule.atoms[i[1]].coord[1],
                              molecule.atoms[i[1]].coord[2]])
                c1 = (a - b)
                c2 = (b - a)
                c = np.zeros(molecule.num_atoms() * 3)
                c[3 * i[0]] = c1[0]
                c[3 * i[0] + 1] = c1[1]
                c[3 * i[0] + 2] = c1[2]
                c[3 * i[1]] = c2[0]
                c[3 * i[1] + 1] = c2[1]
                c[3 * i[1] + 2] = c2[2]
                c = c / np.linalg.norm(c)
                fc = np.dot(np.dot(c, self.hessian), np.transpose(c))
                if fc < 0.002:
                    msg_program_warning(" This force constant is smaller"
                                        " than 0.002")
                if verbosity >= 2:
                    print(" {:<3} ({:3d}) and {:<3} ({:3d}) (Force constant:"
                          " {: .3f})".format(molecule.atoms[i[0]].symbol(),
                                             i[0],
                                             molecule.atoms[i[1]].symbol(),
                                             i[1], fc))
                new_stretch = FFStretch(i[0], i[1],
                                        molecule.atm_atm_dist(i[0], i[1]), 1,
                                        [fc])
                self.add_stretch(new_stretch)

    def __str__(self):
        """ (Molecule) -> str

    Return a string representation of this force field
    """

        res = ''
        for atom in self.atoms:
            res = res + str(atom) + ', '
        res = res[:-2]
        return res

    def __repr__(self):
        """ (Molecule) -> str

    Return a string representation of this force field
    """

        res = ''
        for atom in self.atoms:
            res = res + str(atom) + ', '
        res = res[:-2]
        return res

    def add_stretch(self, stretch):
        self.stretches.append(stretch)

    def coordinates(self):
        coordinates = []
        for atom in self.atoms:
            coordinates.append(atom.xpos())
            coordinates.append(atom.ypos())
            coordinates.append(atom.zpos())
        return coordinates

    def force_constants(self):
        constants = []
        for stretch in self.stretches:
            constants.append(stretch.k_str)
        return constants

    def total_energy(self, coordinates):
        energy = self.energy_baseline

        for stretch in self.stretches:
            atom1_x_coord = coordinates[stretch.atom1 * 3]
            atom1_y_coord = coordinates[stretch.atom1 * 3 + 1]
            atom1_z_coord = coordinates[stretch.atom1 * 3 + 2]
            atom2_x_coord = coordinates[stretch.atom2 * 3]
            atom2_y_coord = coordinates[stretch.atom2 * 3 + 1]
            atom2_z_coord = coordinates[stretch.atom2 * 3 + 2]
            distance = math.sqrt((atom1_x_coord-atom2_x_coord)**2+(atom1_y_coord-atom2_y_coord)**2+(atom1_z_coord-atom2_z_coord)**2)
            energy += stretch.energy(distance)

        return energy

