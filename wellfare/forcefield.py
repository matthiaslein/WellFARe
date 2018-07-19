###########################################################
# Force-Field class and class methods to be defined below #
###########################################################

import math
import numpy as np
import scipy.optimize

from wellfare.messages import *
from wellfare.constants import *
from wellfare.conversions import *

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


def damping_function(a, b, r):
    """
    Distance dependent damping function for atoms with symbols a and b,
    and separation r. Used to make angle dependent terms disappear at
    large distances.
    """

    # Global damping parameter to make angle dependent potentials disappear
    # at large enough distances (about 3 times the covalent distance)
    k_damping = 0.11

    # Covalent distance defined as sum of covalent radii for the two atoms
    r_cov = symbol_to_covalent_radius[a] + symbol_to_covalent_radius[b]

    f_damp = 1 / (1 + k_damping * ((r / r_cov) ** 4))

    return f_damp


def pot_bend_near_linear(a, a0, k_bend, f_damp):
    """
    Bending potential for equilibrium angles close to linearity
    """

    u = k_bend * f_damp * ((a0 - a) ** 2)

    return u


def pot_bend(a, a0, k_bend, f_damp):
    """
    Double minimum bending potential
    """

    u = k_bend * f_damp * ((math.cos(a0) - math.cos(a)) ** 2)

    return u


def lennard_jones_exponent(a, b):
    """
    Calculating the exponent 'a' that is used in the Generalised Lennard-Jones
    potential for stretches
    """

    k_en = -0.164

    # Now taking as input the symbols a and b for the two atoms in a bond
    delta_en = symbol_to_en[a] - symbol_to_en[b]
    exponent = (symbol_to_a[a] * symbol_to_a[b]) + (k_en * (delta_en ** 2))

    return exponent


def pot_generalised_lennard_jones(r, r0, k_str, a):
    """"
    Generalised Lennard-Jones potential for stretches
    """
    # k_str to be set equal to force constant for /that/ bond as read from
    # Hessian as an initial guess, fitting implemented later
    u = k_str * (1 + ((r0 / r) ** a) - 2 * ((r0 / r) ** (a / 2)))

    return u

##############################################################################
# Classes for Force Field Terms defined below
##############################################################################


class FFStretch:
    """ A stretching potential"""

    def __init__(self, a, b, r0, kind, arg):
        """ (FFStretch, int, int, number, int, [number]) -> NoneType

    A stretch potential between atoms number a and b with equilibrium
    distance r0, of type typ with arguments [arg] where arg[2] and arg[3]
    are the atomic symbols of atoms number a and b respectively
    """

        self.atom1 = a
        self.atom2 = b
        self.r0 = r0
        self.kind = kind
        self.k_str = 0.0

        if kind == 1:
            # kind 1 is the harmonic potential
            self.kind = kind
            self.k_str = arg[0]
        elif kind == 2:
            # kind 2 is the generalised Lennard-Jones potential
            self.k_str = arg[0]
            self.a = arg[1]
        else:
            # We'll also default to the harmonic potential
            self.kind = 1
            self.k_str = arg[0]

    def __str__(self):
        """ (FFStretch) -> str

    Return a string representation of the stretching potential in this format:

    (atom1, atom2, r0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, '.format(self.atom1, self.atom2, self.r0,
                                           self.kind)
        r = ''

        if self.kind == 1:
            r = '{0})'.format(self.k_str)
        elif self.kind == 2:
            r = '{0}, {1})'.format(self.k_str, self.a)

        return s + r

    def __repr__(self):
        """ (FFStretch) -> str

    Return a string representation of the stretching potential in this format:

    (atom1, atom2, r0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, '.format(self.atom1, self.atom2, self.r0,
                                           self.kind)
        r = ''

        if self.kind == 1:
            r = '{0})'.format(self.k_str)
        elif self.kind == 2:
            r = '{0}, {1})'.format(self.k_str, self.a)

        return s + r

    def set_k(self, k):
        """ (FFStretch) -> NoneType
    Set the force constant k_str equal to k
    """
        self.k_str = k

    def energy(self, r, alternative_k=None):
        """ Returns the energy of this stretching potential at distance r"""

        energy = 0.0
        if self.kind == 1:
            # Harmonic potential for stretch
            if alternative_k is None:
                energy = pot_harmonic(r, self.r0, self.k_str)
            else:
                energy = pot_harmonic(r, self.r0, alternative_k)
        elif self.kind == 2:
            # generalised Lennard-Jones potential for stretch
            if alternative_k is None:
                energy = pot_generalised_lennard_jones(r, self.r0, self.k_str,
                                                       self.a)
            else:
                energy = pot_generalised_lennard_jones(r, self.r0,
                                                       alternative_k, self.a)

        return energy


class FFBend:
    """ A bending potential"""

    def __init__(self, a, b, c, a0, kind, arg):
        """ (FFStretch, int, int, int, number, int, [number]) -> NoneType

    A bending potential between atoms number a, b and c with equilibrium
    angle a0, of type typ with arguments [arg] comprising angle force
    constant, atomic symbols of atoms a, b, and c, and the distances
    between atoms a and b and b and c
    """

        self.atom1 = a
        self.atom2 = b
        self.atom3 = c
        self.a0 = a0
        self.k_bend = arg[0]
        self.kind = kind
        if kind == 1:
            # kind 1 is the harmonic potential. a0, k_bend and type are its
            # only information, so we don't need to do more here.
            pass
        elif kind == 2:
            # kind 2 is the distance dependent damped harmonic potential. It
            # additionally needs the f_dmp value (which is calculated here).
            self.kind = 2
            r_12 = arg[4]
            r_23 = arg[5]
            f_dmp_12 = damping_function(arg[1], arg[2], r_12)
            f_dmp_23 = damping_function(arg[2], arg[3], r_23)
            self.f_dmp = f_dmp_12 * f_dmp_23

    def __str__(self):
        """ (FFBend) -> str

    Return a string representation of the bending potential in this format:

    (atom1, atom2, atom3, a0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, '.format(self.atom1, self.atom2, self.atom3,
                                           self.a0, self.kind)
        r = ''
        if self.kind == 1:
            r = '{0})'.format(self.k_bend)

        return s + r

    def __repr__(self):
        """ (FFBend) -> str

    Return a string representation of the bending potential in this format:

    (atom1, atom2, atom3, a0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, '.format(self.atom1, self.atom2, self.atom3,
                                           self.a0, self.kind)
        r = ''
        if self.kind == 1:
            r = '{0})'.format(self.k_bend)

        return s + r

    def set_k(self, new_k):
        """ (FFBend) -> NoneType
    Set the bending force constant k equal to new_k
    """
        self.k_bend = new_k

    def energy(self, a, alternative_k=None):
        """ Returns the energy of this bending potential at angle a"""

        energy = 0.0
        if self.kind == 1:
            # Simple harmonic potential
            if alternative_k is None:
                energy = pot_harmonic(a, self.a0, self.k_bend)
            else:
                energy = pot_harmonic(a, self.a0, alternative_k)
        elif self.kind == 2:
            # Distance dependent damped harmonic potential
            if (math.pi - 0.01) <= self.a0 <= (math.pi + 0.01):
                # Tolerance used above is essentially a placeholder,
                # may need changing in either direction.
                if alternative_k is None:
                    energy = pot_bend_near_linear(a, self.a0, self.k_bend,
                                                  self.f_dmp)
                else:
                    energy = pot_bend_near_linear(a, self.a0, alternative_k,
                                                  self.f_dmp)
            else:
                if alternative_k is None:
                    energy = pot_bend(a, self.a0, self.k_bend, self.f_dmp)
                else:
                    energy = pot_bend(a, self.a0, alternative_k, self.f_dmp)

        return energy


class ForceField:
    """A force-field for a given molecule"""

    def __init__(self, molecule, parametrize_bond_stretches=False,
                 parametrize_angle_bends=False,
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
        self.bends = []

        if parametrize_bond_stretches is True:
            if verbosity >= 2:
                print("\nAdding Force Field bond stretching terms to"
                      " WellFARe molecule: ", molecule.name)
            for bond in molecule.bonds:
                # Extracting the force constant, fc, for a<->b from the Hessian
                a = np.array([molecule.atoms[bond[0]].coord[0],
                              molecule.atoms[bond[0]].coord[1],
                              molecule.atoms[bond[0]].coord[2]])
                b = np.array([molecule.atoms[bond[1]].coord[0],
                              molecule.atoms[bond[1]].coord[1],
                              molecule.atoms[bond[1]].coord[2]])
                c1 = (a - b)
                c2 = (b - a)
                c = np.zeros(molecule.num_atoms() * 3)
                c[3 * bond[0]] = c1[0]
                c[3 * bond[0] + 1] = c1[1]
                c[3 * bond[0] + 2] = c1[2]
                c[3 * bond[1]] = c2[0]
                c[3 * bond[1] + 1] = c2[1]
                c[3 * bond[1] + 2] = c2[2]
                c = c / np.linalg.norm(c)
                fc = np.dot(np.dot(c, self.hessian), np.transpose(c))
                if fc < 0.002:
                    msg_program_warning(" This force constant is smaller"
                                        " than 0.002")
                if verbosity >= 3:
                    print(" {:<3} ({:3d}) and {:<3} ({:3d}) (Force constant:"
                          " {: .3f})".format(molecule.atoms[bond[0]].symbol(),
                                             bond[0] + 1,
                                             molecule.atoms[bond[1]].symbol(),
                                             bond[1] + 1, fc))

                # Add as harmonic potential
                new_stretch = FFStretch(bond[0], bond[1], r0=ang_to_bohr(
                    molecule.atm_atm_dist(bond[0], bond[1])), kind=1, arg=[fc])

                # Add as modified Lennard-Jones potential
                new_exponent = lennard_jones_exponent(
                    molecule.atoms[bond[0]].symbol(),
                    molecule.atoms[bond[1]].symbol())
                new_stretch = FFStretch(bond[0], bond[1], r0=ang_to_bohr(
                    molecule.atm_atm_dist(bond[0], bond[1])), kind=2,
                                        arg=[fc, new_exponent])
                self.add_stretch(new_stretch)
        if parametrize_angle_bends is True:
            if verbosity >= 2:
                print("\nAdding Force Field angle bending terms to"
                      " WellFARe molecule: ", molecule.name)
            for angle in molecule.angles:
                a = np.array([molecule.atoms[angle[0]].coord[0],
                              molecule.atoms[angle[0]].coord[1],
                              molecule.atoms[angle[0]].coord[2]])
                b = np.array([molecule.atoms[angle[1]].coord[0],
                              molecule.atoms[angle[1]].coord[1],
                              molecule.atoms[angle[1]].coord[2]])
                c = np.array([molecule.atoms[angle[2]].coord[0],
                              molecule.atoms[angle[2]].coord[1],
                              molecule.atoms[angle[2]].coord[2]])
                aprime = a - b
                bprime = c - b
                p = np.cross(aprime, bprime)
                adprime = np.cross(p, aprime)
                bdprime = np.cross(bprime, p)
                c = np.zeros(molecule.num_atoms() * 3)
                c[3 * angle[0]] = adprime[0]
                c[3 * angle[0] + 1] = adprime[1]
                c[3 * angle[0] + 2] = adprime[2]
                c[3 * angle[2]] = bdprime[0]
                c[3 * angle[2] + 1] = bdprime[1]
                c[3 * angle[2] + 2] = bdprime[2]
                if abs(np.linalg.norm(c)) >= 0.00001:
                    c = c / np.linalg.norm(c)
                    fc = np.dot(np.dot(c, self.hessian), np.transpose(c))
                else:
                    fc = 0.0
                # if fc < 0.002:
                #     msg_program_warning(" This force constant is smaller"
                #                         " than 0.002")
                if verbosity >= 3:
                    print(
                        " {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d})"
                        " (Force constant: {: .3f})".format(
                            molecule.atoms[angle[0]].symbol(),
                            angle[0] + 1,
                            molecule.atoms[angle[1]].symbol(),
                            angle[1] + 1,
                            molecule.atoms[angle[2]].symbol(),
                            angle[2] + 1, fc))
                new_bend = FFBend(angle[0],
                                   angle[1],
                                   angle[2],
                                   molecule.atm_atm_atm_angle(angle[0],angle[1],angle[2]),
                                   2, [fc, molecule.atoms[angle[0]].symbol(),
                                       molecule.atoms[
                                           angle[1]].symbol(),
                                       molecule.atoms[
                                           angle[2]].symbol(),
                                       molecule.atm_atm_dist(
                                           angle[0],
                                           angle[1]),
                                       molecule.atm_atm_dist(
                                           angle[1],
                                           angle[2])])
                self.add_bend(new_bend)

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

    def add_bend(self, bend):
        self.bends.append(bend)

    def coordinates(self):
        coordinates = []
        for atom in self.atoms:
            coordinates.append(ang_to_bohr(atom.xpos()))
            coordinates.append(ang_to_bohr(atom.ypos()))
            coordinates.append(ang_to_bohr(atom.zpos()))
        return coordinates

    def coefficients(self):
        coefficients = []
        for stretch in self.stretches:
            coefficients.append(stretch.k_str)
        for bend in self.bends:
            coefficients.append(bend.k_bend)
        return coefficients

    def force_constants(self):
        constants = []
        for stretch in self.stretches:
            constants.append(stretch.k_str)
        for bend in self.bends:
            constants.append(bend.k_bend)
        return constants

    def gradient(self, coords=None):
        # The force-field gradient with respect to coordinates
        if coords is None:
            coords = self.coordinates()

        epsilon = 1E-5
        return scipy.optimize.approx_fprime(coords, self.total_energy,
                                            epsilon)

    def gradient_coeff(self, coeff=None):
        # The force-field gradient with respect to ff-coefficients
        if coeff is None:
            coeff = self.coefficients()

        epsilon = 1E-5
        return scipy.optimize.approx_fprime(coeff, self.total_energy_coeff,
                                            epsilon)

    def total_energy(self, coordinates):
        energy = self.energy_baseline

        for stretch in self.stretches:
            atom1_x_coord = coordinates[stretch.atom1 * 3]
            atom1_y_coord = coordinates[stretch.atom1 * 3 + 1]
            atom1_z_coord = coordinates[stretch.atom1 * 3 + 2]
            atom2_x_coord = coordinates[stretch.atom2 * 3]
            atom2_y_coord = coordinates[stretch.atom2 * 3 + 1]
            atom2_z_coord = coordinates[stretch.atom2 * 3 + 2]

            x_squared = (atom1_x_coord-atom2_x_coord)**2
            y_squared = (atom1_y_coord-atom2_y_coord)**2
            z_squared = (atom1_z_coord-atom2_z_coord)**2
            dist = math.sqrt(x_squared + y_squared + z_squared)

            # if stretch.kind == 1:
            #     print("Harmonic stretching potential")
            # elif stretch.kind == 2:
            #     print("Generalised Lennard-Jones potential")
            # else:
            #     print("Some other potential")
            # print("Stretch contribution: {}".format(stretch.energy(dist)))
            energy += stretch.energy(dist)

        for bend in self.bends:
            atom1_x_coord = coordinates[bend.atom1 * 3]
            atom1_y_coord = coordinates[bend.atom1 * 3 + 1]
            atom1_z_coord = coordinates[bend.atom1 * 3 + 2]
            atom2_x_coord = coordinates[bend.atom2 * 3]
            atom2_y_coord = coordinates[bend.atom2 * 3 + 1]
            atom2_z_coord = coordinates[bend.atom2 * 3 + 2]
            atom3_x_coord = coordinates[bend.atom3 * 3]
            atom3_y_coord = coordinates[bend.atom3 * 3 + 1]
            atom3_z_coord = coordinates[bend.atom3 * 3 + 2]

            # Calculate the distance between each pair of atoms
            x_squared = (atom1_x_coord - atom2_x_coord) ** 2
            y_squared = (atom1_y_coord - atom2_y_coord) ** 2
            z_squared = (atom1_z_coord - atom2_z_coord) ** 2
            d_bond_1 = math.sqrt(x_squared + y_squared + z_squared)

            x_squared = (atom2_x_coord - atom3_x_coord) ** 2
            y_squared = (atom2_y_coord - atom3_y_coord) ** 2
            z_squared = (atom2_z_coord - atom3_z_coord) ** 2
            d_bond_2 = math.sqrt(x_squared + y_squared + z_squared)

            x_squared = (atom1_x_coord - atom3_x_coord) ** 2
            y_squared = (atom1_y_coord - atom3_y_coord) ** 2
            z_squared = (atom1_z_coord - atom3_z_coord) ** 2
            d_non_bond = math.sqrt(x_squared + y_squared + z_squared)

            # Use those distances and the cosine rule to calculate bond
            # angle theta
            numerator = d_bond_1 ** 2 + d_bond_2 ** 2 - d_non_bond ** 2
            denominator = 2 * d_bond_1 * d_bond_2
            argument = numerator / denominator
            # This is a safety check to account for numerical noise that might
            # screw up angles in linear molecules ...
            if argument > 1.0:
                argument = 1.0
            elif argument < -1.0:
                argument = -1.0
            theta = np.arccos(argument)

            # if bend.kind == 1:
            #     print("Harmonic bending potential")
            # elif bend.kind == 2:
            #     print("Distance damped harmonic bending potential")
            # else:
            #     print("Some other bending potential")
            # print("Bend contribution: {}".format(bend.energy(theta)))
            energy += bend.energy(theta)

        return energy

    def total_energy_coeff(self, coefficients):
        # Calculate the energy as a function of the coefficients
        energy = self.energy_baseline
        coordinates = self.coordinates()

        for idx, stretch in enumerate(self.stretches):
            atom1_x_coord = coordinates[stretch.atom1 * 3]
            atom1_y_coord = coordinates[stretch.atom1 * 3 + 1]
            atom1_z_coord = coordinates[stretch.atom1 * 3 + 2]
            atom2_x_coord = coordinates[stretch.atom2 * 3]
            atom2_y_coord = coordinates[stretch.atom2 * 3 + 1]
            atom2_z_coord = coordinates[stretch.atom2 * 3 + 2]

            x_squared = (atom1_x_coord-atom2_x_coord)**2
            y_squared = (atom1_y_coord-atom2_y_coord)**2
            z_squared = (atom1_z_coord-atom2_z_coord)**2
            dist = math.sqrt(x_squared + y_squared + z_squared)

            # if stretch.kind == 1:
            #     print("Harmonic stretching potential")
            # elif stretch.kind == 2:
            #     print("Generalised Lennard-Jones potential")
            # else:
            #     print("Some other potential")
            # print("Stretch contribution: {}".format(stretch.energy(dist)))
            energy += stretch.energy(dist, alternative_k=coefficients[idx])

        offset = len(self.stretches)
        for idx, bend in enumerate(self.bends):
            atom1_x_coord = coordinates[bend.atom1 * 3]
            atom1_y_coord = coordinates[bend.atom1 * 3 + 1]
            atom1_z_coord = coordinates[bend.atom1 * 3 + 2]
            atom2_x_coord = coordinates[bend.atom2 * 3]
            atom2_y_coord = coordinates[bend.atom2 * 3 + 1]
            atom2_z_coord = coordinates[bend.atom2 * 3 + 2]
            atom3_x_coord = coordinates[bend.atom3 * 3]
            atom3_y_coord = coordinates[bend.atom3 * 3 + 1]
            atom3_z_coord = coordinates[bend.atom3 * 3 + 2]

            # Calculate the distance between each pair of atoms
            x_squared = (atom1_x_coord - atom2_x_coord) ** 2
            y_squared = (atom1_y_coord - atom2_y_coord) ** 2
            z_squared = (atom1_z_coord - atom2_z_coord) ** 2
            d_bond_1 = math.sqrt(x_squared + y_squared + z_squared)

            x_squared = (atom2_x_coord - atom3_x_coord) ** 2
            y_squared = (atom2_y_coord - atom3_y_coord) ** 2
            z_squared = (atom2_z_coord - atom3_z_coord) ** 2
            d_bond_2 = math.sqrt(x_squared + y_squared + z_squared)

            x_squared = (atom1_x_coord - atom3_x_coord) ** 2
            y_squared = (atom1_y_coord - atom3_y_coord) ** 2
            z_squared = (atom1_z_coord - atom3_z_coord) ** 2
            d_non_bond = math.sqrt(x_squared + y_squared + z_squared)

            # Use those distances and the cosine rule to calculate bond
            # angle theta
            numerator = d_bond_1 ** 2 + d_bond_2 ** 2 - d_non_bond ** 2
            denominator = 2 * d_bond_1 * d_bond_2
            argument = numerator / denominator
            # This is a safety check to account for numerical noise that might
            # screw up angles in linear molecules ...
            if argument > 1.0:
                argument = 1.0
            elif argument < -1.0:
                argument = -1.0
            theta = np.arccos(argument)

            # if bend.kind == 1:
            #     print("Harmonic bending potential")
            # elif bend.kind == 2:
            #     print("Distance damped harmonic bending potential")
            # else:
            #     print("Some other bending potential")
            # print("Bend contribution: {}".format(bend.energy(theta)))
            energy += bend.energy(theta,
                                  alternative_k=coefficients[idx + offset])

        return energy
