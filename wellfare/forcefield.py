###########################################################
# Force-Field class and class methods to be defined below #
###########################################################

import itertools
import math
import numpy as np
import scipy.optimize
from typing import Optional

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

    u = 0.5 * k * ((a - a0) ** 2)

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


def pot_bend(a, a0, k_bend, f_damp=1.0):
    """
    harmonic (i.e. quadratic) ending potential
    """

    u = k_bend * f_damp * ((a0 - a) ** 2)

    return u

def dihedral_pot_bend(theta, theta0, k_tors, f_damp=1.0):
    """
    harmonic (i.e. quadratic) ending potential
    """

    # Calculating the difference for dihedral angles is a little complicated
    # because of the periodicity. For this "hack potential", we use the exact
    # Δθ calculation to make sure we don't get negative energies for negative
    # equilibrium dihedrals.
    # delta_theta = 180 - math.fabs(math.fabs(theta0 - theta) - 180)
    #
    # u = k_tors * f_damp * (delta_theta ** 2)

    u = k_tors * (1 + np.cos(math.radians(180) + theta - theta0))

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

    def __init__(self, a, a_sym, b, b_sym, c, c_sym, a0, kind, arg):
        """ (FFStretch, int, int, int, number, int, [number]) -> NoneType

    A bending potential between atoms number a, b and c with equilibrium
    angle a0, of type typ with arguments [arg] comprising angle force
    constant, atomic symbols of atoms a, b, and c, and the distances
    between atoms a and b and b and c
    """

        self.atom1 = a
        self.symbol1 = a_sym
        self.atom2 = b
        self.symbol2 = b_sym
        self.atom3 = c
        self.symbol3 = c_sym
        self.a0 = a0
        self.k_bend = arg[0]
        self.kind = kind

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

    def energy(self, a, dist1, dist2, alternative_k=None):
        """ Returns the energy of this bending potential at angle a"""

        # print("a0 = {: .3f}, a = {: .3f}".format(math.degrees(self.a0),
        #                                        math.degrees(a)))
        energy = 0.0
        if self.kind == 1:
            # Simple harmonic potential
            if alternative_k is None:
                energy = pot_harmonic(a, self.a0, self.k_bend)
            else:
                energy = pot_harmonic(a, self.a0, alternative_k)
        elif self.kind == 2:
            # Calculate distance dependent damping f_dmp
            f_dmp_12 = damping_function(self.symbol1, self.symbol2, dist1)
            f_dmp_23 = damping_function(self.symbol2, self.symbol3, dist2)
            f_dmp = f_dmp_12 * f_dmp_23
            # Distance dependent damped harmonic potential

            if alternative_k is None:
                energy = pot_bend(a, self.a0, self.k_bend, f_dmp)
            else:
                energy = pot_bend(a, self.a0, alternative_k, f_dmp)
        # if energy >= 1E-11:
        #     print("Angle bending energy {: .3E}".format(energy))
        return energy


class FFTorsion:
    """ A torsion potential"""

    def __init__(self, a, a_sym, b, b_sym, c, c_sym, d, d_sym, theta0, kind, arg):
        """ (FFTorsion, int, int, int, int, number, int, [number]) -> NoneType

    A torsion potential between atoms number a, b, c and d with equilibrium
    angle theta0, of type typ with arguments [arg] comprising the dihedral
    force constant, the atomic symbols of atoms a, b, c and d,  the
    ab, bc and cd bond lengths, the values of k_tors_n and equilibrium angle
    from fitting to HMOEnergy result, if applicable, and a Boolean for whether
    the central bond of the dihedral is in a ring
    """

        self.atom1 = a
        self.symbol1 = a_sym
        self.atom2 = b
        self.symbol2 = b_sym
        self.atom3 = c
        self.symbol3 = c_sym
        self.atom4 = d
        self.symbol4 = d_sym
        self.theta0 = theta0
        self.kind = kind
        self.k_tors = arg[0]

    def __str__(self):
        """ (FFTorsion) -> str

    Return a string representation of the torsion potential in this format:

    (atom1, atom2, atom3, atom4, theta0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, {4}, '.format(self.atom1, self.atom2,
                                                self.atom3, self.atom4,
                                                self.theta0, self.kind)

        if self.kind == 1:
            r = '{0})'.format(self.k_tors)

        return s + r

    def __repr__(self):
        """ (FFTorsion) -> str

    Return a string representation of the torsion potential in this format:

    (atom1, atom2, atom3, atom4, theta0, type, arguments)

    """

        s = '({0}, {1}, {2}, {3}, {4}, '.format(self.atom1, self.atom2,
                                                self.atom3, self.atom4,
                                                self.theta0, self.kind)

        if self.kind == 1:
            r = '{0})'.format(self.k_tors)

        return s + r

    def setk(self, newk):
        """ Sets the single force constant k for type 1 or 3 torsion potentials equal to newk """

        self.k_tors = newk

    def energy(self, theta, dist1, dist2, dist3, alternative_k=None):
        """ Returns the energy of this torsion potential at angle theta"""

        # print("theta0 = {: .3f}, theta = {: .3f}".format(
        #     math.degrees(self.theta0), math.degrees(theta)))
        energy = 0.0
        if self.kind == 1:
            if alternative_k is None:
                energy = pot_harmonic(theta, self.theta0, self.k_tors)
            else:
                energy = pot_harmonic(theta, self.theta0, alternative_k)
        elif self.kind == 2:
            # Calculate distance dependent damping f_dmp
            f_dmp_12 = damping_function(self.symbol1, self.symbol2, dist1)
            f_dmp_23 = damping_function(self.symbol2, self.symbol3, dist2)
            f_dmp_34 = damping_function(self.symbol3, self.symbol4, dist3)
            f_dmp = f_dmp_12 * f_dmp_23 * f_dmp_34
            # Distance dependent damped harmonic potential

            if alternative_k is None:
                energy = dihedral_pot_bend(theta, self.theta0, self.k_tors, f_dmp)
            else:
                energy = dihedral_pot_bend(theta, self.theta0, alternative_k, f_dmp)
        # if energy >= 1E-11:
        #     print("Torsional energy: {: .3E}".format(energy))
        return energy

class ForceField:
    """A force-field for a given molecule"""

    def __init__(self, molecule, parametrize_bond_stretches=False,
                 parametrize_angle_bends=False, parametrize_distance_matrix=False,
                 parametrize_torsions=False,
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
        self.qm_hessian = molecule.H_QM
        self.energy_baseline = molecule.qm_energy

        # These are (initially empty) lists for the possible terms in the
        # force field
        self.stretches = []
        self.bends = []
        self.torsions = []
        if parametrize_distance_matrix is True:
            if verbosity >= 2:
                print("\nAdding Force Field stretching terms to"
                      " WellFARe molecule for all combinations: ", molecule.name)
            for i, j in itertools.combinations(range(molecule.num_atoms()), 2):
                fc = 0.2
                if verbosity >= 3:
                    print(" {:<3} ({:3d}) and {:<3} ({:3d}) (Force constant:"
                          " {: .3f})".format(molecule.atoms[i].symbol(),
                                             i + 1,
                                             molecule.atoms[j].symbol(),
                                             j + 1, fc))
                # Add as modified Lennard-Jones potential
                new_exponent = lennard_jones_exponent(
                    molecule.atoms[i].symbol(),
                    molecule.atoms[j].symbol())
                new_stretch = FFStretch(i, j, r0=ang_to_bohr(
                    molecule.atm_atm_dist(i, j)), kind=2,
                                        arg=[fc, new_exponent])
                self.add_stretch(new_stretch)
        if parametrize_bond_stretches is True:
            if verbosity >= 2:
                print("\nAdding Force Field bond stretching terms to"
                      " WellFARe molecule: ", molecule.name)
            for bond in molecule.bonds:
                # The following, cumbersome, calculation of the force constant
                # from the hessian is completely unnecessary!
                # a = np.array([molecule.atoms[bond[0]].coord[0],
                #               molecule.atoms[bond[0]].coord[1],
                #               molecule.atoms[bond[0]].coord[2]])
                # b = np.array([molecule.atoms[bond[1]].coord[0],
                #               molecule.atoms[bond[1]].coord[1],
                #               molecule.atoms[bond[1]].coord[2]])
                # c1 = (a - b)
                # c2 = (b - a)
                # c = np.zeros(molecule.num_atoms() * 3)
                # c[3 * bond[0]] = c1[0]
                # c[3 * bond[0] + 1] = c1[1]
                # c[3 * bond[0] + 2] = c1[2]
                # c[3 * bond[1]] = c2[0]
                # c[3 * bond[1] + 1] = c2[1]
                # c[3 * bond[1] + 2] = c2[2]
                # c = c / np.linalg.norm(c)
                # fc = np.dot(np.dot(c, self.qm_hessian), np.transpose(c))
                fc = 0.5

                if verbosity >= 3:
                    print(" {:<3} ({:3d}) and {:<3} ({:3d}) (Force constant:"
                          " {: .3f})".format(molecule.atoms[bond[0]].symbol(),
                                             bond[0] + 1,
                                             molecule.atoms[bond[1]].symbol(),
                                             bond[1] + 1, fc))
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
                print("\nAdding Force Field angle bending terms and corresponding"
                      " 1,3 stretches to\n"
                      "WellFARe molecule: ", molecule.name)
            for angle in molecule.angles:
                # The following, cumbersome, calculation of the force constant
                # from the hessian is completely unnecessary!
                # a = np.array([molecule.atoms[angle[0]].coord[0],
                #               molecule.atoms[angle[0]].coord[1],
                #               molecule.atoms[angle[0]].coord[2]])
                # b = np.array([molecule.atoms[angle[1]].coord[0],
                #               molecule.atoms[angle[1]].coord[1],
                #               molecule.atoms[angle[1]].coord[2]])
                # c = np.array([molecule.atoms[angle[2]].coord[0],
                #               molecule.atoms[angle[2]].coord[1],
                #               molecule.atoms[angle[2]].coord[2]])
                # aprime = a - b
                # bprime = c - b
                # p = np.cross(aprime, bprime)
                # adprime = np.cross(p, aprime)
                # bdprime = np.cross(bprime, p)
                # c = np.zeros(molecule.num_atoms() * 3)
                # c[3 * angle[0]] = adprime[0]
                # c[3 * angle[0] + 1] = adprime[1]
                # c[3 * angle[0] + 2] = adprime[2]
                # c[3 * angle[2]] = bdprime[0]
                # c[3 * angle[2] + 1] = bdprime[1]
                # c[3 * angle[2] + 2] = bdprime[2]
                # if abs(np.linalg.norm(c)) >= 0.00001:
                #     c = c / np.linalg.norm(c)
                #     fc = np.dot(np.dot(c, self.qm_hessian), np.transpose(c))
                # else:
                #     fc = 0.0
                fc = 0.5
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
                new_bend = FFBend(angle[0], molecule.atm_symbol(angle[0]),
                                   angle[1], molecule.atm_symbol(angle[1]),
                                   angle[2], molecule.atm_symbol(angle[2]),
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
                # And now for the corresponding 1,3 stretch - but only if the
                # angle wasn't near linear.
                if (math.pi - 0.2) >= molecule.atm_atm_atm_angle(angle[0],
                                                                 angle[1],
                                                                 angle[2]):
                    fc = 0.5
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}) and {:<3} ({:3d}) (Force constant:"
                            " {: .3f})".format(
                                molecule.atoms[angle[0]].symbol(),
                                angle[0] + 1,
                                molecule.atoms[angle[2]].symbol(),
                                angle[2] + 1, fc))

                    # Add as modified Lennard-Jones potential
                    new_exponent = lennard_jones_exponent(
                        molecule.atoms[angle[0]].symbol(),
                        molecule.atoms[angle[2]].symbol())
                    new_stretch = FFStretch(angle[0], angle[2], r0=ang_to_bohr(
                        molecule.atm_atm_dist(angle[0], angle[2])), kind=2,
                                            arg=[fc, new_exponent])
                    self.add_stretch(new_stretch)
        if parametrize_torsions is True:
            if verbosity >= 2:
                print(
                    "\nAdding Force Field torsional terms to WellFARe molecule: ",
                    molecule.name)
            for dihedral in molecule.dihedrals:
                fc = 0.5
                angle = molecule.atm_atm_atm_atm_dihedral(
                                            dihedral[0], dihedral[1],
                                            dihedral[2], dihedral[3])
                new_torsion = FFTorsion(dihedral[0],
                                        molecule.atoms[dihedral[0]].symbol(),
                                        dihedral[1],
                                        molecule.atoms[dihedral[1]].symbol(),
                                        dihedral[2],
                                        molecule.atoms[dihedral[2]].symbol(),
                                        dihedral[3],
                                        molecule.atoms[dihedral[3]].symbol(),
                                        theta0=angle, kind=2,
                                        arg=[fc])
                self.add_torsion(new_torsion)

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

    def add_torsion(self, torsion):
        self.torsions.append(torsion)

    def ff_coordinates(self):
        # Returns the current coordinates the force-field is set to
        # (in Bohr, not in Ångström)
        coordinates = []
        for atom in self.atoms:
            coordinates.append(ang_to_bohr(atom.xpos()))
            coordinates.append(ang_to_bohr(atom.ypos()))
            coordinates.append(ang_to_bohr(atom.zpos()))
        return coordinates

    def ff_atm_atm_dist(self, i: int, j: int) -> float:
        """
        Report the distance between atoms i and j in Bohr.

        :param i: First atom for distance measurement.
        :param j: Second atom for distance measurement.
        :return: The distance between atoms i and j in Bohr.
        """

        distance = (self.atm_[i].xpos() - self.atm_[j].xpos()) ** 2
        distance = distance + (self.atm_[i].ypos() - self.atm_[j].ypos()) ** 2
        distance = distance + (self.atm_[i].zpos() - self.atm_[j].zpos()) ** 2
        distance = math.sqrt(distance)

        return ang_to_bohr(distance)

    def ff_coefficients(self):
        coefficients = []
        for stretch in self.stretches:
            coefficients.append(stretch.k_str)
        for bend in self.bends:
            coefficients.append(bend.k_bend)
        for torsion in self.torsions:
            coefficients.append(torsion.k_tors)
        return coefficients

    def ff_set_coefficients(self, new_coefficients):
        for idx, stretch in enumerate(self.stretches):
            stretch.k_str = new_coefficients[idx]
        offset = len(self.stretches)
        for idx, bend in enumerate(self.bends):
            bend.k_bend = new_coefficients[idx + offset]
        offset += len(self.bends)
        for idx, torsion in enumerate(self.torsions):
            torsion.k_tors = new_coefficients[idx + offset]

    def ff_gradient(self, coordinates=None, coefficients=None):
        # epsilon = 1E-4
        epsilon = 1E-5
        # The force-field gradient with respect to coordinates
        if coordinates is None:
            coordinates = self.ff_coordinates()
        if coefficients is None:
            coefficients = self.ff_coefficients()

        return scipy.optimize.approx_fprime(coordinates, self.ff_energy,
                                            epsilon, coefficients)

    def ff_hessian(self, coefficients=None, coordinates=None):
        # epsilon = 1E-4
        epsilon = 1E-5
        # The force-field hessian with respect to coordinates
        if coordinates is None:
            coordinates = self.ff_coordinates()
        if coefficients is None:
            coefficients = self.ff_coefficients()

        # Calculate gradient at given coordinates
        grad = self.ff_gradient(coordinates=coordinates,
                                coefficients=coefficients)

        # Initialise empty matrix for Hessian
        hess = np.zeros((len(coordinates), len(coordinates)))

        # Loop over rows...
        for i in range(len(coordinates)):
            # Store value of current coordinate
            original_coord = coordinates[i]
            # Displace current coordinate
            coordinates[i] = coordinates[i] + epsilon
            # Calculate gradient at this new coordinate
            other_grad = self.ff_gradient(coordinates=coordinates,
                                          coefficients=coefficients)
            # Place the calculated second derivatives for coordinate
            # i into the ith column of the Hessian matrix
            hess[:, i] = (other_grad - grad) / epsilon
            # Restore coordinate
            coordinates[i] = original_coord

        return hess

    def ff_energy(self, coordinates=None, coefficients=None, verbosity=0):
        # Calculate the energy as a function of the coefficients
        energy = self.energy_baseline
        stretch_energy = 0.0
        bend_energy = 0.0
        torsional_energy = 0.0
        if verbosity >= 3:
            print("Evaluating FF energy:")

        if coordinates is None:
            coordinates = self.ff_coordinates()
        if coefficients is None:
            coefficients = self.ff_coefficients()

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

            stretch_energy += stretch.energy(dist,
                                             alternative_k=coefficients[idx])
        if verbosity >= 3:
            print("Total stretch energy: {: .5f}".format(stretch_energy))
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
            this_bend_energy = bend.energy(theta, d_bond_1, d_bond_2,
                                  alternative_k=coefficients[idx + offset])
            bend_energy += this_bend_energy

        if verbosity >= 3:
            print("Total bend energy: {: 7.3E}".format(bend_energy))

        offset += len(self.bends)
        for idx, torsion in enumerate(self.torsions):
            atom1_x_coord = coordinates[torsion.atom1 * 3]
            atom1_y_coord = coordinates[torsion.atom1 * 3 + 1]
            atom1_z_coord = coordinates[torsion.atom1 * 3 + 2]
            atom2_x_coord = coordinates[torsion.atom2 * 3]
            atom2_y_coord = coordinates[torsion.atom2 * 3 + 1]
            atom2_z_coord = coordinates[torsion.atom2 * 3 + 2]
            atom3_x_coord = coordinates[torsion.atom3 * 3]
            atom3_y_coord = coordinates[torsion.atom3 * 3 + 1]
            atom3_z_coord = coordinates[torsion.atom3 * 3 + 2]
            atom4_x_coord = coordinates[torsion.atom4 * 3]
            atom4_y_coord = coordinates[torsion.atom4 * 3 + 1]
            atom4_z_coord = coordinates[torsion.atom4 * 3 + 2]

            # Calculate the distance between each pair of atoms
            x_coord_e1 = (atom1_x_coord - atom2_x_coord)
            y_coord_e1 = (atom1_y_coord - atom2_y_coord)
            z_coord_e1 = (atom1_z_coord - atom2_z_coord)
            end_1 = [x_coord_e1, y_coord_e1, z_coord_e1]
            d_bond_1 = math.sqrt(
                (x_coord_e1 ** 2) + (y_coord_e1 ** 2) + (z_coord_e1 ** 2))

            x_coord_b = (atom2_x_coord - atom3_x_coord)
            y_coord_b = (atom2_y_coord - atom3_y_coord)
            z_coord_b = (atom2_z_coord - atom3_z_coord)
            bridge = [x_coord_b, y_coord_b, z_coord_b]
            d_bond_2 = math.sqrt(
                (x_coord_b ** 2) + (y_coord_b ** 2) + (z_coord_b ** 2))

            x_coord_e2 = (atom3_x_coord - atom4_x_coord)
            y_coord_e2 = (atom3_y_coord - atom4_y_coord)
            z_coord_e2 = (atom3_z_coord - atom4_z_coord)
            end_2 = [x_coord_e2, y_coord_e2, z_coord_e2]
            d_bond_3 = math.sqrt(
                (x_coord_e2 ** 2) + (y_coord_e2 ** 2) + (z_coord_e2 ** 2))

            vnormal_1 = np.cross(end_1, bridge)
            vnormal_2 = np.cross(bridge, end_2)

            # Construct a set of orthogonal basis vectors to define
            # a frame with vnormal_2 as the x axis
            vcross = np.cross(vnormal_2, bridge)
            norm_vn2 = np.linalg.norm(vnormal_2)
            # norm_b = np.linalg.norm(bridge)
            norm_vc = np.linalg.norm(vcross)
            basis_vn2 = [vnormal_2[idx] / norm_vn2 for idx in range(3)]
            # basis_b = [bridge[idx] / norm_b for idx in range(3)]
            basis_cv = [vcross[idx] / norm_vc for idx in range(3)]

            # Find the signed angle betw. vnormal_1 and vnormal_2
            # in the new frame
            vn1_coord_n2 = np.dot(vnormal_1, basis_vn2)
            vn1_coord_vc = np.dot(vnormal_1, basis_cv)
            psi = math.atan2(vn1_coord_vc, vn1_coord_n2)

            torsional_energy += torsion.energy(psi, d_bond_1, d_bond_2,
                                               d_bond_3,
                                               alternative_k=coefficients[
                                                   idx + offset])
            # print(
            #     "psi {: 0.5f} ({: 0.5f}), d1 {: 0.5f}, d2 {: 0.5f}, d3"
            #     " {: 0.5f}, k {: 0.5f}, E {: 7.3E}".format(psi, math.degrees(psi), d_bond_1,d_bond_2,d_bond_3,coefficients[idx + offset],torsional_energy))

        if verbosity >= 3:
            print("Total torsional energy: {: 7.3E}".format(torsional_energy))

        if verbosity >= 3:
            print("Total FF energy: {: .5f}".format(
                energy + stretch_energy + bend_energy + torsional_energy))
        return energy + stretch_energy + bend_energy + torsional_energy

    def ff_diff_qm_ff_hessian(self, coefficients=None, coordinates=None):

        # if coordinates is None:
        #     coordinates = self.coordinates()
        if coefficients is None:
            coefficients = self.ff_coefficients()

        # Calculate the force field hessian
        hessian_from_ff = self.ff_hessian(coefficients=coefficients,
                                          coordinates=coordinates)
        # Calculate the difference between qm and ff hessian, square the
        # result and sum over all elements

        difference = np.subtract(self.qm_hessian,hessian_from_ff)
        difference = np.square(difference)
        difference = np.sum(difference)

        # print("Difference: ", difference)

        return difference

    def ff_optimise_coefficients(self, starttime=None, verbosity=0):
        if verbosity >= 2:
            print("\nOptimising coefficients by fitting to the QM Hessian.")
        # Optimise until we find a set of coefficients that has no
        # linear dependencies
        while True:
            if verbosity >= 3:
                print("\nInitial coefficients:")
                coeff_string = "{} Stretches:\n".format(len(self.stretches))
                str_coeff_string = ""
                for stretch in self.stretches:
                    str_coeff_string += " {: .4f}".format(stretch.k_str)
                    if len(str_coeff_string) >= 70:
                        str_coeff_string += "\n"
                        coeff_string += str_coeff_string
                        str_coeff_string = ""
                coeff_string += str_coeff_string
                b_coeff_string = ""
                if len(self.bends) != 0:
                    coeff_string += "\n{} Bends:\n".format(len(self.bends))
                for bend in self.bends:
                    b_coeff_string += " {: .4f}".format(bend.k_bend)
                    if len(b_coeff_string) >= 70:
                        b_coeff_string += "\n"
                        coeff_string += b_coeff_string
                        b_coeff_string = ""
                coeff_string += b_coeff_string
                t_coeff_string = ""
                if len(self.torsions) != 0:
                    coeff_string += "\n{} Torsions:\n".format(len(self.torsions))
                for bend in self.torsions:
                    t_coeff_string += " {: .4f}".format(bend.k_tors)
                    if len(t_coeff_string) >= 70:
                        t_coeff_string += "\n"
                        coeff_string += t_coeff_string
                        t_coeff_string = ""
                coeff_string += t_coeff_string
                print(coeff_string)

            # Store coefficients in separate array for optimisation
            coeff_to_opt = np.array(self.ff_coefficients())
            # Store boundary condition in a list
            boundaries = [(0,np.Infinity)] * len(coeff_to_opt)
            # Try SLSQP algorithm first
            opt_coeff = scipy.optimize.minimize(
                self.ff_diff_qm_ff_hessian, coeff_to_opt, bounds = boundaries,
                method = "SLSQP", options = {'eps': 0.001})
            print(msg_timestamp("Optimisation finished.", starttime=starttime))
            if opt_coeff.success:
                if verbosity >= 2:
                    print("\nFirst optimisation attempt (SLSQP) converged!")
                    if verbosity >= 3:
                        print(" in {} iterations".format(opt_coeff.nit))
                        print(" Δ(H_QM,H_FF) = {: .4f}".format(opt_coeff.fun))
            else:
                if verbosity >= 2:
                    print("\nFirst optimisation attempt (SLSQP) failed")
                    print(" number of iterations {}".format(opt_coeff.nit))
                    print(" Δ(H_QM,H_FF) = {: .4f}".format(opt_coeff.fun))
                    print(" need to try again...")
                # In case of failure, try the L-BFGS-B algorithm next
                opt_coeff = scipy.optimize.minimize(
                    self.ff_diff_qm_ff_hessian, opt_coeff.x,
                    method="L-BFGS-B", bounds=boundaries,
                    options={'eps': 0.01})
                print(msg_timestamp("Optimisation finished.", starttime=starttime))
                if opt_coeff.success:
                    if verbosity >= 2:
                        print("\nSecond optimisation attempt (L-BFGS) converged!")
                        if verbosity >= 3:
                            print(" in {} iterations".format(opt_coeff.nit))
                            print(" Δ(H_QM,H_FF) = {: .4f}".format(
                                opt_coeff.fun))
                else:
                    if verbosity >= 2:
                        print(
                            "\nSecond optimisation attempt (L-BFGS) also failed")
                        print(" number of iterations {}".format(opt_coeff.nit))
                        print(" Δ(H_QM,H_FF) = {: .4f}".format(opt_coeff.fun))
                        print(" we're still going to use those"
                              " coefficients though...")

            # Checking for negative coefficients (i.e. linear dependencies)
            has_linear_dependency = False
            bend_offset = len(self.stretches)
            torsion_offset = len(self.stretches) + len(self.bends)
            delete_stretches = []
            delete_bends = []
            delete_torsions = []
            for idx, stretch in enumerate(self.stretches):
                if opt_coeff.x[idx] <= 0.0:
                    if verbosity >= 3:
                        print(" stretching coefficient {} is negative"
                              " ({: .4f})".format(idx + 1, opt_coeff.x[idx]))
                    has_linear_dependency = True
                    delete_stretches.append(idx)
            if len(delete_stretches) > 0:
                for idx in reversed(delete_stretches):
                    del self.stretches[idx]
            for idx, bend in enumerate(self.bends):
                if opt_coeff.x[bend_offset + idx] <= 0.0:
                    if verbosity >= 3:
                        print(
                            " bending coefficient {} is negative"
                            " ({: .4f})".format(idx + 1, opt_coeff.x[
                                bend_offset + idx]))
                    has_linear_dependency = True
                    delete_bends.append(idx)
            if len(delete_bends) > 0:
                for idx in reversed(delete_bends):
                    del self.bends[idx]
            for idx, torsion in enumerate(self.torsions):
                if opt_coeff.x[torsion_offset + idx] <= 0.0:
                    if verbosity >= 3:
                        print(
                            " torsion coefficient {} is negative"
                            " ({: .4f})".format(idx + 1, opt_coeff.x[
                                torsion_offset + idx]))
                    has_linear_dependency = True
                    delete_torsions.append(idx)
            if len(delete_torsions) > 0:
                for idx in reversed(delete_torsions):
                    del self.torsions[idx]
            if has_linear_dependency:
                opt_coeff = [item for item in opt_coeff.x if item > 0.0]
                self.ff_set_coefficients(opt_coeff)
                if verbosity >= 2:
                    print(" deleting linear dependencies and re-optimising")
            else:
                self.ff_set_coefficients(opt_coeff.x)
                break

        if verbosity >= 3:
            print("\nSetting FF to these final coefficients:")
            coeff_string = "{} Stretches:\n".format(len(self.stretches))
            str_coeff_string = ""
            for stretch in self.stretches:
                str_coeff_string += " {: .4f}".format(stretch.k_str)
                if len(str_coeff_string) >= 70:
                    str_coeff_string += "\n"
                    coeff_string += str_coeff_string
                    str_coeff_string = ""
            coeff_string += str_coeff_string
            b_coeff_string = ""
            if len(self.bends) != 0:
                coeff_string += "\n{} Bends:\n".format(len(self.bends))
            for bend in self.bends:
                b_coeff_string += " {: .4f}".format(bend.k_bend)
                if len(b_coeff_string) >= 70:
                    b_coeff_string += "\n"
                    coeff_string += b_coeff_string
                    b_coeff_string = ""
            coeff_string += b_coeff_string
            t_coeff_string = ""
            if len(self.torsions) != 0:
                coeff_string += "\n{} Torsions:\n".format(len(self.torsions))
            for bend in self.torsions:
                t_coeff_string += " {: .4f}".format(bend.k_tors)
                if len(t_coeff_string) >= 70:
                    t_coeff_string += "\n"
                    coeff_string += t_coeff_string
                    t_coeff_string = ""
            coeff_string += t_coeff_string
            print(coeff_string)

    def ff_optimise_geometry(self, verbosity=0):
        if verbosity >= 2:
            print("\nOptimising geometry of molecule:")
        initial_coords = np.array(self.ff_coordinates())

        opt_coord = scipy.optimize.minimize(
            self.ff_energy, initial_coords,
            method="L-BFGS-B", options={'eps': 0.001})
        if opt_coord.success:
            if verbosity >= 2:
                print("\nFirst optimisation attempt (L-BFGS) converged!")
                if verbosity >= 3:
                    print(" in {} iterations".format(opt_coord.nit))
                    print(" E(FF) = {: .4f}".format(opt_coord.fun))
        else:
            if verbosity >= 2:
                print("\nFirst optimisation attempt (L-BFGS) failed")
                print(" number of iterations {}".format(opt_coord.nit))
                print(" E(FF) = {: .4f}".format(opt_coord.fun))
                print(" trying again...")
            # In case of failure, try the SLSQP algorithm next
            opt_coord = scipy.optimize.minimize(
                self.ff_energy, opt_coord.x,
                method="SLSQP", options={'eps': 0.001})
            if opt_coord.success:
                if verbosity >= 2:
                    print("\nSecond optimisation attempt (SLSQP) converged!")
                    if verbosity >= 3:
                        print(" in {} iterations".format(opt_coord.nit))
                        print(" E(FF) = {: .4f}".format(
                            opt_coord.fun))
            else:
                if verbosity >= 2:
                    print(
                        "\nSecond optimisation attempt (SLSQP) also failed")
                    print(" number of iterations {}".format(opt_coord.nit))
                    print(" E(FF) = {: .4f}".format(opt_coord.fun))
                    print(
                        " still going to use those coordinates though...")

        self.set_coordinates(opt_coord.x)

    def set_coordinates(self, coordinates) -> None:
        """
        Set all molecular coordinates at once
        :param coordinates: List of molecular xyz coordinates in Bohr
        :return: None
        """

        for idx, atom in enumerate(self.atoms):
            atom.set_x(coordinates[idx * 3])
            atom.set_y(coordinates[idx * 3 + 1])
            atom.set_z(coordinates[idx * 3 + 2])
