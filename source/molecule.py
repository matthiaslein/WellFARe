########################################################
# Molecule class and class methods to be defined below #
########################################################

from typing import Optional
import math
import numpy as np
from multiprocessing import Pool, cpu_count

from atom import Atom
from constants import symbol_to_mass, symbol_to_covalent_radius
from messages import *


class Molecule:
    """A molecule with a name, charge, multiplicity and a list of atoms"""

    def __init__(self, name: str, charge: Optional[int] = 0,
                 multiplicity: Optional[int] = 1) -> None:
        """
        Creates a named Molecule with a charge, a multiplicity and, initially,
        no atoms. The charge is to be understood as the difference between the
        sum of nuclear charges and the number of electrons. The multiplicity is
        automatically set to the lowest possible value (1 or 2). There are also
        lists of bonds, angles and dihedrals (which are empty upon creation).
        The cartesian coordinates of the molecule are inherited from the
        individual cartesian coordinates of the atoms that make up the
        molecule (and are stored in self.atoms).

        :param name: The name of the molecule.
        :param charge: The overall molecular charge.
        :param multiplicity: The spin multiplicity.
        """

        self.name = name
        self.charge = charge
        self.mult = multiplicity
        self.atoms = []  # Initially an empty list
        self.bonds = []  # Initially an empty list
        self.angles = []  # Initially an empty list
        self.dihedrals = []  # Initially an empty list
        self.qm_energy = 0.0  # Initially set to zero

    def __str__(self) -> str:
        """
        Return a string representation of this Molecule in this format:
        (NAME, CHARGE, MULT, (ATOM1, ATOM2, ...))

        :return:
        """

        res = ''
        for atom in self.atoms:
            res = res + str(atom) + ', '
        res = res[:-2]
        return '({0}, {1}, {2}, ({3}))'.format(self.name, self.charge,
                                               self.mult, res)

    def __repr__(self) -> str:
        """
        Return a string representation of this Molecule in this format:
        (NAME, CHARGE, MULT, (ATOM1, ATOM2, ...))

        :return:
        """

        res = ''
        for atom in self.atoms:
            res = res + str(atom) + ', '
        res = res[:-2]
        return '({0}, {1}, {2}, ({3}))'.format(self.name, self.charge,
                                               self.mult, res)

    def mass(self) -> float:
        """

        :return: Returns the molar mass as sum of atomic masses
        """

        mass = 0.0
        for atom in self.atoms:
            mass = mass + atom.mass

        return mass

    def num_atoms(self) -> int:
        """

        :return: Returns the number of atoms in the molecule
        """

        return int(len(self.atoms))

    def add_atom(self, a: Atom, verbosity: int = 0) -> None:
        """
        This method adds an atom to the molecule. The molecule's multiplicity
        will be reset to the smallest possible value (singlet or doublet).

        :param a: An Atom object to be added to the molecule.
        :param verbosity: Print output about the action or not (default: no).
        :return: None
        """

        self.atoms.append(a)
        nucchg = 0
        for i in self.atoms:
            nucchg = nucchg + i.charge
        if (nucchg - self.charge) % 2 != 0:
            self.mult = 2
        else:
            self.mult = 1
        if verbosity >= 1:
            print(" adding {:<3} {: 13.8f} {: 13.8f} {: 13.8f} to {}".format(
                a.symbol(), a.xpos(), a.ypos(), a.zpos(), self.name))

    def ch_atom(self, n: int, at: Atom) -> None:
        """
        Exchanges the nth atom of the Molecule with a new atom.

        :param n: number of the atom to be exchanged.
        :param at: Atom object that replaces the old atom.
        :return: None
        """

        self.atoms[n] = at

    def mov_atom(self, n: int, x: float, y: float, z: float) -> None:
        """
        Move the nth atom to cartesian coordinates x, y, z.
        Coordinates are always given in Ångströms.

        :param n: number of the atom to be moved.
        :param x: target x-coordinate.
        :param y: target y-coordinate.
        :param z: target z-coordinate.
        :return: None
        """

        self.atoms[n].set_x(x)
        self.atoms[n].set_y(y)
        self.atoms[n].set_z(z)

    def add_bond(self, a: int, b: int) -> None:
        """
        Adds a bond between atoms a and b to the list of bonds. The indices a
        and b will be reordered to make sure that a < b and if the new bond is
        already present in the list of bonds, no second one will be added.

        :param a: First atom connected by the new bond.
        :param b: Second atom connected by the new bond.
        :return: None
        """

        # Make sure a < b
        if a < b:
            c = a
            d = b
        else:
            c = b
            d = a

        # Check if the bond already exists
        exists = False
        for i in self.bonds:
            if i == [c, d]:
                exists = True

        # Append bond to list if doesn't exist and is plausible
        if exists is False and a >= 0 and b >= 0 and a <= len(
                self.atoms) and b <= len(self.atoms) and c != d:
            self.bonds.append([c, d])

    def del_bond(self, a: int, b: int) -> None:
        """
        Deletes the bond between atoms a and b from the list of bonds.

        :param a: First atom connected by the bond.
        :param b: Second atom connected by the bond.
        :return: None
        """

        # Make sure a < b
        if a < b:
            c = a
            d = b
        else:
            c = b
            d = a

        # Check if the bond actually exists
        exists = False
        for i in self.bonds:
            if i == [c, d]:
                exists = True

        # Remove if it does
        if exists is True:
            self.bonds.remove([c, d])

    def add_angle(self, a: int, b: int, c: int) -> None:
        """
        Adds an angle between atoms a, b and c to the list of angles.

        :param a: The first atom that defines the angle.
        :param b: The second (i.e. middle) atom that defines the angle.
        :param c: The third atom that defines the angle.
        :return: None
        """

        # Check if the angle already exists
        exists = False
        for i in self.angles:
            if i == [a, b, c]:
                exists = True

        # Append angle to list if doesn't exist and is plausible.
        if exists is False and a >= 0 and b >= 0 and c >= 0 and \
                a <= len(self.atoms) and b <= len(self.atoms) and \
                c <= len(self.atoms) and a != b and a != c and b != c:
            self.angles.append([a, b, c])

    def add_dihedral(self, a: int, b: int, c: int, d: int) -> None:
        """
        Adds a dihedral between atoms a, b, c and d to the list of dihedrals.

        :param a: First atom defining the dihedral. Ordinarily bound to atom b.
        :param b: Second atom defining the dihedral. Ordinarily bound to atoms
                   a and b.
        :param c: Third atom defining the dihedral. Ordinarily bound to atoms
                   b and d.
        :param d: Last atom defining the dihedral. Ordinarily bound to atom c.
        :return: None
        """

        # Check if the dihedral already exists
        exists = False
        for i in self.dihedrals:
            if i == [a, b, c, d]:
                exists = True

        # Append dihedral to list if doesn't exist and is plausible
        if exists is False and a >= 0 and b >= 0 and c >= 0 and d >= 0 and \
                a <= len(self.atoms) and b <= len(self.atoms) and \
                c <= len(self.atoms) and d <= len(self.atoms) and \
                a != b and a != c and a != d and b != c and b != d and c != d:
            self.dihedrals.append([a, b, c, d])

    def atm_symbol(self, i: int) -> str:
        """
        Report the atomic symbol of atom i.

        :param i: Atom to locate.
        :return: The atom's atomic symbol
        """
        return self.atoms[i].symbol()

    def atm_pos_x(self, i: int) -> float:
        """
        Report the x-coordinate of atom i in Ångströms.

        :param i: Atom to locate.
        :return: The atoms x-coordinate in Ångströms.
        """
        return self.atoms[i].coord[0]

    def atm_pos_y(self, i: int) -> float:
        """
        Report the y-coordinate of atom i in Ångströms.

        :param i: Atom to locate.
        :return: The atoms y-coordinate in Ångströms.
        """
        return self.atoms[i].coord[1]

    def atm_pos_z(self, i: int) -> float:
        """
        Report the z-coordinate of atom i in Ångströms.

        :param i: Atom to locate.
        :return: The atoms z-coordinate in Ångströms.
        """
        return self.atoms[i].coord[2]

    def atm_atm_dist(self, i: int, j: int) -> float:
        """
        Report the distance between atoms i and j in Ångström.

        :param i: First atom for distance measurement.
        :param j: Second atom for distance measurement.
        :return: The distance between atoms i and j in Ångström.
        """

        distance = (self.atm_pos_x(i) - self.atm_pos_x(j)) * (
                self.atm_pos_x(i) - self.atm_pos_x(j))
        distance = distance + (self.atm_pos_y(i) - self.atm_pos_y(j)) * (
                self.atm_pos_y(i) - self.atm_pos_y(j))
        distance = distance + (self.atm_pos_z(i) - self.atm_pos_z(j)) * (
                self.atm_pos_z(i) - self.atm_pos_z(j))

        return math.sqrt(distance)

    def bond_dist(self, i: int) -> float:
        """
        Report the distance between two atoms in the bonds list in Ångström.

        :param i: Index of the bond from the list of bonds..
        :return: The distance between the in Ångström.
        """

        atom1 = self.bonds[i][0]
        atom2 = self.bonds[i][1]
        distance = self.atm_atm_dist(atom1, atom2)

        return distance

    def atm_atm_atm_angle(self, i: int, j: int, k: int) -> float:
        """
        Report the angle between three atoms in radians.

        :param i: First atom for angle measurement.
        :param j: Second atom for angle measurement.
        :param k: Third atom for angle measurement.
        :return: The numerical value of the angle in radians.
        """

        # Calculate the distance between each pair of atoms
        d_bond_1 = self.atm_atm_dist(i, j)
        d_bond_2 = self.atm_atm_dist(j, k)
        d_non_bond = self.atm_atm_dist(i, k)

        # Use those distances and the cosine rule to calculate bond angle theta
        numerator = d_bond_1 ** 2 + d_bond_2 ** 2 - d_non_bond ** 2
        denominator = 2 * d_bond_1 * d_bond_2
        argument = numerator / denominator
        # This safety check was necessary because of a bug elsewhere once...
        # if argument > 1.0:
        #     argument = 1.0
        # elif argument < -1.0:
        #     argument = -1.0
        theta = np.arccos(argument)

        return theta

    def bond_angle(self, i: int) -> float:
        """
        Report the angle described by three atoms in the bonds list in radians.

        :param i: Index of the bond angle from the list of angles.
        :return: The numerical value of the angle in radians.
        """
        atom1 = self.angles[i][0]
        atom2 = self.angles[i][1]
        atom3 = self.angles[i][2]
        angle = self.atm_atm_atm_angle(atom1, atom2, atom3)

        return angle

    def atm_atm_atm_atm_dihedral(self, i: int, j: int, k: int,
                                 l: int) -> float:
        """
        Report the dihedral angle described by a set of four atoms in radians.


        :param i: First atom for angle measurement.
        :param j: Second atom for angle measurement.
        :param k: Third atom for angle measurement.
        :param l: Fourth atom for angle measurement.
        :return: The numerical value of the dihedral angle in radians.
        """

        # Calculate the vectors lying along bonds, and their cross products
        atom_e1 = self.atoms[i]
        atom_b1 = self.atoms[j]
        atom_b2 = self.atoms[k]
        atom_e2 = self.atoms[l]
        end_1 = [atom_e1.coord[idx] - atom_b1.coord[idx] for idx in range(3)]
        bridge = [atom_b1.coord[idx] - atom_b2.coord[idx] for idx in range(3)]
        end_2 = [atom_b2.coord[idx] - atom_e2.coord[idx] for idx in range(3)]
        vnormal_1 = np.cross(end_1, bridge)
        vnormal_2 = np.cross(bridge, end_2)

        # Construct a set of orthogonal basis vectors to define a frame with
        # vnormal_2 as the x axis
        vcross = np.cross(vnormal_2, bridge)
        norm_vn2 = np.linalg.norm(vnormal_2)
        # norm_b = np.linalg.norm(bridge)
        norm_vc = np.linalg.norm(vcross)
        basis_vn2 = [vnormal_2[idx] / norm_vn2 for idx in range(3)]
        # basis_b = [bridge[idx] / norm_b for idx in range(3)]
        basis_cv = [vcross[idx] / norm_vc for idx in range(3)]

        # Find the signed angle betw. vnormal_1 and vnormal_2 in the new frame
        vn1_coord_n2 = np.dot(vnormal_1, basis_vn2)
        vn1_coord_vc = np.dot(vnormal_1, basis_cv)
        psi = math.atan2(vn1_coord_vc, vn1_coord_n2)

        return psi

    def dihedral_angle(self, i: int) -> float:
        """
        Report the dihedral angle described by a set of four atoms in the list
        of dihedrals in radians.

        :param i: Index of the dihedral angle in the list of dihedrals.
        :return: The numerical value of the angle in radians.
        """

        atom1 = self.dihedrals[i][0]
        atom2 = self.dihedrals[i][1]
        atom3 = self.dihedrals[i][2]
        atom4 = self.dihedrals[i][3]

        dihedral = self.atm_atm_atm_atm_dihedral(atom1, atom2, atom3, atom4)

        return dihedral

    def set_mult(self, multiplicity: int) -> None:
        """
        Set the multiplicity of the molecule to M

        :param multiplicity: Value of the multiplicity (i.e. 1 for singlet,
                             2 for doublet, ...)
        :return: None
        """

        self.mult = multiplicity

    def print_mol(self, output: str = "cart", comment: Optional[str] = None,
                  file: Optional[str] = None) -> str:
        """
        Returns a string containing the molecular coordinates, which can also
        be written to a file.

        If no input at all is given, we use the "name" of the molecule as the
        comment line. If a file name is given, but no comment, we use the
        file name as comment line. If a comment is specified, we use that as
        the comment line (Note: if an empty string is desired, then "" must
        be submitted as comment)

        :param output: Style of the output:
                         "cart" for plain cartesian coordinates.
                         "xyz" for the xyz-file format (i.e. with # of atoms).
                         "gauss" for Gaussian style.
                         "gamess" for Gamess style.
        :param comment: A comment (string) that can be printed into the output
        :param file: The filename of a file that is created and written to.
        :return: The string with the molecular coordinates.
        """

        if comment is None and file is None:
            commentline = self.name
        elif comment is None and file is not None:
            commentline = file
        else:
            commentline = comment

        s = ""
        if output == "cart":
            # Plain cartesian coordinates - nothing else
            for i in self.atoms:
                t = "{:<3} {: 13.8f} {: 13.8f} {: 13.8f}\n".format(i.symbol(),
                                                                   i.xpos(),
                                                                   i.ypos(),
                                                                   i.zpos())
                s = s + t
        elif output == "xyz":
            # xyz-file format. Number of atoms first, comment line,
            # then coordinates
            s = str(self.num_atoms()) + "\n" + commentline + "\n"
            for i in self.atoms:
                t = "{:<3} {: 13.8f} {: 13.8f} {: 13.8f}\n".format(i.symbol(),
                                                                   i.xpos(),
                                                                   i.ypos(),
                                                                   i.zpos())
                s = s + t
        elif output == "gamess":
            # xyz coordinates in Gamess format
            s = " $DATA\n" + commentline + "\nC1\n"
            for i in self.atoms:
                t = "{:<3} {:<3d} {: 13.8f} {: 13.8f} {: 13.8f}\n".format(
                    i.symbol(),
                    i.charge,
                    i.xpos(),
                    i.ypos(),
                    i.zpos())
                s = s + t
            s = s + " $END\n"
        elif output == "gauss":
            # xyz coordinates in Gaussian format
            s = "\n" + commentline + "\n\n" + str(self.charge) + " " + str(
                self.mult) + "\n"
            for i in self.atoms:
                if i.mass == symbol_to_mass[i.symbol()]:
                    t = "{:<3} {: 13.8f} {: 13.8f} {: 13.8f}\n".format(
                        i.symbol(),
                        i.xpos(),
                        i.ypos(),
                        i.zpos())
                else:
                    t = "{} {: 13.8f} {: 13.8f} {: 13.8f}\n".format(
                        i.symbol() + "(Iso=" + str(i.mass) + ")",
                        i.xpos(),
                        i.ypos(),
                        i.zpos())
                s = s + t
            s = s + "\n"

        if file is not None:
            # Try filename for writability first
            try:
                f = open(file + "." + output, 'w')
                f.close()
            except OSError:
                msg_program_error("Cannot open " + file + " for writing.")
            # Then write the string we created above
            with open(file + "." + output, "w") as text_file:
                print("{}".format(s), file=text_file)

        return s


def build_molecular_dihedrals(molecule, verbosity=0):
    """

    :param molecule: The molecule to be operated on
    :param verbosity: How chatty this method is
    :return: A list of the values of the dihedral angles in the same order as
             the dihedrals that have been identified
    """

    list_of_dihedrals = []

    if len(molecule.angles) > 1:
        if verbosity >= 2:
            print("\nLooking for dihedrals in WellFARe molecule: ",
                  molecule.name)
        for i in range(0, len(molecule.angles)):
            for j in range(i + 1, len(molecule.angles)):
                if molecule.angles[i][1] == molecule.angles[j][0] and \
                        molecule.angles[i][2] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][0],
                                          molecule.angles[i][1],
                                          molecule.angles[i][2],
                                          molecule.angles[j][2])
                    list_of_dihedrals.append(
                        math.degrees(molecule.dihedral_angle(
                            len(molecule.dihedrals) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({: 7.2f}°)".format(
                                molecule.atoms[molecule.angles[i][0]].symbol(),
                                molecule.angles[i][0],
                                molecule.atoms[molecule.angles[i][1]].symbol(),
                                molecule.angles[i][1],
                                molecule.atoms[molecule.angles[i][2]].symbol(),
                                molecule.angles[i][2],
                                molecule.atoms[molecule.angles[j][2]].symbol(),
                                molecule.angles[j][2],
                                list_of_dihedrals[-1]))
                if molecule.angles[i][1] == molecule.angles[j][2] and \
                        molecule.angles[i][2] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][0],
                                          molecule.angles[i][1],
                                          molecule.angles[i][2],
                                          molecule.angles[j][0])
                    list_of_dihedrals.append(
                        math.degrees(molecule.dihedral_angle(
                            len(molecule.dihedrals) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({: 7.2f}°)".format(
                                molecule.atoms[molecule.angles[i][0]].symbol(),
                                molecule.angles[i][0],
                                molecule.atoms[molecule.angles[i][1]].symbol(),
                                molecule.angles[i][1],
                                molecule.atoms[molecule.angles[i][2]].symbol(),
                                molecule.angles[i][2],
                                molecule.atoms[molecule.angles[j][0]].symbol(),
                                molecule.angles[j][0],
                                list_of_dihedrals[-1]))
                if molecule.angles[i][1] == molecule.angles[j][0] and \
                        molecule.angles[i][0] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][2],
                                          molecule.angles[j][0],
                                          molecule.angles[j][1],
                                          molecule.angles[j][2])
                    list_of_dihedrals.append(
                        math.degrees(molecule.dihedral_angle(
                            len(molecule.dihedrals) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({: 7.2f}°)".format(
                                molecule.atoms[molecule.angles[i][2]].symbol(),
                                molecule.angles[i][2],
                                molecule.atoms[molecule.angles[j][0]].symbol(),
                                molecule.angles[j][0],
                                molecule.atoms[molecule.angles[j][1]].symbol(),
                                molecule.angles[j][1],
                                molecule.atoms[molecule.angles[j][2]].symbol(),
                                molecule.angles[j][2],
                                list_of_dihedrals[-1]))
                if molecule.angles[i][1] == molecule.angles[j][2] and \
                        molecule.angles[i][0] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][2],
                                          molecule.angles[j][2],
                                          molecule.angles[j][1],
                                          molecule.angles[j][0])
                    list_of_dihedrals.append(
                        math.degrees(molecule.dihedral_angle(
                            len(molecule.dihedrals) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({: 7.2f}°)".format(
                                molecule.atoms[molecule.angles[i][2]].symbol(),
                                molecule.angles[i][2],
                                molecule.atoms[molecule.angles[j][2]].symbol(),
                                molecule.angles[j][2],
                                molecule.atoms[molecule.angles[j][1]].symbol(),
                                molecule.angles[j][1],
                                molecule.atoms[molecule.angles[j][0]].symbol(),
                                molecule.angles[j][0],
                                list_of_dihedrals[-1]))
    else:
        if verbosity >= 2:
            print("\nNot looking for dihedrals in WellFARe molecule: ",
                  molecule.name)
            print(
                "(because there are fewer than two angles"
                " identified in the molecule)")
    return list_of_dihedrals


def build_molecular_angles(molecule, verbosity=0):
    """

    :param molecule: The molecule to be operated on
    :param verbosity: How chatty this method is.
    :return: A list with the values of all bond angles in degrees in the same
             order as the bond angles.
    """

    list_of_bond_angles = []

    # Now that we know where the bonds are, find angles
    if len(molecule.bonds) > 1:
        if verbosity >= 2:
            print("\nLooking for angles in WellFARe molecule: ", molecule.name)
        for i in range(0, len(molecule.bonds)):
            for j in range(i + 1, len(molecule.bonds)):
                if molecule.bonds[i][0] == molecule.bonds[j][0]:
                    molecule.add_angle(molecule.bonds[i][1],
                                       molecule.bonds[i][0],
                                       molecule.bonds[j][1])
                    list_of_bond_angles.append(
                        math.degrees(molecule.bond_angle(
                            len(molecule.angles) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({:6.2f}°)".format(
                                molecule.atoms[molecule.bonds[i][1]].symbol(),
                                molecule.bonds[i][1],
                                molecule.atoms[molecule.bonds[i][0]].symbol(),
                                molecule.bonds[i][0],
                                molecule.atoms[molecule.bonds[j][1]].symbol(),
                                molecule.bonds[j][1],
                                list_of_bond_angles[-1]))
                if molecule.bonds[i][0] == molecule.bonds[j][1]:
                    molecule.add_angle(molecule.bonds[i][1],
                                       molecule.bonds[i][0],
                                       molecule.bonds[j][0])
                    list_of_bond_angles.append(
                        math.degrees(molecule.bond_angle(
                            len(molecule.angles) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({:6.2f}°)".format(
                                molecule.atoms[molecule.bonds[i][1]].symbol(),
                                molecule.bonds[i][1],
                                molecule.atoms[molecule.bonds[i][0]].symbol(),
                                molecule.bonds[i][0],
                                molecule.atoms[molecule.bonds[j][0]].symbol(),
                                molecule.bonds[j][0],
                                list_of_bond_angles[-1]))
                if molecule.bonds[i][1] == molecule.bonds[j][0]:
                    molecule.add_angle(molecule.bonds[i][0],
                                       molecule.bonds[i][1],
                                       molecule.bonds[j][1])
                    list_of_bond_angles.append(
                        math.degrees(molecule.bond_angle(
                            len(molecule.angles) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({:6.2f}°)".format(
                                molecule.atoms[molecule.bonds[i][0]].symbol(),
                                molecule.bonds[i][0],
                                molecule.atoms[molecule.bonds[i][1]].symbol(),
                                molecule.bonds[i][1],
                                molecule.atoms[molecule.bonds[j][1]].symbol(),
                                molecule.bonds[j][1],
                                list_of_bond_angles[-1]))
                if molecule.bonds[i][1] == molecule.bonds[j][1]:
                    molecule.add_angle(molecule.bonds[i][0],
                                       molecule.bonds[i][1],
                                       molecule.bonds[j][0])
                    list_of_bond_angles.append(
                        math.degrees(molecule.bond_angle(
                            len(molecule.angles) - 1)))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}), {:<3} ({:3d})"
                            " and {:<3} ({:3d}) ({:6.2f}°)".format(
                                molecule.atoms[molecule.bonds[i][0]].symbol(),
                                molecule.bonds[i][0],
                                molecule.atoms[molecule.bonds[i][1]].symbol(),
                                molecule.bonds[i][1],
                                molecule.atoms[molecule.bonds[j][0]].symbol(),
                                molecule.bonds[j][0],
                                list_of_bond_angles[-1]))
    else:
        if verbosity >= 2:
            print("\nNot looking for angles in WellFARe molecule: ",
                  molecule.name)
            print(
                "(because there are fewer than 2 bonds"
                " identified in the molecule)")
    return list_of_bond_angles


def batch_compare_distances(molecule, list_of_pairs, tolerance, verbosity = 0):
    results = []
    for i in list_of_pairs:
        distance = molecule.atm_atm_dist(i[0], i[1])
        if distance <= (
                symbol_to_covalent_radius[molecule.atm_symbol(i[0])] +
                symbol_to_covalent_radius[
                    molecule.atm_symbol(i[1])]) * tolerance:
            if verbosity >= 3:
                print(
                    " {:<3} ({:3d}) and {:<3} ({:3d})"
                    " (Distance: {:.3f} Å)".format(
                        molecule.atm_symbol(i[0]), i[0],
                        molecule.atm_symbol(i[1]), i[1],
                        distance))
            results.append([i[0],i[1]])
    return results

def build_bond_orders(molecule, bo=None, verbosity=0, bondcutoff=0.45,
                      distfactor=1.3, deletefirst=True, cpu_number=1):
    """
    This method populates the list of bonds of a given molecule.

    :param molecule: The molecule to operate on
    :param bo: an n by n matrix (i.e. array) that holds qm bond orders (if not
               available, bonds will be built from distances alone).
    :param verbosity: How talkative this method is, default is silence.
    :param bondcutoff: Cutoff value for the qm bond orders below which no bond
                       will be assigned between two atoms.
    :param distfactor: Cutoff value for a multiple of the sum of covalent radii
                       above which no bond will be assigned between two atoms.
    :param deletefirst: Should the list of bonds be deleted, before a new list
                        is built?
    :return: A list of all bond lengths in Å in the same order as the bonds.
    """

    if bo is None:
        bo = []
    list_of_bond_lengths = []

    if deletefirst is True:
        molecule.bonds = []

    # Test if we actually have Quantum Mechanical (qm) Bond orders
    if np.count_nonzero(bo) != 0 and molecule.num_atoms() != 1:
        if verbosity >= 2:
            print("\nLooking for bonds in WellFARe molecule: ",
                  molecule.name)
            print("(using bond orders with a cutoff of {: .2f}):".format(
                bondcutoff))
        for i in range(0, molecule.num_atoms()):
            for j in range(i + 1, molecule.num_atoms()):
                if bo[i][j] >= bondcutoff:
                    molecule.add_bond(i, j)
                    list_of_bond_lengths.append(
                        molecule.atm_atm_dist(i, j))
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:4d}) and {:<3} ({:4d})"
                            " (Bond order: {: 6.3f}, length {:6.3f} Å)".format(
                                molecule.atoms[i].symbol(), i,
                                molecule.atoms[j].symbol(), j,
                                float(bo[i][j]), list_of_bond_lengths[-1]))
    # Only do bonds if there's more than one atom
    elif molecule.num_atoms() != 1:
        if verbosity >= 2:
            print("\nLooking for bonds in WellFARe molecule:", molecule.name)
            print(
                "(using covalent radii scaled by {: .2f}):".format(distfactor))
        pairs = []
        for i in range(0, molecule.num_atoms()):
            for j in range(i + 1, molecule.num_atoms()):
                pairs.append([i, j])
        chunks = [pairs[i:i + (len(pairs) // cpu_number)] for i in
                  range(0, len(pairs), (len(pairs) // cpu_number))]
        with Pool(processes=cpu_number) as p:
            res = [p.apply_async(batch_compare_distances, args=(molecule, i, distfactor, verbosity)) for i in chunks]
            results = [p.get() for p in res]
        for i in results:
            for j in i:
                molecule.add_bond(j[0],j[1])
    else:
        if verbosity >= 2:
            print("\nNot looking for bonds in WellFARe molecule:",
                  molecule.name)
            print("(because we only have one atom)")
    return list_of_bond_lengths


def main():
    # Create an example molecule and add some atoms
    example = Molecule("Example Molecule")
    example.add_atom(Atom("C", x=-0.63397128, y=0.57416267, z=0.000000))
    example.add_atom(Atom("H", x=-0.27731685, y=-0.43464733, z=0.000000))
    example.add_atom(Atom("H", x=-0.27729844, y=1.07856086, z=0.87365150))
    example.add_atom(Atom("H", x=-0.27729844, y=1.07856086, z=-0.87365150))
    example.add_atom(
        Atom("H", x=-1.70397128, y=0.57417585, z=0.00000000, mass=2.0))
    print(example.print_mol(output="gauss"))


if __name__ == '__main__':
    main()
