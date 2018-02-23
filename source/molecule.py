#############################################################################################################
# Molecule class and class methods to be defined below
#############################################################################################################

from typing import Optional
import math
import numpy as np

from atom import Atom
from constants import SymbolToMass


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
            print(" adding {:<3} {: .8f} {: .8f} {: .8f} to {}".format(
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

    def replaceHwithTetrahedral(self, h, replacement="C", addH=3,
                                orientation=0, ignoreWarning=False):
        """ (Molecule) -> NoneType

    Exchange the given hydrogen atom with an arbitrary other atom and, optionally, add up to 3
    hydrogen atoms in a tetrahedral geometry.
    Default behaviour is substitution by a CH₃ group. If only 1 or 2 hydrogen
    atoms are added, an additional parameter "orientation" is needed to decide where the new
    group "points".
    """
        # Safety check: h is indeed an H atom and has only one bond
        if self.atoms[h].symbol != "H":
            ProgramWarning("This is not a hydrogen atom!")
            if ignoreWarning == False:
                return
        position = -1
        for i in self.bonds:
            if i[0] == h and position == -1:
                position = i[1]
            elif i[1] == h and position == -1:
                position = i[0]
            elif i[0] == h and position != -1:
                ProgramWarning("This hydrogen atom has more than one bond!")
                position = i[1]
                if ignoreWarning == False:
                    return
            elif i[1] == h and position != -1:
                ProgramWarning("This hydrogen atom has more than one bond!")
                position = i[0]
                if ignoreWarning == False:
                    return
        if position == -1:
            ProgramWarning("This hydrogen atom has no bond!")
            return
        # Find one more atom that is bound to the atom in "position"
        position2 = -1
        for i in self.bonds:
            if i[0] == position and i[1] != h:
                position2 = i[1]
            if i[1] == position and i[0] != h:
                position2 = i[0]
        if position2 == -1:
            ProgramWarning("The C atom no seems to have only one bond!")
            return
        # Calculate the vector from C to H, scale it and put the replacement atom there
        xcompo = self.atoms[h].coord[0] - self.atoms[position].coord[0]
        ycompo = self.atoms[h].coord[1] - self.atoms[position].coord[1]
        zcompo = self.atoms[h].coord[2] - self.atoms[position].coord[2]
        # Scale this vector to the "right" length:
        # We choose 110% of the sum of vdW radii  as the length of a slightly elongated bond
        scale = (SymbolToRadius["C"] + SymbolToRadius[replacement]) * 1.1
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * scale
        ycompo = (ycompo / norm) * scale
        zcompo = (zcompo / norm) * scale
        # Calculate the position of the replacement atom
        xcompo = self.atoms[position].coord[0] + xcompo
        ycompo = self.atoms[position].coord[1] + ycompo
        zcompo = self.atoms[position].coord[2] + zcompo
        # Add this new atom
        self.add_atom(Atom(replacement, xcompo, ycompo, zcompo, 0.1))
        # And call it's number newN for future reference
        newAtom = self.num_atoms() - 1
        if addH != 0:
            # Setup the coordinates of the first hydrogen atom:
            # We start by constructing the vector from the newly created N atom
            # to the "old one" it is bonded to
            xcompo = self.atoms[position].coord[0] - self.atoms[newAtom].coord[
                0]
            ycompo = self.atoms[position].coord[1] - self.atoms[newAtom].coord[
                1]
            zcompo = self.atoms[position].coord[2] - self.atoms[newAtom].coord[
                2]
            # Scale this vector to the "right" length:
            # We choose 110% of the sum of vdW radii  as the length of a slightly elongated bond
            scale = (SymbolToRadius["H"] + SymbolToRadius[replacement]) * 1.1
            norm = math.sqrt(
                xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
            xcompo = (xcompo / norm) * scale
            ycompo = (ycompo / norm) * scale
            zcompo = (zcompo / norm) * scale
            # Next we determine the axis around which we want to rotate this vector:
            # We use the vectors from the new N atom to the C atom and the vector
            # from the N atom to the "second atom" we found that was also bonded
            # to the C atom.
            # Importantly, we keep its coordinates untouched once we have them,
            # so we can create the other two hydrogen atoms in the same way.
            crossPx = self.atoms[position2].coord[0] - \
                      self.atoms[newAtom].coord[0]
            crossPy = self.atoms[position2].coord[1] - \
                      self.atoms[newAtom].coord[1]
            crossPz = self.atoms[position2].coord[2] - \
                      self.atoms[newAtom].coord[2]
            cross = np.cross([crossPx, crossPy, crossPz],
                             [xcompo, ycompo, zcompo])
            # Then we build the rotation matrix
            L = (cross[0] ** 2) + (cross[1] ** 2) + (cross[2] ** 2)
            pi = np.radians(109.4712206)

            RotMatrix = []
            Rxx = (cross[0] ** 2 + (cross[1] ** 2 + cross[2] ** 2) * np.cos(
                pi)) / L
            Rxy = ((cross[0] * cross[1]) * (1 - np.cos(pi)) - cross[
                2] * np.sqrt(L) * np.sin(pi)) / L
            Rxz = ((cross[0] * cross[2]) * (1 - np.cos(pi)) + cross[
                1] * np.sqrt(L) * np.sin(pi)) / L

            Ryx = ((cross[0] * cross[1]) * (1 - np.cos(pi)) + cross[
                2] * np.sqrt(L) * np.sin(pi)) / L
            Ryy = (cross[1] ** 2 + (cross[0] ** 2 + cross[2] ** 2) * np.cos(
                pi)) / L
            Ryz = ((cross[1] * cross[2]) * (1 - np.cos(pi)) - cross[
                0] * np.sqrt(L) * np.sin(pi)) / L

            Rzx = ((cross[0] * cross[2]) * (1 - np.cos(pi)) - cross[
                1] * np.sqrt(L) * np.sin(pi)) / L
            Rzy = ((cross[1] * cross[2]) * (1 - np.cos(pi)) + cross[
                0] * np.sqrt(L) * np.sin(pi)) / L
            Rzz = (cross[2] ** 2 + (cross[0] ** 2 + cross[1] ** 2) * np.cos(
                pi)) / L

            RotMatrix.append([Rxx, Rxy, Rxz])
            RotMatrix.append([Ryx, Ryy, Ryz])
            RotMatrix.append([Rzx, Rzy, Rzz])

            RotMatrix = np.matrix(RotMatrix)

            vector = [xcompo, ycompo, zcompo]
            vector = np.matrix(vector)
            vector = RotMatrix.dot(np.matrix.transpose(vector))
            vector = np.array(vector).flatten().tolist()
            xcompo = vector[0]
            ycompo = vector[1]
            zcompo = vector[2]
            # Calculate the position of the new H atom
            xNew1 = self.atoms[newAtom].coord[0] + xcompo
            yNew1 = self.atoms[newAtom].coord[1] + ycompo
            zNew1 = self.atoms[newAtom].coord[2] + zcompo

            # Next, we re-use the rotated coordinates and rotate them further
            # by 120 degrees around the C-N bond
            cross[0] = self.atoms[position].coord[0] - \
                       self.atoms[newAtom].coord[0]
            cross[1] = self.atoms[position].coord[1] - \
                       self.atoms[newAtom].coord[1]
            cross[2] = self.atoms[position].coord[2] - \
                       self.atoms[newAtom].coord[2]
            # We build the second rotation matrix
            L = (cross[0] ** 2) + (cross[1] ** 2) + (cross[2] ** 2)
            pi = np.radians(120.0)

            RotMatrix = []
            Rxx = (cross[0] ** 2 + (cross[1] ** 2 + cross[2] ** 2) * np.cos(
                pi)) / L
            Rxy = ((cross[0] * cross[1]) * (1 - np.cos(pi)) - cross[
                2] * np.sqrt(L) * np.sin(pi)) / L
            Rxz = ((cross[0] * cross[2]) * (1 - np.cos(pi)) + cross[
                1] * np.sqrt(L) * np.sin(pi)) / L

            Ryx = ((cross[0] * cross[1]) * (1 - np.cos(pi)) + cross[
                2] * np.sqrt(L) * np.sin(pi)) / L
            Ryy = (cross[1] ** 2 + (cross[0] ** 2 + cross[2] ** 2) * np.cos(
                pi)) / L
            Ryz = ((cross[1] * cross[2]) * (1 - np.cos(pi)) - cross[
                0] * np.sqrt(L) * np.sin(pi)) / L

            Rzx = ((cross[0] * cross[2]) * (1 - np.cos(pi)) - cross[
                1] * np.sqrt(L) * np.sin(pi)) / L
            Rzy = ((cross[1] * cross[2]) * (1 - np.cos(pi)) + cross[
                0] * np.sqrt(L) * np.sin(pi)) / L
            Rzz = (cross[2] ** 2 + (cross[0] ** 2 + cross[1] ** 2) * np.cos(
                pi)) / L

            RotMatrix.append([Rxx, Rxy, Rxz])
            RotMatrix.append([Ryx, Ryy, Ryz])
            RotMatrix.append([Rzx, Rzy, Rzz])

            RotMatrix = np.matrix(RotMatrix)

            vector = [xcompo, ycompo, zcompo]
            vector = np.matrix(vector)
            vector = RotMatrix.dot(np.matrix.transpose(vector))
            vector = np.array(vector).flatten().tolist()
            xcompo = vector[0]
            ycompo = vector[1]
            zcompo = vector[2]
            # Calculate the position of the new H atom
            xNew2 = self.atoms[newAtom].coord[0] + xcompo
            yNew2 = self.atoms[newAtom].coord[1] + ycompo
            zNew2 = self.atoms[newAtom].coord[2] + zcompo

            # And then, we do this a third and last time...
            vector = [xcompo, ycompo, zcompo]
            vector = np.matrix(vector)
            vector = RotMatrix.dot(np.matrix.transpose(vector))
            vector = np.array(vector).flatten().tolist()
            xcompo = vector[0]
            ycompo = vector[1]
            zcompo = vector[2]
            # Calculate the position of the new H atom
            xNew3 = self.atoms[newAtom].coord[0] + xcompo
            yNew3 = self.atoms[newAtom].coord[1] + ycompo
            zNew3 = self.atoms[newAtom].coord[2] + zcompo

            if addH == 3:
                self.add_atom(Atom("H", xNew1, yNew1, zNew1, 0.1))
                self.add_atom(Atom("H", xNew2, yNew2, zNew2, 0.1))
                self.add_atom(Atom("H", xNew3, yNew3, zNew3, 0.1))
            elif addH == 2:
                # Add two of the new atoms according to the chosen orientation
                if orientation == 1:
                    self.add_atom(Atom("H", xNew1, yNew1, zNew1, 0.1))
                    self.add_atom(Atom("H", xNew3, yNew3, zNew3, 0.1))
                elif orientation == 2:
                    self.add_atom(Atom("H", xNew2, yNew2, zNew2, 0.1))
                    self.add_atom(Atom("H", xNew3, yNew3, zNew3, 0.1))
                else:
                    self.add_atom(Atom("H", xNew1, yNew1, zNew1, 0.1))
                    self.add_atom(Atom("H", xNew2, yNew2, zNew2, 0.1))
            elif addH == 1:
                # Add one of the new atoms according to the chosen orientation
                if orientation == 1:
                    self.add_atom(Atom("H", xNew2, yNew2, zNew2, 0.1))
                elif orientation == 2:
                    self.add_atom(Atom("H", xNew3, yNew3, zNew3, 0.1))
                else:
                    self.add_atom(Atom("H", xNew1, yNew1, zNew1, 0.1))

        # Delete the initial hydrogen atom h # EXTREMELY MESSY!!!
        del self.atoms[h]
        return

    def replaceHwithEthenyl(self, h, ignoreWarning=False):
        """ (Molecule) -> NoneType

    Exchange the given hydrogen atom by a "planar" CH2 group. This method requires the carbon atom whose
    hydrogen atom is being replaced to possess two more hydrogen atoms (i.e. it only works on
    methyl groups), because it will substitute the other two hydrogen atoms with a single h atom
    in the correct position, given by the sp2 hybridisation of the carbon atoms.
    """

        # Safety check: h is indeed an H atom and has only one bond to a C atom
        if self.atoms[h].symbol != "H":
            ProgramWarning("This is not a hydrogen atom!")
            if ignoreWarning == False:
                return
        position = -1
        for i in self.bonds:
            if i[0] == h and position == -1 and self.atoms[i[1]].symbol == "C":
                position = i[1]
            elif i[1] == h and position == -1 and self.atoms[
                i[0]].symbol == "C":
                position = i[0]
            elif i[0] == h and position != -1:
                ProgramWarning("This hydrogen atom has more than one bond!")
                position = i[1]
                if ignoreWarning == False:
                    return
            elif i[1] == h and position != -1:
                ProgramWarning("This hydrogen atom has more than one bond!")
                position = i[0]
                if ignoreWarning == False:
                    return
        if position == -1:
            ProgramWarning("This hydrogen atom has no bond!")
            return
        # Find two more hydrogen atoms that are bound to the C atom in "position"
        position2 = -1
        position3 = -1
        for i in self.bonds:
            if i[0] == position and i[1] != h and self.atoms[
                i[1]].symbol == "H":
                if position2 == -1:
                    position2 = i[1]
                elif position3 == -1:
                    position3 = i[1]
                    break
            if i[1] == position and i[0] != h and self.atoms[
                i[0]].symbol == "H":
                if position2 == -1:
                    position2 = i[0]
                elif position3 == -1:
                    position3 = i[0]
                    break
        print("The two other hydrogens are:", position2, " and ", position3)
        if position3 == -1:
            ProgramWarning("The C atom is not part of a CH₃ group!")
            return

        # Calculate the vector from C to H, scale it and put a new C atom there
        xcompo = self.atoms[h].coord[0] - self.atoms[position].coord[0]
        ycompo = self.atoms[h].coord[1] - self.atoms[position].coord[1]
        zcompo = self.atoms[h].coord[2] - self.atoms[position].coord[2]
        # Scale this vector to the "right" length:
        # We choose 1.65 as the length of a slightly elongated C=C (double) bond
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * 1.65
        ycompo = (ycompo / norm) * 1.65
        zcompo = (zcompo / norm) * 1.65
        # Calculate the position of the new C atom
        xcompo = self.atoms[position].coord[0] + xcompo
        ycompo = self.atoms[position].coord[1] + ycompo
        zcompo = self.atoms[position].coord[2] + zcompo
        # Add this new atom
        self.add_atom(Atom("C", xcompo, ycompo, zcompo, 0.1))
        # And call it's number newC for future reference
        newC = self.num_atoms() - 1

        # Again, setup the coordinates of the second new hydrogen atom:
        # We start by constructing the vector from the newly created C atom
        # to the "old one" it is bonded to
        xcompo = self.atoms[position].coord[0] - self.atoms[newC].coord[0]
        ycompo = self.atoms[position].coord[1] - self.atoms[newC].coord[1]
        zcompo = self.atoms[position].coord[2] - self.atoms[newC].coord[2]
        # Scale this vector to the "right" length:
        # We choose 1.25 as the length of a slightly elongated C-H bond
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * 1.25
        ycompo = (ycompo / norm) * 1.25
        zcompo = (zcompo / norm) * 1.25
        # Next we determine the axis around which we want to rotate this vector:
        # We use the vectors from the new C atom to the first one and the vector
        # from the new C atom to the "second atom" we found that was also bonded
        # to the first C atom.

        # Create a "composite position" between the other two hydrogens.
        # We'll place the last hydrogen atom along this vector
        xcomposit = (self.atoms[position2].coord[0] +
                     self.atoms[position3].coord[0]) / 2
        ycomposit = (self.atoms[position2].coord[1] +
                     self.atoms[position3].coord[1]) / 2
        zcomposit = (self.atoms[position2].coord[2] +
                     self.atoms[position3].coord[2]) / 2

        # Importantly, we keep its coordinates untouched once we have them,
        # so we can create the other two hydrogen atoms in the same way.
        crossPx = xcomposit - self.atoms[newC].coord[0]
        crossPy = ycomposit - self.atoms[newC].coord[1]
        crossPz = zcomposit - self.atoms[newC].coord[2]
        cross = np.cross([crossPx, crossPy, crossPz], [xcompo, ycompo, zcompo])
        # Then we build the rotation matrix
        L = (cross[0] ** 2) + (cross[1] ** 2) + (cross[2] ** 2)
        pi = np.radians(120.0)

        RotMatrix = []
        Rxx = (cross[0] ** 2 + (cross[1] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Rxy = ((cross[0] * cross[1]) * (1 - np.cos(pi)) - cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Rxz = ((cross[0] * cross[2]) * (1 - np.cos(pi)) + cross[1] * np.sqrt(
            L) * np.sin(pi)) / L

        Ryx = ((cross[0] * cross[1]) * (1 - np.cos(pi)) + cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Ryy = (cross[1] ** 2 + (cross[0] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Ryz = ((cross[1] * cross[2]) * (1 - np.cos(pi)) - cross[0] * np.sqrt(
            L) * np.sin(pi)) / L

        Rzx = ((cross[0] * cross[2]) * (1 - np.cos(pi)) - cross[1] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzy = ((cross[1] * cross[2]) * (1 - np.cos(pi)) + cross[0] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzz = (cross[2] ** 2 + (cross[0] ** 2 + cross[1] ** 2) * np.cos(
            pi)) / L

        RotMatrix.append([Rxx, Rxy, Rxz])
        RotMatrix.append([Ryx, Ryy, Ryz])
        RotMatrix.append([Rzx, Rzy, Rzz])

        RotMatrix = np.matrix(RotMatrix)

        vector = [xcompo, ycompo, zcompo]
        vector = np.matrix(vector)
        vector = RotMatrix.dot(np.matrix.transpose(vector))
        vector = np.array(vector).flatten().tolist()
        xcompo = vector[0]
        ycompo = vector[1]
        zcompo = vector[2]
        # Calculate the position of the new H atom
        xNew1 = self.atoms[newC].coord[0] + xcompo
        yNew1 = self.atoms[newC].coord[1] + ycompo
        zNew1 = self.atoms[newC].coord[2] + zcompo
        # Add this new atom
        self.add_atom(Atom("H", xNew1, yNew1, zNew1, 0.1))

        # Setup the coordinates of the first new hydrogen atom:
        # We start by constructing the vector from the newly created C atom
        # to the "old one" it is bonded to
        xcompo = self.atoms[position].coord[0] - self.atoms[newC].coord[0]
        ycompo = self.atoms[position].coord[1] - self.atoms[newC].coord[1]
        zcompo = self.atoms[position].coord[2] - self.atoms[newC].coord[2]
        # Scale this vector to the "right" length:
        # We choose 1.25 as the length of a slightly elongated C-H bond
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * 1.25
        ycompo = (ycompo / norm) * 1.25
        zcompo = (zcompo / norm) * 1.25
        # Next we determine the axis around which we want to rotate this vector:
        # We use the vectors from the new C atom to the first one and the vector
        # from the new C atom to the "second atom" we found that was also bonded
        # to the first C atom.

        # Create a "composite position" between the other two hydrogens.
        # We'll place the last hydrogen atom along this vector
        xcomposit = (self.atoms[position2].coord[0] +
                     self.atoms[position3].coord[0]) / 2
        ycomposit = (self.atoms[position2].coord[1] +
                     self.atoms[position3].coord[1]) / 2
        zcomposit = (self.atoms[position2].coord[2] +
                     self.atoms[position3].coord[2]) / 2

        # Importantly, we keep its coordinates untouched once we have them,
        # so we can create the other two hydrogen atoms in the same way.
        crossPx = xcomposit - self.atoms[newC].coord[0]
        crossPy = ycomposit - self.atoms[newC].coord[1]
        crossPz = zcomposit - self.atoms[newC].coord[2]
        cross = np.cross([crossPx, crossPy, crossPz], [xcompo, ycompo, zcompo])
        # Then we build the rotation matrix
        L = (cross[0] ** 2) + (cross[1] ** 2) + (cross[2] ** 2)
        pi = np.radians(240.0)

        RotMatrix = []
        Rxx = (cross[0] ** 2 + (cross[1] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Rxy = ((cross[0] * cross[1]) * (1 - np.cos(pi)) - cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Rxz = ((cross[0] * cross[2]) * (1 - np.cos(pi)) + cross[1] * np.sqrt(
            L) * np.sin(pi)) / L

        Ryx = ((cross[0] * cross[1]) * (1 - np.cos(pi)) + cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Ryy = (cross[1] ** 2 + (cross[0] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Ryz = ((cross[1] * cross[2]) * (1 - np.cos(pi)) - cross[0] * np.sqrt(
            L) * np.sin(pi)) / L

        Rzx = ((cross[0] * cross[2]) * (1 - np.cos(pi)) - cross[1] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzy = ((cross[1] * cross[2]) * (1 - np.cos(pi)) + cross[0] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzz = (cross[2] ** 2 + (cross[0] ** 2 + cross[1] ** 2) * np.cos(
            pi)) / L

        RotMatrix.append([Rxx, Rxy, Rxz])
        RotMatrix.append([Ryx, Ryy, Ryz])
        RotMatrix.append([Rzx, Rzy, Rzz])

        RotMatrix = np.matrix(RotMatrix)

        vector = [xcompo, ycompo, zcompo]
        vector = np.matrix(vector)
        vector = RotMatrix.dot(np.matrix.transpose(vector))
        vector = np.array(vector).flatten().tolist()
        xcompo = vector[0]
        ycompo = vector[1]
        zcompo = vector[2]
        # Calculate the position of the new H atom
        xNew1 = self.atoms[newC].coord[0] + xcompo
        yNew1 = self.atoms[newC].coord[1] + ycompo
        zNew1 = self.atoms[newC].coord[2] + zcompo
        # Add this new atom
        self.add_atom(Atom("H", xNew1, yNew1, zNew1, 0.1))

        # Calculate the vector from C to the compisite position, scale it and put the last H atom there
        xcompo = xcomposit - self.atoms[position].coord[0]
        ycompo = ycomposit - self.atoms[position].coord[1]
        zcompo = zcomposit - self.atoms[position].coord[2]
        # Scale this vector to the "right" length:
        # We choose 1.25 as the length of a slightly elongated C-H bond
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * 1.25
        ycompo = (ycompo / norm) * 1.25
        zcompo = (zcompo / norm) * 1.25
        # Calculate the position of the new C atom
        xcompo = self.atoms[position].coord[0] + xcompo
        ycompo = self.atoms[position].coord[1] + ycompo
        zcompo = self.atoms[position].coord[2] + zcompo
        # Add this new atom
        self.add_atom(Atom("H", xcompo, ycompo, zcompo, 0.1))

        # Delete the initial hydrogen atom h and the two hydrogen atoms in position2 and position3
        # That were bound to the other carbon atom
        # EXTREMELY MESSY!!!
        for i in sorted([h, position2, position3], key=int, reverse=True):
            del self.atoms[i]

    def replaceHwithCHO(self, h, orientation=0, ignoreWarning=False):
        """ (Molecule) -> NoneType

    Exchange the given hydrogen atom with a keto (CHO) group.
    The "orientation" parameter is needed to decide where the new
    group "points".
    """
        # Safety check: h is indeed an H atom and has only one bond
        if self.atoms[h].symbol != "H":
            ProgramWarning("This is not a hydrogen atom!")
            if ignoreWarning == False:
                return
        position = -1
        for i in self.bonds:
            if i[0] == h and position == -1:
                position = i[1]
            elif i[1] == h and position == -1:
                position = i[0]
            elif i[0] == h and position != -1:
                ProgramWarning("This hydrogen atom has more than one bond!")
                position = i[1]
                if ignoreWarning == False:
                    return
            elif i[1] == h and position != -1:
                ProgramWarning("This hydrogen atom has more than one bond!")
                position = i[0]
                if ignoreWarning == False:
                    return
        if position == -1:
            ProgramWarning("This hydrogen atom has no bond!")
            return
        # Find one more atom that is bound to the atom in "position"
        position2 = -1
        for i in self.bonds:
            if i[0] == position and i[1] != h:
                position2 = i[1]
            if i[1] == position and i[0] != h:
                position2 = i[0]
        if position2 == -1:
            ProgramWarning("The C atom no seems to have only one bond!")
            return
        # Calculate the vector from C to H, scale it and put the replacement atom there
        xcompo = self.atoms[h].coord[0] - self.atoms[position].coord[0]
        ycompo = self.atoms[h].coord[1] - self.atoms[position].coord[1]
        zcompo = self.atoms[h].coord[2] - self.atoms[position].coord[2]
        # Scale this vector to the "right" length:
        # We choose 110% of the sum of vdW radii  as the length of a slightly elongated bond
        scale = (SymbolToRadius["C"] + SymbolToRadius["C"]) * 1.1
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * scale
        ycompo = (ycompo / norm) * scale
        zcompo = (zcompo / norm) * scale
        # Calculate the position of the replacement atom
        xcompo = self.atoms[position].coord[0] + xcompo
        ycompo = self.atoms[position].coord[1] + ycompo
        zcompo = self.atoms[position].coord[2] + zcompo
        # Add this new atom
        self.add_atom(Atom("C", xcompo, ycompo, zcompo, 0.1))
        # And call it's number newAtom for future reference
        newAtom = self.num_atoms() - 1

        # Setup the coordinates of the oxygen atom:
        # We start by constructing the vector from the newly created C atom
        # to the "old one" it is bonded to
        xcompo = self.atoms[position].coord[0] - self.atoms[newAtom].coord[0]
        ycompo = self.atoms[position].coord[1] - self.atoms[newAtom].coord[1]
        zcompo = self.atoms[position].coord[2] - self.atoms[newAtom].coord[2]
        # Scale this vector to the "right" length:
        # We choose 110% of the sum of vdW radii  as the length of a slightly elongated bond
        scale = (SymbolToRadius["O"] + SymbolToRadius["C"]) * 1.1
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * scale
        ycompo = (ycompo / norm) * scale
        zcompo = (zcompo / norm) * scale
        # Next we determine the axis around which we want to rotate this vector:
        # We use the vectors from the new N atom to the C atom and the vector
        # from the N atom to the "second atom" we found that was also bonded
        # to the C atom.
        # Importantly, we keep its coordinates untouched once we have them,
        # so we can create the other two hydrogen atoms in the same way.
        crossPx = self.atoms[position2].coord[0] - self.atoms[newAtom].coord[0]
        crossPy = self.atoms[position2].coord[1] - self.atoms[newAtom].coord[1]
        crossPz = self.atoms[position2].coord[2] - self.atoms[newAtom].coord[2]
        cross = np.cross([crossPx, crossPy, crossPz], [xcompo, ycompo, zcompo])
        # Then we build the rotation matrix
        L = (cross[0] ** 2) + (cross[1] ** 2) + (cross[2] ** 2)
        pi = np.radians(120.0)

        RotMatrix = []
        Rxx = (cross[0] ** 2 + (cross[1] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Rxy = ((cross[0] * cross[1]) * (1 - np.cos(pi)) - cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Rxz = ((cross[0] * cross[2]) * (1 - np.cos(pi)) + cross[1] * np.sqrt(
            L) * np.sin(pi)) / L

        Ryx = ((cross[0] * cross[1]) * (1 - np.cos(pi)) + cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Ryy = (cross[1] ** 2 + (cross[0] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Ryz = ((cross[1] * cross[2]) * (1 - np.cos(pi)) - cross[0] * np.sqrt(
            L) * np.sin(pi)) / L

        Rzx = ((cross[0] * cross[2]) * (1 - np.cos(pi)) - cross[1] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzy = ((cross[1] * cross[2]) * (1 - np.cos(pi)) + cross[0] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzz = (cross[2] ** 2 + (cross[0] ** 2 + cross[1] ** 2) * np.cos(
            pi)) / L

        RotMatrix.append([Rxx, Rxy, Rxz])
        RotMatrix.append([Ryx, Ryy, Ryz])
        RotMatrix.append([Rzx, Rzy, Rzz])

        RotMatrix = np.matrix(RotMatrix)

        vector = [xcompo, ycompo, zcompo]
        vector = np.matrix(vector)
        vector = RotMatrix.dot(np.matrix.transpose(vector))
        vector = np.array(vector).flatten().tolist()
        xcompo = vector[0]
        ycompo = vector[1]
        zcompo = vector[2]
        # Calculate the position of the new O atom
        # xNewO = self.atoms[newAtom].coord[0] + xcompo
        # yNewO = self.atoms[newAtom].coord[1] + ycompo
        # zNewO = self.atoms[newAtom].coord[2] + zcompo
        xNewO = xcompo
        yNewO = ycompo
        zNewO = zcompo

        # The, we setup the coordinates of the hydrogen atom in the same way:
        # We start by constructing the vector from the newly created C atom
        # to the "old one" it is bonded to
        xcompo = self.atoms[position].coord[0] - self.atoms[newAtom].coord[0]
        ycompo = self.atoms[position].coord[1] - self.atoms[newAtom].coord[1]
        zcompo = self.atoms[position].coord[2] - self.atoms[newAtom].coord[2]
        # Scale this vector to the "right" length:
        # We choose 110% of the sum of vdW radii  as the length of a slightly elongated bond
        scale = (SymbolToRadius["H"] + SymbolToRadius["C"]) * 1.1
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * scale
        ycompo = (ycompo / norm) * scale
        zcompo = (zcompo / norm) * scale
        # Next we determine the axis around which we want to rotate this vector:
        # We use the vectors from the new C atom to the C atom and the vector
        # from the C atom to the "second atom" we found that was also bonded
        # to the C atom.
        # Importantly, we keep its coordinates untouched once we have them,
        # so we can create the other two hydrogen atoms in the same way.
        crossPx = self.atoms[position2].coord[0] - self.atoms[newAtom].coord[0]
        crossPy = self.atoms[position2].coord[1] - self.atoms[newAtom].coord[1]
        crossPz = self.atoms[position2].coord[2] - self.atoms[newAtom].coord[2]
        cross = np.cross([crossPx, crossPy, crossPz], [xcompo, ycompo, zcompo])
        # Then we build the rotation matrix
        L = (cross[0] ** 2) + (cross[1] ** 2) + (cross[2] ** 2)
        pi = np.radians(240.0)

        RotMatrix = []
        Rxx = (cross[0] ** 2 + (cross[1] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Rxy = ((cross[0] * cross[1]) * (1 - np.cos(pi)) - cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Rxz = ((cross[0] * cross[2]) * (1 - np.cos(pi)) + cross[1] * np.sqrt(
            L) * np.sin(pi)) / L

        Ryx = ((cross[0] * cross[1]) * (1 - np.cos(pi)) + cross[2] * np.sqrt(
            L) * np.sin(pi)) / L
        Ryy = (cross[1] ** 2 + (cross[0] ** 2 + cross[2] ** 2) * np.cos(
            pi)) / L
        Ryz = ((cross[1] * cross[2]) * (1 - np.cos(pi)) - cross[0] * np.sqrt(
            L) * np.sin(pi)) / L

        Rzx = ((cross[0] * cross[2]) * (1 - np.cos(pi)) - cross[1] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzy = ((cross[1] * cross[2]) * (1 - np.cos(pi)) + cross[0] * np.sqrt(
            L) * np.sin(pi)) / L
        Rzz = (cross[2] ** 2 + (cross[0] ** 2 + cross[1] ** 2) * np.cos(
            pi)) / L

        RotMatrix.append([Rxx, Rxy, Rxz])
        RotMatrix.append([Ryx, Ryy, Ryz])
        RotMatrix.append([Rzx, Rzy, Rzz])

        RotMatrix = np.matrix(RotMatrix)

        vector = [xcompo, ycompo, zcompo]
        vector = np.matrix(vector)
        vector = RotMatrix.dot(np.matrix.transpose(vector))
        vector = np.array(vector).flatten().tolist()
        xcompo = vector[0]
        ycompo = vector[1]
        zcompo = vector[2]
        # Calculate the position of the new O atom
        # xNewH = self.atoms[newAtom].coord[0] + xcompo
        # yNewH = self.atoms[newAtom].coord[1] + ycompo
        # zNewH = self.atoms[newAtom].coord[2] + zcompo
        xNewH = xcompo
        yNewH = ycompo
        zNewH = zcompo

        if orientation != 0:
            # Next, we re-use the rotated coordinates and rotate them further
            # by 120 or 240 degrees around the C-C bond
            cross[0] = self.atoms[position].coord[0] - \
                       self.atoms[newAtom].coord[0]
            cross[1] = self.atoms[position].coord[1] - \
                       self.atoms[newAtom].coord[1]
            cross[2] = self.atoms[position].coord[2] - \
                       self.atoms[newAtom].coord[2]
            # We build the second rotation matrix
            L = (cross[0] ** 2) + (cross[1] ** 2) + (cross[2] ** 2)
            if orientation == 1:
                pi = np.radians(120.0)
            else:
                pi = np.radians(240.0)

            RotMatrix = []
            Rxx = (cross[0] ** 2 + (cross[1] ** 2 + cross[2] ** 2) * np.cos(
                pi)) / L
            Rxy = ((cross[0] * cross[1]) * (1 - np.cos(pi)) - cross[
                2] * np.sqrt(L) * np.sin(pi)) / L
            Rxz = ((cross[0] * cross[2]) * (1 - np.cos(pi)) + cross[
                1] * np.sqrt(L) * np.sin(pi)) / L

            Ryx = ((cross[0] * cross[1]) * (1 - np.cos(pi)) + cross[
                2] * np.sqrt(L) * np.sin(pi)) / L
            Ryy = (cross[1] ** 2 + (cross[0] ** 2 + cross[2] ** 2) * np.cos(
                pi)) / L
            Ryz = ((cross[1] * cross[2]) * (1 - np.cos(pi)) - cross[
                0] * np.sqrt(L) * np.sin(pi)) / L

            Rzx = ((cross[0] * cross[2]) * (1 - np.cos(pi)) - cross[
                1] * np.sqrt(L) * np.sin(pi)) / L
            Rzy = ((cross[1] * cross[2]) * (1 - np.cos(pi)) + cross[
                0] * np.sqrt(L) * np.sin(pi)) / L
            Rzz = (cross[2] ** 2 + (cross[0] ** 2 + cross[1] ** 2) * np.cos(
                pi)) / L

            RotMatrix.append([Rxx, Rxy, Rxz])
            RotMatrix.append([Ryx, Ryy, Ryz])
            RotMatrix.append([Rzx, Rzy, Rzz])

            RotMatrix = np.matrix(RotMatrix)

            vector = [xNewO, yNewO, zNewO]
            vector = np.matrix(vector)
            vector = RotMatrix.dot(np.matrix.transpose(vector))
            vector = np.array(vector).flatten().tolist()
            xNewO = vector[0]
            yNewO = vector[1]
            zNewO = vector[2]

            vector = [xNewH, yNewH, zNewH]
            vector = np.matrix(vector)
            vector = RotMatrix.dot(np.matrix.transpose(vector))
            vector = np.array(vector).flatten().tolist()
            xNewH = vector[0]
            yNewH = vector[1]
            zNewH = vector[2]

        # Calculate the position of the new O and H atoms and add them
        xNewO = self.atoms[newAtom].coord[0] + xNewO
        yNewO = self.atoms[newAtom].coord[1] + yNewO
        zNewO = self.atoms[newAtom].coord[2] + zNewO
        self.add_atom(Atom("O", xNewO, yNewO, zNewO, 0.1))
        xNewH = self.atoms[newAtom].coord[0] + xNewH
        yNewH = self.atoms[newAtom].coord[1] + yNewH
        zNewH = self.atoms[newAtom].coord[2] + zNewH
        self.add_atom(Atom("H", xNewH, yNewH, zNewH, 0.1))

        # Delete the initial hydrogen atom h # EXTREMELY MESSY!!!
        del self.atoms[h]
        return

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
        if exists == False and a >= 0 and b >= 0 and a <= len(
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
        if exists == True:
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
        if exists is False and a >= 0 and b >= 0 and c >= 0 and a <= len(
                self.atoms) and b <= len(
            self.atoms) and c <= len(
            self.atoms) and a != b and a != c and b != c:
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
        Report the distance between atoms i and j in Ångströms.

        :param i: First atom for distance measurement.
        :param j: Second atom for distance measurement.
        :return: The distance between atoms i and j in Ångströms.
        """

        distance = (self.atm_pos_x(i) - self.atm_pos_x(j)) * (
                self.atm_pos_x(i) - self.atm_pos_x(j))
        distance = distance + (self.atm_pos_y(i) - self.atm_pos_y(j)) * (
                self.atm_pos_y(i) - self.atm_pos_y(j))
        distance = distance + (self.atm_pos_z(i) - self.atm_pos_z(j)) * (
                self.atm_pos_z(i) - self.atm_pos_z(j))

        return math.sqrt(distance)

    def bond_angle(self, i: int) -> float:
        """
        Report the angle described by three atoms in the bonds list.

        :param i: Index of the bond angle from the list of angles.
        :return: The numerical value of the angle in radians.
        """


        # Calculate the distance between each pair of atoms
        angle = self.angles[i]
        d_bond_1 = self.atm_atm_dist(angle[0], angle[1])
        d_bond_2 = self.atm_atm_dist(angle[1], angle[2])
        d_non_bond = self.atm_atm_dist(angle[0], angle[2])

        # Use those distances and the cosine rule to calculate bond angle theta
        numerator = d_bond_1 ** 2 + d_bond_2 ** 2 - d_non_bond ** 2
        denominator = 2 * d_bond_1 * d_bond_2
        argument = numerator / denominator
        theta = np.arccos(argument)

        return theta

    def dihedral_angle(self, i: int) -> float:
        """
        Report the dihedral angle described by a set of four atoms in the list
        of dihedrals.

        :param i: Index of the dihedral angle in the list of dihedrals.
        :return: The numerical value of the angle in radians.
        """


        # Calculate the vectors lying along bonds, and their cross products
        dihedral = self.dihedrals[i]
        atom_e1 = self.atoms[dihedral[0]]
        atom_b1 = self.atoms[dihedral[1]]
        atom_b2 = self.atoms[dihedral[2]]
        atom_e2 = self.atoms[dihedral[3]]
        end_1 = [atom_e1.coord[i] - atom_b1.coord[i] for i in range(3)]
        bridge = [atom_b1.coord[i] - atom_b2.coord[i] for i in range(3)]
        end_2 = [atom_b2.coord[i] - atom_e2.coord[i] for i in range(3)]
        vnormal_1 = np.cross(end_1, bridge)
        vnormal_2 = np.cross(bridge, end_2)

        # Construct a set of orthogonal basis vectors to define a frame with
        # vnormal_2 as the x axis
        vcross = np.cross(vnormal_2, bridge)
        norm_vn2 = np.linalg.norm(vnormal_2)
        norm_b = np.linalg.norm(bridge)
        norm_vc = np.linalg.norm(vcross)
        basis_vn2 = [vnormal_2[i] / norm_vn2 for i in range(3)]
        basis_b = [bridge[i] / norm_b for i in range(3)]
        basis_cv = [vcross[i] / norm_vc for i in range(3)]

        # Find the signed angle betw. vnormal_1 and vnormal_2 in the new frame
        vn1_coord_n2 = np.dot(vnormal_1, basis_vn2)
        vn1_coord_vc = np.dot(vnormal_1, basis_cv)
        psi = math.atan2(vn1_coord_vc, vn1_coord_n2)

        return psi

    def set_mult(self, M: int) -> None:
        """
        Set the multiplicity of the molecule to M

        :param M: Value of the multiplicity (i.e. 1 for singlet, 2 for doublet)
        :return: None
        """

        self.mult = M

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

        if output == "cart":
            # Plain cartesian coordinates - nothing else
            s = ""
            for i in self.atoms:
                t = "{:<3} {: .8f} {: .8f} {: .8f}\n".format(i.symbol(),
                                                             i.xpos(),
                                                             i.ypos(),
                                                             i.zpos())
                s = s + t
        elif output == "xyz":
            # xyz-file format. Number of atoms first, comment line,
            # then coordinates
            s = str(self.num_atoms()) + "\n" + commentline + "\n"
            for i in self.atoms:
                t = "{:<3} {: .8f} {: .8f} {: .8f}\n".format(i.symbol(),
                                                             i.xpos(),
                                                             i.ypos(),
                                                             i.zpos())
                s = s + t
        elif output == "gamess":
            # xyz coordinates in Gamess format
            s = " $DATA\n" + commentline + "\nC1\n"
            for i in self.atoms:
                t = "{:<3} {:<3d} {: .8f} {: .8f} {: .8f}\n".format(i.symbol(),
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
                if i.mass == SymbolToMass[i.symbol()]:
                    t = "{:<3} {: .8f} {: .8f} {: .8f}\n".format(i.symbol(),
                                                                 i.xpos(),
                                                                 i.ypos(),
                                                                 i.zpos())
                else:
                    t = "{} {: .8f} {: .8f} {: .8f}\n".format(
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
            except:
                ProgramError("Cannot open " + file + " for writing.")
            # Then write the string we created above
            with open(file + "." + output, "w") as text_file:
                print("{}".format(s), file=text_file)

        return s


def main():
    # Create an example molecule and add some atoms
    example = Molecule("Example Molecule")
    example.add_atom(Atom("C", x=-0.63397128, y=0.57416267, z=0.000000))
    example.add_atom(Atom("H", x=-0.27731685, y=-0.43464733, z=0.000000))
    example.add_atom(Atom("H", x=-0.27729844, y=1.07856086, z=0.87365150))
    example.add_atom(Atom("H", x=-0.27729844, y=1.07856086, z=-0.87365150))
    example.add_atom(
        Atom("H", x=-1.70397128, y=0.57417585, z=0.00000000, mass=2.0))
    print(example.print_mol(output = "gauss"))



if __name__ == '__main__':
    main()
