################################################################################
# Atom class and class methods to be defined below
################################################################################

from typing import Optional

from wellfare.constants import *


class Atom:
    """
    An atom defined by its nucleus (i.e. charge or symbol and (optional) mass)
    and its cartesian coordinates in Ångströms.
    """

    def __init__(self, sym: Optional[str] = None,
                 charge: Optional[int] = None, x: Optional[float] = 0.0,
                 y: Optional[float] = 0.0, z: Optional[float] = 0.0,
                 mass: Optional[float] = None) -> None:
        """
        This method creates an atom somewhere in cartesian space. Either charge
        (i.e. atomic number, number of protons, etc.) or the atomic symbol can
        be specified. Optionally, the atomic mass can be specified (in order to
        create a specific isotope/nuclide). Cartesian coordinates can be given
        to specify the location in space. By default, a hydrogen atom at the
        origin of the coordinate system is created.

        If both symbol and atomic number are given, only the symbol is used in
        the creation, the atomic number (consistent with the symbol or not) is
        discarded.

        Specification of an unknown atomic number will revert back to hydrogen.

        :param charge: The nuclear charge (i.e. the atomic number) that defines
                       the element.
        :param x: The cartesian x-coordinate of the atom in Ångströms.
        :param y: The cartesian y-coordinate of the atom in Ångströms.
        :param z: The cartesian z-coordinate of the atom in Ångströms.
        :param mass: Optionally, the mass of the atom in amu. If unspecified,
                     the standard atomic weight (as defined in constants.py
                     will be used)
        """

        if sym is None and charge is not None:
            if charge in number_to_symbol:
                self.charge = charge
            else:
                self.charge = 1
        elif sym is not None and charge is None:
            if sym in symbol_to_number:
                self.charge = symbol_to_number[sym]
            else:
                self.charge = 1
        elif sym is None and charge is None:
            self.charge = 1
        else:
            # If both are specified, the symbol takes precedence
            self.charge = symbol_to_number[sym]

        if mass is None:
            self.mass = symbol_to_mass[self.symbol()]
        else:
            self.mass = mass

        self.coord = [x, y, z]

    def __str__(self) -> str:
        """

        :return: Returns a string representation of this Atom in this
                 format: (MASS)SYMBOL, (X, Y, Z)
        """

        return '({: 8.3f}){}: ({: 13.8f}, {: 13.8f}, {: 13.8f}))'.format(self.mass,
                                                             self.symbol(),
                                                             self.coord[0],
                                                             self.coord[1],
                                                             self.coord[2])

    def __repr__(self) -> str:
        """

        :return: Returns a string representation of this Atom in this
                 format: (MASS)SYMBOL, (X, Y, Z)
        """

        return '({: 8.3f}){}: ({: 13.8f}, {: 13.8f}, {: 13.8f}))'.format(self.mass,
                                                             self.symbol(),
                                                             self.coord[0],
                                                             self.coord[1],
                                                             self.coord[2])

    def symbol(self) -> str:
        """

        :return: Returns the atomic symbol of the atom (from constants.py)
        """

        return number_to_symbol[self.charge]

    def xpos(self) -> float:
        """

        :return: Returns the x-coordinate of the atom
        """

        return self.coord[0]

    def ypos(self) -> float:
        """

        :return: Returns the y-coordinate of the atom
        """

        return self.coord[1]

    def zpos(self) -> float:
        """

        :return: Returns the z-coordinate of the atom
        """

        return self.coord[2]

    def set_charge(self, q: int):
        """
        This method re-sets the nuclear charge, i.e. the type of element.

        :param q: The nuclear charge (i.e. atomic number)
        :return: None
        """
        if q in number_to_symbol:
            self.charge = q

    def set_x(self, x: float) -> None:
        """
        This method re-sets the x-coordinate of the atom.

        :param x: Sets the cartesion x-coordinate of the atom to x.
        :return: None
        """
        self.coord[0] = x

    def set_y(self, y: float) -> None:
        """
        This method re-sets the y-coordinate of the atom.

        :param y: Sets the cartesion x-coordinate of the atom to y.
        :return: None
        """
        self.coord[1] = y

    def set_z(self, z: float) -> None:
        """
        This method re-sets the z-coordinate of the atom.

        :param z: Sets the cartesion x-coordinate of the atom to z.
        :return: None
        """
        self.coord[2] = z

    def set_xyz(self, x: float, y: float, z: float) -> None:
        """
        This method re-sets all x, y and z coordinates of the atom.

        :param x: Sets the cartesion x-coordinate of the atom to x.
        :param y: Sets the cartesion y-coordinate of the atom to y.
        :param z: Sets the cartesion x-coordinate of the atom to z.
        :return: None
        """
        self.coord[0] = x
        self.coord[1] = y
        self.coord[2] = z


def main():
    # Create all atoms in the periodic table up to ₁₁₈Og.
    for i in symbol_to_number:
        print(Atom(sym=i))


if __name__ == '__main__':
    main()
