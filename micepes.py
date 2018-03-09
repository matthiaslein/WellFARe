###############################################################################
#                                                                             #
# This is where the MICEPES program is put together                           #
#                                                                             #
###############################################################################

import argparse
import math
import os
import copy
import numpy as np

from messages import *
from atom import Atom
from molecule import Molecule
from qmparser import extract_molecular_data
from constants import SymbolToRadius


def build_rotation_matrix(axis, angle):
    """

    :param axis: A vector describing the axis around which will be rotated
                 (e.g. [0, 0, 1] for a rotation around the z-axis).
    :param angle: The angle for the rotation in degrees (not radians!)
    :return: An np.matrix object with the rotation matrix.
    """

    length = (axis[0] ** 2) + (axis[1] ** 2) + (axis[2] ** 2)
    pi = np.radians(angle)

    r_xx = axis[0] ** 2 + (axis[1] ** 2 + axis[2] ** 2) * np.cos(pi)
    r_xy = (axis[0] * axis[1]) * (1 - np.cos(pi)) - (
            axis[2] * np.sqrt(length) * np.sin(pi))
    r_xz = (axis[0] * axis[2]) * (1 - np.cos(pi)) + (
            axis[1] * np.sqrt(length) * np.sin(pi))

    r_yx = (axis[0] * axis[1]) * (1 - np.cos(pi)) + (
            axis[2] * np.sqrt(length) * np.sin(pi))
    r_yy = axis[1] ** 2 + (axis[0] ** 2 + axis[2] ** 2) * np.cos(pi)
    r_yz = (axis[1] * axis[2]) * (1 - np.cos(pi)) - (
            axis[0] * np.sqrt(length) * np.sin(pi))

    r_zx = (axis[0] * axis[2]) * (1 - np.cos(pi)) - (
            axis[1] * np.sqrt(length) * np.sin(pi))
    r_zy = (axis[1] * axis[2]) * (1 - np.cos(pi)) + (
            axis[0] * np.sqrt(length) * np.sin(pi))
    r_zz = axis[2] ** 2 + (axis[0] ** 2 + axis[1] ** 2) * np.cos(pi)

    rot_matrix = [[r_xx, r_xy, r_xz], [r_yx, r_yy, r_yz], [r_zx, r_zy, r_zz]]

    rot_matrix = np.matrix(rot_matrix)
    rot_matrix = np.divide(rot_matrix, length)

    # Alternative for element-wise division
    # rot_matrix[:] = [x / length for x in rot_matrix]

    return rot_matrix


def replace_h_with_tetrahedral(molecule, h, replacement="C", add_h=3,
                               orientation=0, ignore_warning=False):
    """ (Molecule) -> (Molecule)

    Exchange the given hydrogen atom with an arbitrary other atom and,
    optionally, add up to 3 hydrogen atoms in a tetrahedral geometry.
    Default behaviour is substitution by a CH₃ group. If only 1 or 2
    hydrogen atoms are added, an additional parameter "orientation" is
    needed to decide where the new group "points".
    """
    # First, create a copy of the molecule to work on (and later, return)
    new_mol = copy.deepcopy(molecule)

    # Safety check: h is indeed an H atom and has only one bond
    if new_mol.atm_symbol(h) != "H":
        msg_program_warning("This is not a hydrogen atom!")
        if ignore_warning is False:
            return
    position = -1
    for i in new_mol.bonds:
        if i[0] == h and position == -1:
            position = i[1]
        elif i[1] == h and position == -1:
            position = i[0]
        elif i[0] == h and position != -1:
            msg_program_warning("This hydrogen atom has more "
                                "than one bond!")
            position = i[1]
            if ignore_warning is False:
                return
        elif i[1] == h and position != -1:
            msg_program_warning("This hydrogen atom has more "
                                "than one bond!")
            position = i[0]
            if ignore_warning is False:
                return
    if position == -1:
        msg_program_warning("This hydrogen atom has no bond!")
        return
    # Find one more atom that is bound to the atom in "position"
    position2 = -1
    for i in new_mol.bonds:
        if i[0] == position and i[1] != h:
            position2 = i[1]
        if i[1] == position and i[0] != h:
            position2 = i[0]
    if position2 == -1:
        msg_program_warning("The C atom no seems to have only one bond!")
        return
    # Calculate the vector from C to H, scale it and put the
    # replacement atom there
    xcompo = new_mol.atm_pos_x(h) - new_mol.atm_pos_x(position)
    ycompo = new_mol.atm_pos_y(h) - new_mol.atm_pos_y(position)
    zcompo = new_mol.atm_pos_z(h) - new_mol.atm_pos_z(position)
    # Scale this vector to the "right" length:
    # We choose 110% of the sum of vdW radii  as the length of a
    # slightly elongated bond
    scale = (SymbolToRadius["C"] + SymbolToRadius[replacement]) * 1.1
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * scale
    ycompo = (ycompo / norm) * scale
    zcompo = (zcompo / norm) * scale
    # Calculate the position of the replacement atom
    xcompo = new_mol.atm_pos_x(position) + xcompo
    ycompo = new_mol.atm_pos_y(position) + ycompo
    zcompo = new_mol.atm_pos_z(position) + zcompo
    # Add this new atom
    new_mol.add_atom(Atom(sym=replacement, x=xcompo, y=ycompo, z=zcompo))
    # And call it's number newN for future reference
    new_atom = new_mol.num_atoms() - 1
    if add_h != 0:
        # Setup the coordinates of the first hydrogen atom:
        # We start by constructing the vector from the newly created N atom
        # to the "old one" it is bonded to
        xcompo = new_mol.atm_pos_x(position) - new_mol.atm_pos_x(new_atom)
        ycompo = new_mol.atm_pos_y(position) - new_mol.atm_pos_y(new_atom)
        zcompo = new_mol.atm_pos_z(position) - new_mol.atm_pos_z(new_atom)
        # Scale this vector to the "right" length:
        # We choose 110% of the sum of vdW radii  as the length of a
        # slightly elongated bond
        scale = (SymbolToRadius["H"] + SymbolToRadius[replacement]) * 1.1
        norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
        xcompo = (xcompo / norm) * scale
        ycompo = (ycompo / norm) * scale
        zcompo = (zcompo / norm) * scale
        # Next we determine the axis around which we want to rotate
        # this vector:
        # We use the vectors from the new N atom to the C atom and
        # the vector from the N atom to the "second atom" we found that
        # was also bonded to the C atom.
        # Importantly, we keep its coordinates untouched once we have them,
        # so we can create the other two hydrogen atoms in the same way.
        cross_px = new_mol.atm_pos_x(position2) - new_mol.atm_pos_x(new_atom)
        cross_py = new_mol.atm_pos_y(position2) - new_mol.atm_pos_y(new_atom)
        cross_pz = new_mol.atm_pos_z(position2) - new_mol.atm_pos_z(new_atom)
        cross = np.cross([cross_px, cross_py, cross_pz],
                         [xcompo, ycompo, zcompo])
        rot_matrix = build_rotation_matrix(cross, 109.4712206)

        vector = np.matrix([xcompo, ycompo, zcompo])
        vector = rot_matrix @ np.matrix.transpose(vector)
        vector = np.array(vector).flatten().tolist()

        xcompo = vector[0]
        ycompo = vector[1]
        zcompo = vector[2]
        # Calculate the position of the new H atom
        x_new1 = new_mol.atm_pos_x(new_atom) + xcompo
        y_new1 = new_mol.atm_pos_y(new_atom) + ycompo
        z_new1 = new_mol.atm_pos_z(new_atom) + zcompo

        # Next, we re-use the rotated coordinates and rotate them further
        # by 120 degrees around the C-N bond
        cross[0] = new_mol.atm_pos_x(position) - new_mol.atm_pos_x(new_atom)
        cross[1] = new_mol.atm_pos_y(position) - new_mol.atm_pos_y(new_atom)
        cross[2] = new_mol.atm_pos_z(position) - new_mol.atm_pos_z(new_atom)

        # We build the second rotation matrix
        rot_matrix = build_rotation_matrix(cross, 120.0)

        # Rotate a second time
        vector = np.matrix([xcompo, ycompo, zcompo])
        vector = rot_matrix @ np.matrix.transpose(vector)
        vector = np.array(vector).flatten().tolist()
        xcompo = vector[0]
        ycompo = vector[1]
        zcompo = vector[2]
        # Calculate the position of the new H atom
        x_new2 = new_mol.atm_pos_x(new_atom) + xcompo
        y_new2 = new_mol.atm_pos_y(new_atom) + ycompo
        z_new2 = new_mol.atm_pos_z(new_atom) + zcompo

        # And then, we do this a third and last time... (re-using the rotation
        # matrix because we're rotating about 120 degrees again).
        vector = np.matrix([xcompo, ycompo, zcompo])
        vector = rot_matrix @ np.matrix.transpose(vector)
        vector = np.array(vector).flatten().tolist()
        xcompo = vector[0]
        ycompo = vector[1]
        zcompo = vector[2]
        # Calculate the position of the new H atom
        x_new3 = new_mol.atm_pos_x(new_atom) + xcompo
        y_new3 = new_mol.atm_pos_y(new_atom) + ycompo
        z_new3 = new_mol.atm_pos_z(new_atom) + zcompo

        if add_h == 3:
            new_mol.add_atom(Atom(sym="H", x=x_new1, y=y_new1, z=z_new1))
            new_mol.add_atom(Atom(sym="H", x=x_new2, y=y_new2, z=z_new2))
            new_mol.add_atom(Atom(sym="H", x=x_new3, y=y_new3, z=z_new3))
        elif add_h == 2:
            # Add two of the new atoms according to the chosen orientation
            if orientation == 1:
                new_mol.add_atom(Atom(sym="H", x=x_new1, y=y_new1, z=z_new1))
                new_mol.add_atom(Atom(sym="H", x=x_new3, y=y_new3, z=z_new3))
            elif orientation == 2:
                new_mol.add_atom(Atom(sym="H", x=x_new2, y=y_new2, z=z_new2))
                new_mol.add_atom(Atom(sym="H", x=x_new3, y=y_new3, z=z_new3))
            else:
                new_mol.add_atom(Atom(sym="H", x=x_new1, y=y_new1, z=z_new1))
                new_mol.add_atom(Atom(sym="H", x=x_new2, y=y_new2, z=z_new2))
        elif add_h == 1:
            # Add one of the new atoms according to the chosen orientation
            if orientation == 1:
                new_mol.add_atom(Atom(sym="H", x=x_new2, y=y_new2, z=z_new2))
            elif orientation == 2:
                new_mol.add_atom(Atom(sym="H", x=x_new3, y=y_new3, z=z_new3))
            else:
                new_mol.add_atom(Atom(sym="H", x=x_new1, y=y_new1, z=z_new1))

    # Delete the initial hydrogen atom h # EXTREMELY MESSY!!!
    del new_mol.atoms[h]
    return new_mol


def replace_h_with_ethenyl(molecule, h, ignore_warning=False):
    """ (Molecule) -> NoneType

    Exchange the given hydrogen atom by a "planar" CH2 group. This method
    requires the carbon atom whose hydrogen atom is being replaced to
    possess two more hydrogen atoms (i.e. it only works on methyl groups),
    because it will substitute the other two hydrogen atoms with a single
    hydrogen atom in the correct position, given by the sp2 hybridisation
    of the carbon atoms.
    """
    # First, create a copy of the molecule to work on (and later, return)
    new_mol = copy.deepcopy(molecule)

    # Safety check: h is indeed an H atom and has only one bond
    if new_mol.atm_symbol(h) != "H":
        msg_program_warning("This is not a hydrogen atom!")
        if ignore_warning is False:
            return
    position = -1
    for i in new_mol.bonds:
        if i[0] == h and position == -1:
            position = i[1]
        elif i[1] == h and position == -1:
            position = i[0]
        elif i[0] == h and position != -1:
            msg_program_warning("This hydrogen atom has more "
                                "than one bond!")
            position = i[1]
            if ignore_warning is False:
                return
        elif i[1] == h and position != -1:
            msg_program_warning("This hydrogen atom has more "
                                "than one bond!")
            position = i[0]
            if ignore_warning is False:
                return
    if position == -1:
        msg_program_error("This hydrogen atom has no bond!")
    # Find two more hydrogen atoms that are bound to the C
    # atom in "position"
    position2 = -1
    position3 = -1
    for i in new_mol.bonds:
        if i[0] == position and i[1] != h and \
                new_mol.atm_symbol(i[1]) == "H":
            if position2 == -1:
                position2 = i[1]
            elif position3 == -1:
                position3 = i[1]
                break
        if i[1] == position and i[0] != h and new_mol.atm_symbol(i[0]) == "H":
            if position2 == -1:
                position2 = i[0]
            elif position3 == -1:
                position3 = i[0]
                break
    # print("The two other hydrogens are:", position2, " and ", position3)
    if position3 == -1:
        msg_program_error("The C atom is not part of a CH₃ group!")

    # Calculate the vector from C to H, scale it and put a new C atom there
    xcompo = new_mol.atm_pos_x(h) - new_mol.atm_pos_x(position)
    ycompo = new_mol.atm_pos_y(h) - new_mol.atm_pos_y(position)
    zcompo = new_mol.atm_pos_z(h) - new_mol.atm_pos_z(position)
    # Scale this vector to the "right" length: We choose 1.65 as the
    # length of a slightly elongated C=C (double) bond
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * 1.65
    ycompo = (ycompo / norm) * 1.65
    zcompo = (zcompo / norm) * 1.65
    # Calculate the position of the new C atom
    xcompo = new_mol.atm_pos_x(position) + xcompo
    ycompo = new_mol.atm_pos_y(position) + ycompo
    zcompo = new_mol.atm_pos_z(position) + zcompo
    # Add this new atom
    new_mol.add_atom(Atom(sym="C", x=xcompo, y=ycompo, z=zcompo))
    # And call it's number newC for future reference
    new_c = new_mol.num_atoms() - 1

    # Again, setup the coordinates of the second new hydrogen atom:
    # We start by constructing the vector from the newly created C atom
    # to the "old one" it is bonded to
    xcompo = new_mol.atm_pos_x(position) - new_mol.atm_pos_x(new_c)
    ycompo = new_mol.atm_pos_y(position) - new_mol.atm_pos_y(new_c)
    zcompo = new_mol.atm_pos_z(position) - new_mol.atm_pos_z(new_c)
    # Scale this vector to the "right" length:
    # We choose 1.25 as the length of a slightly elongated C-H bond
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * 1.25
    ycompo = (ycompo / norm) * 1.25
    zcompo = (zcompo / norm) * 1.25
    # Next we determine the axis around which we want to rotate
    # this vector: We use the vectors from the new C atom to the first
    # one and the vector from the new C atom to the "second atom" we
    # found that was also bonded to the first C atom.

    # Create a "composite position" between the other two hydrogens.
    # We'll place the last hydrogen atom along this vector
    xcomposit = (new_mol.atm_pos_x(position2) + new_mol.atm_pos_x(
        position3)) / 2
    ycomposit = (new_mol.atm_pos_y(position2) + new_mol.atm_pos_y(
        position3)) / 2
    zcomposit = (new_mol.atm_pos_z(position2) + new_mol.atm_pos_z(
        position3)) / 2

    # Importantly, we keep its coordinates untouched once we have them,
    # so we can create the other two hydrogen atoms in the same way.
    cross_px = xcomposit - new_mol.atm_pos_x(new_c)
    cross_py = ycomposit - new_mol.atm_pos_y(new_c)
    cross_pz = zcomposit - new_mol.atm_pos_z(new_c)
    cross = np.cross([cross_px, cross_py, cross_pz], [xcompo, ycompo, zcompo])
    # Then we build the rotation matrix
    rot_matrix = build_rotation_matrix(cross, 120.0)

    vector = np.matrix([xcompo, ycompo, zcompo])
    vector = rot_matrix @ np.matrix.transpose(vector)
    vector = np.array(vector).flatten().tolist()
    xcompo = vector[0]
    ycompo = vector[1]
    zcompo = vector[2]
    # Calculate the position of the new H atom
    x_new1 = new_mol.atm_pos_x(new_c) + xcompo
    y_new1 = new_mol.atm_pos_y(new_c) + ycompo
    z_new1 = new_mol.atm_pos_z(new_c) + zcompo
    # Add this new atom
    new_mol.add_atom(Atom(sym="H", x=x_new1, y=y_new1, z=z_new1))

    # Setup the coordinates of the first new hydrogen atom:
    # We start by constructing the vector from the newly created C atom
    # to the "old one" it is bonded to
    xcompo = new_mol.atm_pos_x(position) - new_mol.atm_pos_x(new_c)
    ycompo = new_mol.atm_pos_y(position) - new_mol.atm_pos_y(new_c)
    zcompo = new_mol.atm_pos_z(position) - new_mol.atm_pos_z(new_c)
    # Scale this vector to the "right" length:
    # We choose 1.25 as the length of a slightly elongated C-H bond
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * 1.25
    ycompo = (ycompo / norm) * 1.25
    zcompo = (zcompo / norm) * 1.25

    # Build new orthogonal vector
    cross = np.cross([cross_px, cross_py, cross_pz], [xcompo, ycompo, zcompo])
    # And the build the new rotation matrix
    rot_matrix = build_rotation_matrix(cross, 240.0)

    vector = np.matrix([xcompo, ycompo, zcompo])
    vector = rot_matrix @ np.matrix.transpose(vector)
    vector = np.array(vector).flatten().tolist()

    xcompo = vector[0]
    ycompo = vector[1]
    zcompo = vector[2]
    # Calculate the position of the new H atom
    x_new1 = new_mol.atm_pos_x(new_c) + xcompo
    y_new1 = new_mol.atm_pos_y(new_c) + ycompo
    z_new1 = new_mol.atm_pos_z(new_c) + zcompo
    # Add this new atom
    new_mol.add_atom(Atom(sym="H", x=x_new1, y=y_new1, z=z_new1))

    # Calculate the vector from C to the compisite position, scale it
    # and put the last H atom there
    xcompo = xcomposit - new_mol.atm_pos_x(position)
    ycompo = ycomposit - new_mol.atm_pos_y(position)
    zcompo = zcomposit - new_mol.atm_pos_z(position)
    # Scale this vector to the "right" length:
    # We choose 1.25 as the length of a slightly elongated C-H bond
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * 1.25
    ycompo = (ycompo / norm) * 1.25
    zcompo = (zcompo / norm) * 1.25
    # Calculate the position of the new C atom
    xcompo = new_mol.atm_pos_x(position) + xcompo
    ycompo = new_mol.atm_pos_y(position) + ycompo
    zcompo = new_mol.atm_pos_z(position) + zcompo
    # Add this new atom
    new_mol.add_atom(Atom(sym="H", x=xcompo, y=ycompo, z=zcompo))

    # Delete the initial hydrogen atom h and the two hydrogen atoms
    # in position2 and position3
    # That were bound to the other carbon atom
    # EXTREMELY MESSY!!!
    for i in sorted([h, position2, position3], key=int, reverse=True):
        del new_mol.atoms[i]

    return new_mol


def replace_h_with_cho(molecule, h, orientation=0, ignore_warning=False):
    """ (Molecule) -> NoneType

    Exchange the given hydrogen atom with a keto (CHO) group.
    The "orientation" parameter is needed to decide where the new
    group "points".
    """
    # First, create a copy of the molecule to work on (and later, return)
    new_mol = copy.deepcopy(molecule)

    # Safety check: h is indeed an H atom and has only one bond
    if new_mol.atm_symbol(h) != "H":
        msg_program_warning("This is not a hydrogen atom!")
        if ignore_warning is False:
            return
    position = -1
    for i in new_mol.bonds:
        if i[0] == h and position == -1:
            position = i[1]
        elif i[1] == h and position == -1:
            position = i[0]
        elif i[0] == h and position != -1:
            msg_program_warning("This hydrogen atom has"
                                " more than one bond!")
            position = i[1]
            if ignore_warning is False:
                return
        elif i[1] == h and position != -1:
            msg_program_warning("This hydrogen atom has"
                                " more than one bond!")
            position = i[0]
            if ignore_warning is False:
                return
    if position == -1:
        msg_program_warning("This hydrogen atom has no bond!")
        return
    # Find one more atom that is bound to the atom in "position"
    position2 = -1
    for i in new_mol.bonds:
        if i[0] == position and i[1] != h:
            position2 = i[1]
        if i[1] == position and i[0] != h:
            position2 = i[0]
    if position2 == -1:
        msg_program_warning("The C atom no seems to have only one bond!")
        return
    # Calculate the vector from C to H, scale it and put the
    # replacement atom there
    xcompo = new_mol.atm_pos_x(h) - new_mol.atm_pos_x(position)
    ycompo = new_mol.atm_pos_y(h) - new_mol.atm_pos_y(position)
    zcompo = new_mol.atm_pos_z(h) - new_mol.atm_pos_z(position)
    # Scale this vector to the "right" length: We choose 110% of the sum
    # of vdW radii  as the length of a slightly elongated bond
    scale = (SymbolToRadius["C"] + SymbolToRadius["C"]) * 1.1
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * scale
    ycompo = (ycompo / norm) * scale
    zcompo = (zcompo / norm) * scale
    # Calculate the position of the replacement atom
    xcompo = new_mol.atm_pos_x(position) + xcompo
    ycompo = new_mol.atm_pos_y(position) + ycompo
    zcompo = new_mol.atm_pos_z(position) + zcompo
    # Add this new atom
    new_mol.add_atom(Atom(sym="C", x=xcompo, y=ycompo, z=zcompo))
    # And call it's number newAtom for future reference
    new_atom = new_mol.num_atoms() - 1

    # Setup the coordinates of the oxygen atom:
    # We start by constructing the vector from the newly created C atom
    # to the "old one" it is bonded to
    xcompo = new_mol.atm_pos_x(position) - new_mol.atm_pos_x(new_atom)
    ycompo = new_mol.atm_pos_y(position) - new_mol.atm_pos_y(new_atom)
    zcompo = new_mol.atm_pos_z(position) - new_mol.atm_pos_z(new_atom)
    # Scale this vector to the "right" length: We choose 110% of the sum
    # of vdW radii  as the length of a slightly elongated bond
    scale = (SymbolToRadius["O"] + SymbolToRadius["C"]) * 1.1
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * scale
    ycompo = (ycompo / norm) * scale
    zcompo = (zcompo / norm) * scale
    # Next we determine the axis around which we want to rotate this
    # vector: We use the vectors from the new N atom to the C atom and
    # the vector from the N atom to the "second atom" we found that was
    # also bonded to the C atom.
    # Importantly, we keep its coordinates untouched once we have them,
    # so we can create the other two hydrogen atoms in the same way.
    cross_px = new_mol.atm_pos_x(position2) - new_mol.atm_pos_x(new_atom)
    cross_py = new_mol.atm_pos_y(position2) - new_mol.atm_pos_y(new_atom)
    cross_pz = new_mol.atm_pos_z(position2) - new_mol.atm_pos_z(new_atom)
    cross = np.cross([cross_px, cross_py, cross_pz], [xcompo, ycompo, zcompo])

    # Then we build the rotation matrix
    rot_matrix = build_rotation_matrix(cross, 120.0)

    vector = np.matrix([xcompo, ycompo, zcompo])
    vector = rot_matrix @ np.matrix.transpose(vector)
    vector = np.array(vector).flatten().tolist()
    xcompo = vector[0]
    ycompo = vector[1]
    zcompo = vector[2]
    # Calculate the position of the new O atom
    x_new_o = xcompo
    y_new_o = ycompo
    z_new_o = zcompo

    # The, we setup the coordinates of the hydrogen atom in the same way:
    # We start by constructing the vector from the newly created C atom
    # to the "old one" it is bonded to
    xcompo = new_mol.atm_pos_x(position) - new_mol.atm_pos_x(new_atom)
    ycompo = new_mol.atm_pos_y(position) - new_mol.atm_pos_y(new_atom)
    zcompo = new_mol.atm_pos_z(position) - new_mol.atm_pos_z(new_atom)
    # Scale this vector to the "right" length: We choose 110% of the sum
    # of vdW radii  as the length of a slightly elongated bond
    scale = (SymbolToRadius["H"] + SymbolToRadius["C"]) * 1.1
    norm = math.sqrt(xcompo * xcompo + ycompo * ycompo + zcompo * zcompo)
    xcompo = (xcompo / norm) * scale
    ycompo = (ycompo / norm) * scale
    zcompo = (zcompo / norm) * scale
    # Next we determine the axis around which we want to rotate this
    # vector: We use the vectors from the new C atom to the C atom and
    # the vector from the C atom to the "second atom" we found that was
    # also bonded to the C atom.
    # Importantly, we keep its coordinates untouched once we have them,
    # so we can create the other two hydrogen atoms in the same way.
    cross_px = new_mol.atm_pos_x(position2) - new_mol.atm_pos_x(new_atom)
    cross_py = new_mol.atm_pos_y(position2) - new_mol.atm_pos_y(new_atom)
    cross_pz = new_mol.atm_pos_z(position2) - new_mol.atm_pos_z(new_atom)
    cross = np.cross([cross_px, cross_py, cross_pz], [xcompo, ycompo, zcompo])
    # Then we build the rotation matrix
    rot_matrix = build_rotation_matrix(cross, 240.0)

    vector = np.matrix([xcompo, ycompo, zcompo])
    vector = rot_matrix @ np.matrix.transpose(vector)
    vector = np.array(vector).flatten().tolist()
    xcompo = vector[0]
    ycompo = vector[1]
    zcompo = vector[2]
    # Calculate the position of the nextnew H atom
    x_new_h = xcompo
    y_new_h = ycompo
    z_new_h = zcompo

    if orientation != 0:
        # Next, we re-use the rotated coordinates and rotate them further
        # by 120 or 240 degrees around the C-C bond
        cross[0] = new_mol.atm_pos_x(position) - new_mol.atm_pos_x(new_atom)
        cross[1] = new_mol.atm_pos_y(position) - new_mol.atm_pos_y(new_atom)
        cross[2] = new_mol.atm_pos_z(position) - new_mol.atm_pos_z(new_atom)
        # We build the second rotation matrix
        if orientation == 1:
            rot_matrix = build_rotation_matrix(cross, 120.0)
        else:
            rot_matrix = build_rotation_matrix(cross, 240.0)

        vector = np.matrix([x_new_o, y_new_o, z_new_o])
        vector = rot_matrix @ np.matrix.transpose(vector)
        vector = np.array(vector).flatten().tolist()
        x_new_o = vector[0]
        y_new_o = vector[1]
        z_new_o = vector[2]

        vector = np.matrix([x_new_h, y_new_h, z_new_h])
        vector = rot_matrix @ np.matrix.transpose(vector)
        vector = np.array(vector).flatten().tolist()
        x_new_h = vector[0]
        y_new_h = vector[1]
        z_new_h = vector[2]

    # Calculate the position of the new O and H atoms and add them
    x_new_o = new_mol.atm_pos_x(new_atom) + x_new_o
    y_new_o = new_mol.atm_pos_y(new_atom) + y_new_o
    z_new_o = new_mol.atm_pos_z(new_atom) + z_new_o
    new_mol.add_atom(Atom(sym="O", x=x_new_o, y=y_new_o, z=z_new_o))
    x_new_h = new_mol.atm_pos_x(new_atom) + x_new_h
    y_new_h = new_mol.atm_pos_y(new_atom) + y_new_h
    z_new_h = new_mol.atm_pos_z(new_atom) + z_new_h
    new_mol.add_atom(Atom(sym="H", x=x_new_h, y=y_new_h, z=z_new_h))

    # Delete the initial hydrogen atom h # EXTREMELY MESSY!!!
    del new_mol.atoms[h]
    return new_mol


############################################################################
#                                                                          #
# This is the part of the program where the cmd line arguments are defined #
#                                                                          #
############################################################################

parser = argparse.ArgumentParser(
    description="MICEPES: Method for the Incremental Construction and "
                "Exploration of the Potential Energy Surface",
    epilog="recognised filetypes: gaussian, orca, turbomole, xyz")
parser.add_argument("-v", "--verbosity", help="increase output verbosity",
                    type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument("-o", "--output",
                    help="type of output file to be written: xyz file or just "
                         "cartesian coordinates without header.",
                    choices=["xyz", "cart"],
                    default="cart")
parser.add_argument("-g", "--group", help="replacement group",
                    choices=["methyl", "ethenyl", "oh", "oh2", "nh2", "nh3",
                             "cho", "f", "cl", "br", "i"],
                    default="methyl")
parser.add_argument("inputfile", metavar='file',
                    help="input file(s) with molecular structure")
parser.add_argument("-r", "--replace", type=int, nargs='+',
                    help="list of individual hydrogen atoms to replace")
parser.add_argument("-a", "--alloncarbon", type=int, nargs='+',
                    help="list of carbon atoms whose hydrogen atoms "
                         "should be replaced")
parser.add_argument("-t", "--terminal", type=int,
                    help="number of terminal hydrogen atoms to replace")

args = parser.parse_args()


###############################################################################
#                                                                             #
# The main part of the program starts here                                    #
#                                                                             #
###############################################################################

def main():
    # Print GPL v3 statement and program header
    prg_start_time = time.time()
    if args.verbosity >= 1:
        msg_program_header("MICEPES", 1.0)

    molecule = Molecule("Input Structure", 0)

    extract_molecular_data(args.inputfile, molecule, verbosity=args.verbosity,
                           read_coordinates=True, read_bond_orders=True,
                           build_angles=True, build_dihedrals=True)

    list_of_hydrogens = []

    if args.replace is not None:
        if args.verbosity >= 3:
            print("\nAdding hydrogen atoms from the --replace key")
        for i in args.replace:
            if molecule.atoms[i - 1].symbol() == "H":
                list_of_hydrogens.append(i)
                if args.verbosity >= 3:
                    print("Hydrogen atom ", i, " added to the list of "
                                               "atoms for replacement.")
            else:
                msg_program_warning("Atom " + str(i) + " in the input "
                                                       "is not hydrogen!")
    if args.alloncarbon is not None:
        if args.verbosity >= 3:
            print("\nAdding hydrogen atoms from the --alloncarbon key")
        for i in args.alloncarbon:
            if molecule.atm_symbol(i - 1) == "C":
                if args.verbosity >= 3:
                    print("Adding the hydrogen atoms bonded to carbon atom ",
                          i)
                for j in molecule.bonds:
                    at1 = j[0]  # First atom in the bond
                    at2 = j[1]  # Second atom in the bond
                    if at1 + 1 == i and molecule.atm_symbol(at2) == "H":
                        list_of_hydrogens.append(at2 + 1)
                        if args.verbosity >= 3:
                            print("Hydrogen atom ", at2 + 1,
                                  " added to the list of atoms"
                                  " for replacement.")
                    elif at2 + 1 == i and molecule.atm_symbol(at1) == "H":
                        list_of_hydrogens.append(at1 + 1)
                        if args.verbosity >= 3:
                            print("Hydrogen atom ", at1 + 1,
                                  " added to the list of atoms"
                                  " for replacement.")
            else:
                msg_program_warning(
                    "Atom " + str(i) + " in the input is not carbon!")
    if args.terminal is not None:
        if args.verbosity >= 3:
            print("\nAdding hydrogen atoms from the --terminal key")
        # Count backwards and add the given number of hydrogen atoms to list
        counter = 0
        for i in range(molecule.num_atoms(), 0, -1):
            if counter < args.terminal and \
                    molecule.atm_symbol(i - 1) == "H":
                if args.verbosity >= 3:
                    print("Hydrogen atom ", i,
                          " added to the list of atoms for replacement.")
                list_of_hydrogens.append(i)
                counter += 1
    elif args.replace is None and args.alloncarbon is None:
        # If no instructions were given at all: default to replacing
        # the terminal 3 hydrogens
        if args.verbosity >= 2:
            print("\nDefaulting to replacement of the last 3 hydrogen atoms")
        counter = 0
        for i in range(molecule.num_atoms(), 0, -1):
            if counter < 3 and molecule.atm_symbol(i - 1) == "H":
                if args.verbosity >= 3:
                    print("Hydrogen atom ", i,
                          " added to the list of atoms for replacement.")
                list_of_hydrogens.append(i)
                counter += 1

    # Abort if we somehow managed to end up with an empty list
    if not list_of_hydrogens:
        msg_program_error("No atoms selected for replacement")

    # Ensuring our list is unique
    list_of_hydrogens = set(list_of_hydrogens)
    if args.verbosity >= 2:
        print("\nThere are ", len(list_of_hydrogens),
              "hydrogen atoms that have been selected for replacement.")
        if args.verbosity >= 3:
            print("Hydrogen atoms: ", list_of_hydrogens)

    for j, i in enumerate(list_of_hydrogens):
        filesuffix = str(j + 1).zfill(
            int(math.ceil(math.log(len(list_of_hydrogens), 10))))
        filename, file_extension = os.path.splitext(args.inputfile)
        # Look up which replacement was selected
        if args.group == "methyl":
            newmol = replace_h_with_tetrahedral(molecule, h=(i - 1))
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            msg = newmol.print_mol(output=args.output,
                                   file=filename + "-" + filesuffix,
                                   comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "ethenyl":
            newmol = replace_h_with_ethenyl(molecule, h=(i - 1))
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            msg = newmol.print_mol(output=args.output,
                                   file=filename + "-" + filesuffix,
                                   comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "cho":
            # If we replace H by CHO, we create the three possible
            #  orientations at the same time
            for l, k in enumerate(["a", "b", "c"]):
                newmol = replace_h_with_cho(molecule, h=(i - 1), orientation=l)
                newmol.name = "Molecule " + filesuffix + " out of " + str(
                    len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                    i) + " replaced"
                msg = newmol.print_mol(output=args.output,
                                       file=filename + "-" + filesuffix + k,
                                       comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "nh3":
            newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                replacement="N")
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            msg = newmol.print_mol(output=args.output,
                                   file=filename + "-" + filesuffix,
                                   comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "oh":
            # If we replace H by OH, we create the three possible
            #  orientations at the same time
            for l, k in enumerate(["a", "b", "c"]):
                newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                    replacement="O",
                                                    add_h=1,
                                                    orientation=l)
                newmol.name = "Molecule " + filesuffix + " out of " + str(
                    len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                    i) + " replaced"
                msg = newmol.print_mol(output=args.output,
                                       file=filename + "-" + filesuffix + k,
                                       comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "nh2":
            # If we replace H by NH₂, we create the three possible
            #  orientations at the same time
            for l, k in enumerate(["a", "b", "c"]):
                newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                    replacement="N",
                                                    add_h=2,
                                                    orientation=l)
                newmol.name = "Molecule " + filesuffix + " out of " + str(
                    len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                    i) + " replaced"
                msg = newmol.print_mol(output=args.output,
                                       file=filename + "-" + filesuffix + k,
                                       comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "oh2":
            # If we replace H by OH₂, we create the three possible
            #  orientations at the same time
            for l, k in enumerate(["a", "b", "c"]):
                newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                    replacement="O",
                                                    add_h=2,
                                                    orientation=l)
                newmol.name = "Molecule " + filesuffix + " out of " + str(
                    len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                    i) + " replaced"
                msg = newmol.print_mol(output=args.output,
                                       file=filename + "-" + filesuffix + k,
                                       comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "f":
            newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                replacement="F",
                                                add_h=0)
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            msg = newmol.print_mol(output=args.output,
                                   file=filename + "-" + filesuffix,
                                   comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "cl":
            newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                replacement="Cl",
                                                add_h=0)
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            msg = newmol.print_mol(output=args.output,
                                   file=filename + "-" + filesuffix,
                                   comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "br":
            newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                replacement="Br",
                                                add_h=0)
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            msg = newmol.print_mol(output=args.output,
                                   file=filename + "-" + filesuffix,
                                   comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)
        elif args.group == "i":
            newmol = replace_h_with_tetrahedral(molecule, h=(i - 1),
                                                replacement="I",
                                                add_h=0)
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(list_of_hydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            msg = newmol.print_mol(output=args.output,
                                   file=filename + "-" + filesuffix,
                                   comment=newmol.name)
            if args.verbosity >= 3:
                print("\nNew Structure:")
                print(msg)

    # Print program footer
    if args.verbosity >= 1:
        msg_program_footer(prg_start_time)


if __name__ == '__main__':
    main()
