###############################################################################
#                                                                             #
# This is the part of the program where the methods for reading QM files are  #
# defined                                                                     #
#                                                                             #
###############################################################################

import numpy as np

from typing import Any

from constants import *
from conversions import *
from messages import *
from atom import Atom
from molecule import Molecule, build_molecular_dihedrals, \
    build_molecular_angles, build_bond_orders


def print_add_atom(symbol: Any, xcoord: Any, ycoord: Any, zcoord: Any):
    """
    This method will print a status message after an atom is being added to the
    current molecule. Note that this method will convert
    coordinates into floats before printing if necessary.

    :param symbol: The atomic symbol of the atom
    :param xcoord: Cartesian x-coordinate.
    :param ycoord: Cartesian y-coordinate.
    :param zcoord: Cartesian z-coordinate.
    :return: None
    """
    print(
        " Adding atom: {:<3} {: 13.8f} {: 13.8f}"
        " {: 13.8f} to molecule.".format(str(symbol), float(xcoord),
                                         float(ycoord), float(zcoord)))


def print_found_atom(symbol: Any, xcoord: Any, ycoord: Any, zcoord: Any):
    """
    This method will print a status message after an atom has been found in the
    input stream while reading a file. Note that this method will convert
    coordinates into floats before printing if necessary.

    :param symbol: The atomic symbol of the atom
    :param xcoord: Cartesian x-coordinate.
    :param ycoord: Cartesian y-coordinate.
    :param zcoord: Cartesian z-coordinate.
    :return: None
    """
    print(
        " Found atom: {:<3} {: 13.8f} {: 13.8f}"
        " {: 13.8f} while reading file.".format(str(symbol), float(xcoord),
                                                float(ycoord), float(zcoord)))


def read_gauss_bond_orders(filename, molecule, verbosity=0):
    bo = []
    f = open(filename, 'r')
    for line in f:
        if line.find(
                "Atomic Valencies and Mayer Atomic Bond Orders:") != -1:
            bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
            if verbosity >= 2:
                print(
                    "\nAtomic Valencies and Mayer Atomic"
                    " Bond Orders found, reading data")
            bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
            columns = ""
            while True:
                read_buffer = f.__next__()
                # Check if the whole line is integers only (Header line)
                if is_int("".join(read_buffer.split())) is True:
                    # And use this information to label the columns
                    columns = read_buffer.split()
                # If we get to the Löwdin charges, we're done reading
                elif read_buffer.find("Lowdin Atomic Charges") != -1:
                    break
                else:
                    row = read_buffer.split()
                    j = 1
                    for i in columns:
                        j = j + 1
                        bo[int(row[0]) - 1][int(i) - 1] = float(row[j])
    f.close()
    # if verbosity >= 3:
    #     print("\nBond Orders:")
    #     np.set_printoptions(suppress=True)
    #     np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    #     print(bo)
    build_bond_orders(molecule, bo, verbosity=verbosity)


def read_orca_bond_orders(filename, molecule, verbosity=0):
    bo = []
    f = open(filename, 'r')
    for line in f:
        if line.find("Mayer bond orders larger than 0.1") != -1:
            bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
            if verbosity >= 2:
                print(
                    "\nMayer bond orders larger than 0.1 found, reading data")
            bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
            while True:
                read_buffer = f.__next__()
                # Check if the whole line isn't empty (in that case we're done)
                if read_buffer and read_buffer.strip():
                    # Break the line into pieces
                    read_buffer = read_buffer[1:].strip()
                    read_buffer = read_buffer.split("B")
                    for i in read_buffer:
                        bondpair1 = int(i[1:4].strip())
                        bondpair2 = int(i[8:11].strip())
                        order = i[-9:].strip()
                        bo[bondpair1][bondpair2] = order
                        bo[bondpair2][bondpair1] = order
                else:
                    break
    f.close()
    # if verbosity >= 3:
    #     print("\nBond Orders:")
    #     np.set_printoptions(suppress=True)
    #     np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    #     print(bo)
    build_bond_orders(molecule, bo, verbosity=verbosity)


def read_xyz_coord(filename, molecule, verbosity=0):
    geom = []
    f = open(filename, 'r')
    # Now examine every line until a line that doesn't conform to the
    #  template or EOF is found
    mark_for_delete = 0
    for line in f:
        # Check if line conforms to template
        read_buffer = line.split()
        if len(read_buffer) != 4:
            # If there are not *exactly* four entries on this line, we mark
            #  the current geometry for deletion (but delete only if we find
            #  more atoms).
            mark_for_delete = 1
        else:
            if mark_for_delete == 1:
                del geom[:]
                mark_for_delete = 0
            if is_atom_symbol(read_buffer[0]) and is_float(
                    read_buffer[1]) and is_float(read_buffer[2])\
                    and is_float(read_buffer[3]):
                geom.append(read_buffer)
                if verbosity >= 3:
                    if len(geom) == 1:
                        print(
                            "New structure found, starting to read structure")
                        print_found_atom(
                            read_buffer[0], float(read_buffer[1]),
                            float(read_buffer[2]),
                            float(read_buffer[3]))
                    else:
                        print_found_atom(
                            read_buffer[0], float(read_buffer[1]),
                            float(read_buffer[2]),
                            float(read_buffer[3]))
            else:
                # If the entries aren't an atomic symbol and 3 coords, we
                # delete the current geometry and start fresh.
                mark_for_delete = 1

    if verbosity >= 2:
        print("\nReading of geometry finished.")
        print("\nAdding atoms to WellFARe molecule: ", molecule.name)
    for i in geom:
        molecule.add_atom(Atom(sym=i[0], x=float(i[1]), y=float(i[2]),
                               z=float(i[3])))
        if verbosity >= 3:
            print_add_atom(i[0], i[1], i[2], i[3])
    f.close()


def read_turbo_coord(filename, molecule, verbosity=0):
    # Reading from Turbomole's aoforce file
    geom = []
    f = open(filename, 'r')
    for line in f:
        if line.find(
                "atomic coordinates            atom    charge  isotop") != -1:
            if verbosity >= 2:
                print("\nCartesian Coordinates found")
            del geom[:]
            while True:
                read_buffer = f.__next__()
                if read_buffer and read_buffer.strip():
                    geom.append(read_buffer)
                    if verbosity >= 3:
                        read_buffer = read_buffer.split()
                        print_found_atom(
                            NumberToSymbol[int(float(read_buffer[4]))],
                            bohr_to_ang(float(read_buffer[0])),
                            bohr_to_ang(float(read_buffer[1])),
                            bohr_to_ang(float(read_buffer[2])))
                else:
                    break
    if verbosity >= 2:
        print("\nReading of geometry finished.")
        print("\nAdding atoms to WellFARe molecule: ", molecule.name)
    for i in geom:
        read_buffer = i.split()
        molecule.add_atom(Atom(charge=int(float(read_buffer[4])),
                               x=bohr_to_ang(float(read_buffer[0])),
                               y=bohr_to_ang(float(read_buffer[1])),
                               z=bohr_to_ang(float(read_buffer[2]))))
        if verbosity >= 3:
            print_add_atom(
                NumberToSymbol[int(float(read_buffer[4]))],
                bohr_to_ang(float(read_buffer[0])),
                bohr_to_ang(float(read_buffer[1])),
                bohr_to_ang(float(read_buffer[2])))
    f.close()


def read_orca_coord(filename, molecule, verbosity=0):
    geom = []
    f = open(filename, 'r')
    for line in f:
        if line.find("CARTESIAN COORDINATES (ANGSTROEM)") != -1:
            if verbosity >= 2:
                print("\nCartesian Coordinates found")
            del geom[:]
            read_buffer = f.__next__()
            while True:
                read_buffer = f.__next__()
                if read_buffer and read_buffer.strip():
                    geom.append(read_buffer)
                    if verbosity >= 3:
                        read_buffer = read_buffer.split()
                        print_found_atom(read_buffer[0], float(read_buffer[1]),
                                         float(read_buffer[2]),
                                         float(read_buffer[3]))
                else:
                    break
    if verbosity >= 2:
        print("\nReading of geometry finished.")
        print("\nAdding atoms to WellFARe molecule: ", molecule.name)
    for i in geom:
        read_buffer = i.split()
        molecule.add_atom(
            Atom(sym=read_buffer[0], x=float(read_buffer[1]),
                 y=float(read_buffer[2]),
                 z=float(read_buffer[3])))
        if verbosity >= 3:
            print_add_atom(read_buffer[0],
                           read_buffer[1],
                           read_buffer[2],
                           read_buffer[3])
    f.close()


def read_gauss_coord(filename, molecule, verbosity=0):
    geom = []
    f = open(filename, 'r')
    for line in f:
        if line.find("Input orientation:") != -1:
            if verbosity >= 2:
                print("\nInput orientation found, reading coordinates")
            del geom[:]
            for i in range(0, 4):
                read_buffer = f.__next__()
            while True:
                read_buffer = f.__next__()
                if read_buffer.find("-----------") == -1:
                    geom.append(read_buffer)
                    if verbosity >= 3:
                        read_buffer = read_buffer.split()
                        print_found_atom(
                            NumberToSymbol[int(read_buffer[1])],
                            float(read_buffer[3]), float(read_buffer[4]),
                            float(read_buffer[5]))
                else:
                    break
    f.close()
    if len(geom) == 0:
        # For some reason, we don't have the "input orientation" printed, let's try and read from the
        # standard orientation instead
        f = open(filename, 'r')
        for line in f:
            if line.find("Standard orientation:") != -1:
                if verbosity >= 2:
                    print(
                        "\nStandard orientation found, reading coordinates")
                del geom[:]
                for i in range(0, 4):
                    read_buffer = f.__next__()
                while True:
                    read_buffer = f.__next__()
                    if read_buffer.find("-----------") == -1:
                        geom.append(read_buffer)
                        if verbosity >= 3:
                            read_buffer = read_buffer.split()
                            print_found_atom(
                                NumberToSymbol[int(read_buffer[1])],
                                float(read_buffer[3]), float(read_buffer[4]),
                                float(read_buffer[5]))
                    else:
                        break
        f.close()
    if verbosity >= 2:
        print("\nReading of geometry finished.")
        print("\nAdding atoms to WellFARe molecule: ", molecule.name)
    for i in geom:
        read_buffer = i.split()
        molecule.add_atom(
            Atom(charge=int(read_buffer[1]), x=float(read_buffer[3]),
                 y=float(read_buffer[4]),
                 z=float(read_buffer[5])))
        if verbosity >= 3:
            print_add_atom(
                NumberToSymbol[int(read_buffer[1])], read_buffer[3],
                read_buffer[4], read_buffer[5])
    return


def extract_molecular_data(filename, molecule, verbosity=0,
                           read_coordinates=True, read_bond_orders=True,
                           build_angles=False, build_dihedrals=False):
    if verbosity >= 1:
        print("\nSetting up WellFARe molecule {} from file {}.".format(
            molecule.name, filename))

    # Try filename for readability first
    try:
        f = open(filename, 'r')
        f.close()
    except OSError:
        msg_program_error("Cannot open " + filename + " for reading.")

    # Next, establish which kind of file we're dealing with
    # Determine which QM program we're dealing with
    f = open(filename, 'r')
    program = "N/A"
    for line in f:
        if line.find("Entering Gaussian System, Link 0") != -1:
            if verbosity >= 2:
                print("Reading Gaussian output file: ", filename)
            program = "gauss"
            break
        elif line.find("* O   R   C   A *") != -1:
            if verbosity >= 2:
                print("Reading Orca output file: ", filename)
            program = "orca"
            break
        elif line.find("T U R B O M O L E") != -1:
            if verbosity >= 2:
                print("Reading Turbomole output file: ", filename)
            program = "turbomole"
            break
    f.close()
    if program == "N/A":
        if verbosity >= 2:
            print("Reading xyz file: ", filename)
        if verbosity >= 3:
            print("(Note that this is also a fallback)")
        program = "xyz"

    # Check if we need to read coordinates
    if read_coordinates is True:
        if program == "gauss":
            read_gauss_coord(filename, molecule, verbosity=verbosity)
        elif program == "orca":
            read_orca_coord(filename, molecule, verbosity=verbosity)
        elif program == "turbomole":
            read_turbo_coord(filename, molecule, verbosity=verbosity)
        else:
            read_xyz_coord(filename, molecule, verbosity=verbosity)

    # Check if we need to read bond orders
    if read_bond_orders is True:
        if program == "gauss":
            read_gauss_bond_orders(filename, molecule, verbosity=verbosity)
        elif program == "orca":
            read_orca_bond_orders(filename, molecule, verbosity=verbosity)
        elif program == "turbomole":
            build_bond_orders(molecule, [], verbosity=verbosity)
        else:
            build_bond_orders(molecule, [], verbosity=verbosity)

    # Check if we need to build bond angles
    if build_angles is True:
        build_molecular_angles(molecule, verbosity=verbosity)

    # Check if we need to build dihedrals
    if build_dihedrals is True:
        build_molecular_dihedrals(molecule, verbosity=verbosity)


def main():
    prg_start_time = time.time()
    # Create an example molecule and add some atoms
    example = Molecule("Example Molecule")
    extract_molecular_data("../examples/g09-h3b-nh3.log", example, verbosity=3)
    # extract_molecular_data("../examples/pdb100d.xyz", example, verbosity=3)
    print(example.print_mol(output="gauss"))
    msg_program_footer(prg_start_time)


if __name__ == '__main__':
    main()
