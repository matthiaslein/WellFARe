###############################################################################
#                                                                             #
# This is the part of the program where the methods for reading QM files are  #
# defined                                                                     #
#                                                                             #
###############################################################################

import numpy as np

from typing import Any

from wellfare.constants import *
from wellfare.conversions import *
from wellfare.messages import *
from wellfare.atom import Atom
from wellfare.molecule import Molecule, build_molecular_dihedrals, \
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
            if is_atom_symbol(read_buffer[0]) and is_float(read_buffer[1])\
                    and is_float(read_buffer[2]) and is_float(read_buffer[3]):
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


def read_adf_coord(filename, molecule, verbosity=3):
    geom = []
    f = open(filename, 'r')
    for line in f:
        # if line.find(" Coordinates (Cartesian)") != -1:
        #     if verbosity >= 2:
        #         print("\nCartesian Coordinates found")
        #     del geom[:]
        #     for i in range(0, 5):
        #         read_buffer = f.__next__()
        #     while True:
        #         read_buffer = f.__next__()
        #         if read_buffer.find("-----------") == -1:
        #             read_buffer = read_buffer.split()
        #             geom.append([read_buffer[1],
        #                     float(read_buffer[5]), float(read_buffer[6]),
        #                     float(read_buffer[7])])
        #             if verbosity >= 3:
        #                 print_found_atom(geom[-1][0], geom[-1][1], geom[-1][2],
        #                                  geom[-1][3])
        #         else:
        #             break
        if line.find(" FRAGMENTS") != -1:
            if verbosity >= 2:
                print("\nCartesian Coordinates found")
            del geom[:]
            # Move 2 lines forward, the data begins there
            for i in range(0, 2):
                read_buffer = f.__next__()
            while True:
                read_buffer = f.__next__()
                if read_buffer.isspace() is True:
                    break
                else:
                    read_buffer = read_buffer.split()
                    geom.append([read_buffer[1],
                                 float(read_buffer[4]), float(read_buffer[5]),
                                 float(read_buffer[6])])
                    if verbosity >= 3:
                        print_found_atom(geom[-1][0], geom[-1][1], geom[-1][2],
                                         geom[-1][3])

    if verbosity >= 2:
        print("\nReading of geometry finished.")
        print("\nAdding atoms to WellFARe molecule: ", molecule.name)
    for read_buffer in geom:
        molecule.add_atom(
            Atom(sym=read_buffer[0], x=read_buffer[1],
                 y=read_buffer[2],
                 z=read_buffer[3]))
        if verbosity >= 3:
            print_add_atom(read_buffer[0],
                           read_buffer[1],
                           read_buffer[2],
                           read_buffer[3])
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
                            number_to_symbol[int(float(read_buffer[4]))],
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
                number_to_symbol[int(float(read_buffer[4]))],
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
                            number_to_symbol[int(read_buffer[1])],
                            float(read_buffer[3]), float(read_buffer[4]),
                            float(read_buffer[5]))
                else:
                    break
    f.close()
    if len(geom) == 0:
        # For some reason, we don't have the "input orientation" printed,
        #  let's try and read from the standard orientation instead
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
                                number_to_symbol[int(read_buffer[1])],
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
                number_to_symbol[int(read_buffer[1])], read_buffer[3],
                read_buffer[4], read_buffer[5])
    return


def read_gauss_qm_energy(filename, molecule, verbosity=0):
    f = open(filename, 'r')
    qm_energies = []
    for line in f:
        if line.find("SCF Done:") != -1:
            readBuffer = line.split()
            readBuffer = float(readBuffer[4])
            qm_energies.append(readBuffer)
            if verbosity >= 3:
                print("\nEnergy from SCF cycle found")
                print("SCF:", str(readBuffer))
        elif line.find(" EUMP2 = ") != -1:
            readBuffer = line.split()
            readBuffer = float(readBuffer[5].replace('D', 'E'))
            qm_energies.append(readBuffer)
            if verbosity >= 3:
                print("\nEnergy from MP2 calculation found")
                print("MP2:", str(readBuffer))
        elif line.find(" CCSD(T)= ") != -1:
            readBuffer = line.split()
            readBuffer = float(readBuffer[1].replace('D', 'E'))
            qm_energies.append(readBuffer)
            if verbosity >= 3:
                print("\nEnergy from CCSD(T) calculation found")
                print("CCSD(T):", str(readBuffer))

    # Take the last QM energy from the list and assign that value as the
    # QM equilibrium energy of the molecule
    qm_energy = qm_energies[len(qm_energies) - 1]
    molecule.qm_energy = qm_energy
    if verbosity >= 2:
        print("\nReading of QM equilibrium energy complete")
        if verbosity >= 3:
            print("Ee_QM = " + str(qm_energy))
    f.close()


def read_orca_qm_energy(filename, molecule, verbosity=0):
    f = open(filename, 'r')
    qm_energies = []
    for line in f:
        if line.find("FINAL SINGLE POINT ENERGY") != -1:
            if verbosity >= 3:
                print("\nSingle point energy found")
                print(str(line))
            qm_energies.append(line)
    i = qm_energies[len(qm_energies) - 1]
    read_buffer = i.split()
    qm_energy_on_file = float(read_buffer[4])
    molecule.qm_energy = qm_energy_on_file
    if verbosity >= 2:
        print("\nReading of QM equilibrium energy complete")
        if verbosity >= 3:
            print("Ee_QM = " + str(qm_energy_on_file))
    f.close()


def read_turbomole_qm_energy(filename, molecule, verbosity=0):
    f = open(filename, 'r')
    qm_energies = []
    for line in f:
        if line.find("         *    SCF-energy") != -1:
            if verbosity >= 3:
                print("\nSCF energy found")
                print(str(line))
            qm_energies.append(line)
    i = qm_energies[len(qm_energies) - 1]
    read_buffer = i.split()
    qm_energy_on_file = float(read_buffer[3])
    molecule.qm_energy = qm_energy_on_file
    if verbosity >= 2:
        print("\nReading of QM equilibrium energy complete")
        if verbosity >= 3:
            print("Ee_QM = " + str(qm_energy_on_file))
    f.close()


def read_adf_qm_energy(filename, molecule, verbosity=0):
    f = open(filename, 'r')
    qm_energies = []
    for line in f:
        if line.find("  Total Bonding Energy:") != -1:
            if verbosity >= 3:
                print("\n Total Bonding Energy: found")
                print(str(line))
            qm_energies.append(line)
    i = qm_energies[len(qm_energies) - 1]
    read_buffer = i.split()
    qm_energy_on_file = float(read_buffer[3])
    molecule.qm_energy = qm_energy_on_file
    if verbosity >= 2:
        print("\nReading of QM equilibrium energy complete")
        if verbosity >= 3:
            print("Ee_QM = " + str(qm_energy_on_file))
    f.close()


def read_gauss_force_constants(filename, molecule, verbosity=0):
    H = []
    f = open(filename, 'r')
    for line in f:
        if line.find("Force constants in Cartesian coordinates") != -1:
            if verbosity >= 2:
                print(
                    "\nForce constants in Cartesian coordinates, reading data")
            H = np.zeros((3 * molecule.num_atoms(), 3 * molecule.num_atoms()))
            while True:
                readBuffer = f.__next__()
                # Check if the whole line is integers only (Header line)
                if is_int("".join(readBuffer.split())) == True:
                    # And use this information to label the columns
                    columns = readBuffer.split()
                # Once we find one of these statements, we're done reading
                elif readBuffer.find(
                        "FormGI is forming") != -1 or readBuffer.find(
                        "Cartesian forces in FCRed") != -1 or readBuffer.find(
                    "Final forces over variables") != -1:
                    break
                else:
                    row = readBuffer.split()
                    for i in range(0, len(row) - 1):
                        H[int(row[0]) - 1][int(columns[i]) - 1] = row[
                            i + 1].replace('D', 'E')
                        H[int(columns[i]) - 1][int(row[0]) - 1] = row[
                            i + 1].replace('D', 'E')
    # Store H as the QM calculated Hessian for this molecule
    molecule.set_hessian(H)
    if verbosity >= 3:
        print(
            "\nForce constants in Cartesian coordinates (Input orientation):")
        # np.set_printoptions(suppress=True)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(H)
    f.close()


def read_orca_force_constants(filename, molecule, verbosity=0):
    H = np.zeros((3 * molecule.num_atoms(), 3 * molecule.num_atoms()))
    molecule.set_hessian(H)
    if verbosity >= 3:
        print(
            "\nForce constants in Cartesian coordinates:")
        # np.set_printoptions(suppress=True)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(H)


def read_turbo_force_constants(filename, molecule, verbosity=0):
    H = []
    f = open(filename, 'r')
    for line in f:
        if line.find("translational & rotational space projected out") != -1:
            if verbosity >= 2:
                print(
                    "\nForce constants in Cartesian coordinates, reading data")
            H = np.zeros((3 * molecule.num_atoms(), 3 * molecule.num_atoms()))
            for i in range(0, 2):
                readBuffer = f.__next__()
            while True:
                readBuffer = f.__next__()
                # Check if the line is labeled: "ATOM" (Header line)
                if readBuffer.find("ATOM") != -1:
                    # And use this information to label the columns
                    columns = []
                    columns.append(int(readBuffer.split()[1]))
                    if len(readBuffer.split()) > 3:
                        columns.append(int(readBuffer.split()[3]))
                # Look for the lines below the header lines
                elif readBuffer.find(
                        "dx        dy        dz") != -1 or readBuffer.isspace() == True:
                    # And then we do nothing with this line
                    pass
                # Look for lines that begin with an atom number
                elif len(readBuffer.split()) > 0 and is_int(readBuffer.split()[0]) == True:
                    # And then we use that bit to set the row number
                    row = int(readBuffer.split()[0])
                    rowAdd = 0
                    readRawData = readBuffer.replace("-", " -")
                    readData = readRawData.split()[3:]
                    for j, i in enumerate(readData):
                        H[(row - 1) * 3][((columns[0] - 1) * 3) + j] = i
                        H[((columns[0] - 1) * 3) + j][(row - 1) * 3] = i
                # Once we find one of these statements, we're done reading
                elif readBuffer.find(
                        "*** projected hessian written onto") != -1:
                    break
                else:
                    rowAdd += 1
                    readRawData = readBuffer.replace("-", " -")
                    readData = readRawData.split()[1:]
                    for j, i in enumerate(readData):
                        H[((row - 1) * 3) + rowAdd][
                            ((columns[0] - 1) * 3) + j] = i
                        H[((columns[0] - 1) * 3) + j][
                            ((row - 1) * 3) + rowAdd] = i

    molecule.set_hessian(H)  # Store H as the QM calculated Hessian for this molecule
    if verbosity >= 3:
        print(
            "\nForce constants in Cartesian coordinates (translational & rotational space projected out):")
        # np.set_printoptions(suppress=True)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(H)
    f.close()


def read_adf_force_constants(filename, molecule, verbosity=0):
    H = []
    f = open(filename, 'r')
    for line in f:
        if line.find("HESSIAN : AFTER SYM EXPAND & DIAGONAL CORRECTION") != -1:
            if verbosity >= 2:
                print(
                    "\nForce constants in Cartesian coordinates, reading data")
            # skip 3 lines ahead, that's where the data starts
            for i in range(0, 3):
                readBuffer = f.__next__()
            while True:
                readBuffer = f.__next__()
                if readBuffer.find("=====") != -1:
                    break
                else:
                    h_line=[float(i) for i in readBuffer.split()[3:]]
                    H.append(h_line)
    # Store H as the QM calculated Hessian for this molecule
    molecule.set_hessian(H)
    if verbosity >= 3:
        print(
            "\nForce constants in Cartesian coordinates (Input orientation):")
        # np.set_printoptions(suppress=True)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print(H)
    f.close()


def read_gauss_frequencies(filename, molecule, verbosity=0):
    freq = []
    f = open(filename, 'r')
    for line in f:
        # The line below resets the frequency list in files that contain more than one frequency calculation
        if line.find(" Harmonic frequencies (cm**-1),") != -1:
            freq = []
        # The line below actually reads the frequencies
        if line.find(" Frequencies -- ") != -1:
            if verbosity >= 2:
                print("\nReading frequencies from file.")
            readBuffer = line.split()
            for i in readBuffer[2:]:
                freq.append(float(i))
    if verbosity >= 3:
        print("\nFrequencies:")
        print(freq)
    f.close()
    molecule.frequencies = freq


def read_orca_frequencies(filename, molecule, verbosity=0):
    freq = []
    f = open(filename, 'r')
    for line in f:
        if line.find("VIBRATIONAL FREQUENCIES") != -1:
            if verbosity >= 2:
                print("\nReading frequencies from file.")
            while True:
                readBuffer = f.__next__()
                # Check if the whole line is integers only (Header line)
                if readBuffer.find("cm**-1") != -1:
                    if math.fabs(float(readBuffer.split()[1])) > 1.0:
                        freq.append(float(readBuffer.split()[1]))
                # Once we find this statement, we're done reading
                elif readBuffer.find("NORMAL MODES") != -1:
                    break
    if verbosity >= 3:
        print("\nFrequencies:")
        print(freq)
    f.close()
    molecule.frequencies = freq


def read_turbo_frequencies(filename, molecule, verbosity=0):
    freq = []
    f = open(filename, 'r')
    for line in f:
        if line.find("VIBRATIONAL FREQUENCIES") != -1:
            if verbosity >= 2:
                print("\nReading frequencies from file.")
            while True:
                readBuffer = f.__next__()
                # Check if the whole line is integers only (Header line)
                if readBuffer.find("cm**-1") != -1:
                    if math.fabs(float(readBuffer.split()[1])) > 1.0:
                        freq.append(float(readBuffer.split()[1]))
                # Once we find this statement, we're done reading
                elif readBuffer.find("NORMAL MODES") != -1:
                    break
    if verbosity >= 3:
        print("\nFrequencies:")
        print(freq)
    f.close()
    molecule.frequencies = freq


def read_adf_frequencies(filename, molecule, verbosity=0):
    freq = []
    f = open(filename, 'r')
    for line in f:
        if line.find("List of All Frequencies:") != -1:
            if verbosity >= 2:
                print("\nReading frequencies from file.")
            # skip 8 lines ahead, that's where the data starts
            for i in range(0,8):
                readBuffer = f.__next__()
            while True:
                readBuffer = f.__next__()
                # Check if the whole line is integers only (Header line)
                if readBuffer.isspace():
                    break
                else:
                    freq.append(float(readBuffer.split()[0]))
    if verbosity >= 3:
        print("\nFrequencies:")
        print(freq)
    f.close()
    molecule.frequencies = freq


def read_gauss_multiplicity(filename, molecule, verbosity=0):
    QM_multiplicities = []
    f = open(filename, 'r')
    for line in f:
        if line.find(" Multiplicity = ") != -1:
            if verbosity >= 3:
                print("\nMultiplicity found")
                print(str(line))
            QM_multiplicities.append(line)
    # Take the last multiplicity from the file and assign that value as the multiplicity of the molecule
    i = QM_multiplicities[len(QM_multiplicities) - 1]
    readBuffer = i.split()
    QMmult = int(readBuffer[5])
    molecule.set_mult(QMmult)
    if verbosity >= 2:
        print("\nReading of multiplicities complete")
        if verbosity >= 3:
            print("Multiplicity = " + str(QMmult))
    f.close()


def read_orca_multiplicity(filename, molecule, verbosity=0):
    QM_multiplicities = []
    f = open(filename, 'r')
    for line in f:
        if line.find(" Multiplicity           Mult") != -1:
            if verbosity >= 3:
                print("\nMultiplicity found")
                print(str(line))
            QM_multiplicities.append(line)
    # Take the last multiplicity from the file and assign that value as the multiplicity of the molecule
    i = QM_multiplicities[len(QM_multiplicities) - 1]
    readBuffer = i.split()
    QMmult = int(readBuffer[3])
    molecule.set_mult(QMmult)
    if verbosity >= 2:
        print("\nReading of multiplicities complete")
        if verbosity >= 3:
            print("Multiplicity = " + str(QMmult))
    f.close()


def read_turbo_multiplicity(filename, molecule, verbosity=0):
    if verbosity >= 3:
        print(" Reading multiplicity from turbomole file not implemented: assuming singlet.")
        molecule.set_mult(1)
    if verbosity >= 3:
        print(" Multiplicity = " + str(1))


def read_adf_multiplicity(filename, molecule, verbosity=0):
    QM_multiplicities = []
    f = open(filename, 'r')
    for line in f:
        if line.find("  Spin polar:") != -1:
            if verbosity >= 3:
                print("\nSpin polarisation (multiplicity) found")
                print(str(line))
            QM_multiplicities.append(line)
    if verbosity >= 2:
        print("\nReading of multiplicities complete")
    # Take the last multiplicity from the file and assign that value as
    # the multiplicity of the molecule. If none was found: singlet
    if len(QM_multiplicities) > 0:
        i = QM_multiplicities[len(QM_multiplicities) - 1]
        readBuffer = i.split()
        # Need to add "1" here to get from spin polarisation to multiplicity
        QMmult = int(float(readBuffer[4])) + 1
    else:
        QMmult = 1
        if verbosity >= 3:
            print(" Couldn't read multiplicity from file, assuming singlet.")
    molecule.set_mult(QMmult)
    if verbosity >= 3:
        print(" Multiplicity = " + str(QMmult))
    f.close()


def read_gauss_rotational_symmetry_number(filename, molecule, verbosity=0):
    QM_sigmaRotnums = []
    f = open(filename, 'r')
    for line in f:
        if line.find(" Rotational symmetry number ") != -1:
            if verbosity >= 3:
                print("\n Rotational symmetry number")
                print(str(line))
            QM_sigmaRotnums.append(line)
    # Take the last multiplicity from the file and assign that value as the multiplicity of the molecule
    if len(QM_sigmaRotnums) != 0:
        i = QM_sigmaRotnums[len(QM_sigmaRotnums) - 1]
        readBuffer = i.split()
        QM_sigmaRotnum = int(float(readBuffer[3]))
    else:
        if verbosity >= 3:
            print(
                "\n No rotational symmetry number found (this is probably an atom), using default value of 1")
        QM_sigmaRotnum = 1
    molecule.sigmaRot = QM_sigmaRotnum
    if verbosity >= 2:
        print(" Rotational symmetry number = " + str(QM_sigmaRotnum))
    f.close()


def read_orca_rotational_symmetry_number(filename, molecule, verbosity=0):
    QM_sigmaRotnums = []
    f = open(filename, 'r')
    for line in f:
        if line.find("sn is the rotational symmetry number. We have assumed ") != -1:
            if verbosity >= 3:
                print("\n Rotational symmetry number found")
                print(str(line))
            QM_sigmaRotnums.append(line)
    # Take the last multiplicity from the file and assign that value as the multiplicity of the molecule
    if len(QM_sigmaRotnums) != 0:
        i = QM_sigmaRotnums[len(QM_sigmaRotnums) - 1]
        read_buffer = i.split()
        QM_sigmaRotnum = int(float(read_buffer[9]))
    else:
        if verbosity >= 3:
            print(
                "\n No rotational symmetry number found (this is probably an atom), using default value of 1")
        QM_sigmaRotnum = 1
    molecule.sigmaRot = QM_sigmaRotnum
    if verbosity >= 2:
        print(" Rotational symmetry number = " + str(QM_sigmaRotnum))
    f.close()


def read_turbo_rotational_symmetry_number(filename, molecule, verbosity=0):
    QM_sigmaRotnums = []
    #TODO: Find a way to determine from Turbomole output or write own module.


def read_adf_rotational_symmetry_number(filename, molecule, verbosity=0):
    QM_sigmaRotnums = []
    f = open(filename, 'r')
    for line in f:
        if line.find(" reported below were computed using sigma = ") != -1:
            if verbosity >= 3:
                print("\n Rotational symmetry number found")
                print(str(line))
            QM_sigmaRotnums.append(line)
    # Take the last multiplicity from the file and assign that value as the multiplicity of the molecule
    if len(QM_sigmaRotnums) != 0:
        i = QM_sigmaRotnums[len(QM_sigmaRotnums) - 1]
        read_buffer = i.split()
        QM_sigmaRotnum = int(float(read_buffer[7].replace(",","")))
    else:
        if verbosity >= 3:
            print(
                "\n No rotational symmetry number found (this is probably an atom), using default value of 1")
        QM_sigmaRotnum = 1
    molecule.sigmaRot = QM_sigmaRotnum
    if verbosity >= 2:
        print(" Rotational symmetry number = " + str(QM_sigmaRotnum))
    f.close()


def extract_molecular_data(filename, molecule, verbosity=0,
                           read_coordinates=True, read_bond_orders=True,
                           read_qm_energy=False,
                           build_angles=False, build_dihedrals=False,
                           read_force_constants=False,
                           read_multiplicity=False,
                           read_rotational_symmetry_number=False,
                           cpu_number=1):
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
        elif line.find("*   Amsterdam Density Functional  (ADF)") != -1:
            if verbosity >= 2:
                print("Reading ADF output file: ", filename)
            program = "adf"
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
        elif program == "adf":
            read_adf_coord(filename, molecule, verbosity=verbosity)
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
            build_bond_orders(molecule, [], verbosity=verbosity,
                                       cpu_number=cpu_number)

    # Check if we need to read the quantum mechanically calculated energy
    if read_qm_energy is True:
        if program == "gauss":
            read_gauss_qm_energy(filename, molecule, verbosity=verbosity)
        elif program == "orca":
            read_orca_qm_energy(filename, molecule, verbosity=verbosity)
        elif program == "turbomole":
            read_turbomole_qm_energy(filename, molecule, verbosity=verbosity)
        elif program == "adf":
            read_adf_qm_energy(filename, molecule, verbosity=verbosity)
        else:
            # There's no QM energy in xyz files...
            pass

    # Check if we need to build bond angles
    if build_angles is True:
        build_molecular_angles(molecule, verbosity=verbosity)

    # Check if we need to build dihedrals
    if build_dihedrals is True:
        build_molecular_dihedrals(molecule, verbosity=verbosity)

    # Check if we need to read (harmonic) force constants. If, after that, the
    # matrix of force constants is still empty, we'll try to read frequencies.
    if read_force_constants is True:
        if program == "gauss":
            read_gauss_force_constants(filename, molecule, verbosity=verbosity)
            if molecule.H_QM == []:
                read_gauss_frequencies(filename, molecule, verbosity=verbosity)
        elif program == "orca":
            read_orca_force_constants(filename, molecule, verbosity=verbosity)
            if molecule.H_QM == []:
                read_orca_frequencies(filename, molecule, verbosity=verbosity)
        elif program == "turbomole":
            read_turbo_force_constants(filename, molecule, verbosity=verbosity)
            if molecule.H_QM == []:
                read_turbo_frequencies(filename, molecule, verbosity=verbosity)
        elif program == "adf":
            read_adf_force_constants(filename, molecule, verbosity=verbosity)
            if molecule.H_QM == []:
                read_adf_frequencies(filename, molecule, verbosity=verbosity)
        else:
            # There are no force constants in xyz files...
            pass

    # Check if we need to read multiplicities
    if read_multiplicity is True:
        if program == "gauss":
            read_gauss_multiplicity(filename, molecule,
                                       verbosity=verbosity)
        elif program == "orca":
            read_orca_multiplicity(filename, molecule,
                                      verbosity=verbosity)
        elif program == "turbomole":
            read_turbo_multiplicity(filename, molecule,
                                       verbosity=verbosity)
        elif program == "adf":
            read_adf_multiplicity(filename, molecule,
                                     verbosity=verbosity)
        else:
            # There is no multiplicity in an xyz file...
            pass

    # Check if we need to read rotational symmetry numbers
    if read_rotational_symmetry_number is True:
        if program == "gauss":
            read_gauss_rotational_symmetry_number(filename, molecule,
                                    verbosity=verbosity)
        elif program == "orca":
            read_orca_rotational_symmetry_number(filename, molecule,
                                   verbosity=verbosity)
        elif program == "turbomole":
            read_turbo_rotational_symmetry_number(filename, molecule,
                                    verbosity=verbosity)
        elif program == "adf":
            read_adf_rotational_symmetry_number(filename, molecule,
                                  verbosity=verbosity)
        else:
            # There are no rotational symmetry numbers in an xyz file...
            pass


def main():
    prg_start_time = time.time()
    # Create an example molecule and add some atoms
    example = Molecule("Example Molecule")
    # extract_molecular_data("../examples/g09-h3b-nh3.log", example, verbosity=3)
    extract_molecular_data("../examples/pdb200l.xyz", example, verbosity=3,
                           read_coordinates=True, read_bond_orders=True,
                           build_angles=True, build_dihedrals=True, cpu_number=4)
    # print(example.print_mol(output="gauss"))
    msg_program_footer(prg_start_time)


if __name__ == '__main__':
    main()
