################################################################################
#                                                                              #
# This is the part of the program where the methods for reading QM files are   #
# defined                                                                      #
#                                                                              #
################################################################################

def extractMolecularData(filename, molecule, verbosity=0, distfactor=1.3, bondcutoff=0.45):
    # Try filename for readability first
    try:
        f = open(filename, 'r')
        f.close()
    except:
        ProgramError("Cannot open " + filename + " for reading.")

    if verbosity >= 1:
        print("\nSetting up WellFARe molecule: ", molecule.name)
    f = open(filename, 'r')
    program = "N/A"
    # Determine which QM program we're dealing with
    for line in f:
        if line.find("Entering Gaussian System, Link 0") != -1:
            if verbosity >= 2:
                print("Reading Gaussian output file: ", filename)
            program = "g09"
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
        else:
            if verbosity >= 2:
                print("Can't find any other structure, guessing this is an xyz file... ", filename)
            program = "xyz"
            break
    f.close()

    # GEOMETRY READING SECTION
    geom = []
    # Read through Gaussian file, read *last* "Input orientation"
    if program == "g09":
        f = open(filename, 'r')
        for line in f:
            if line.find("Input orientation:") != -1:
                if verbosity >= 2:
                    print("\nInput orientation found, reading coordinates")
                del geom[:]
                for i in range(0, 4):
                    readBuffer = f.__next__()
                while True:
                    readBuffer = f.__next__()
                    if readBuffer.find("-----------") == -1:
                        geom.append(readBuffer)
                        if verbosity >= 3:
                            readBuffer = readBuffer.split()
                            print(" Found atom: {:<3} {: .8f} {: .8f} {: .8f} in current Input orientation".format(
                                NumberToSymbol[int(readBuffer[1])], float(readBuffer[3]), float(readBuffer[4]),
                                float(readBuffer[5])))
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
                        print("\nStandard orientation found, reading coordinates")
                    del geom[:]
                    for i in range(0, 4):
                        readBuffer = f.__next__()
                    while True:
                        readBuffer = f.__next__()
                        if readBuffer.find("-----------") == -1:
                            geom.append(readBuffer)
                            if verbosity >= 3:
                                readBuffer = readBuffer.split()
                                print(
                                    " Found atom: {:<3} {: .8f} {: .8f} {: .8f} in current Standard orientation".format(
                                        NumberToSymbol[int(readBuffer[1])], float(readBuffer[3]), float(readBuffer[4]),
                                        float(readBuffer[5])))
                        else:
                            break
            f.close()
        if verbosity >= 2:
            print("\nReading of geometry finished.\nAdding atoms to WellFARe molecule: ", molecule.name)
        for i in geom:
            readBuffer = i.split()
            molecule.add_atom(Atom(NumberToSymbol[int(readBuffer[1])], float(readBuffer[3]), float(readBuffer[4]),
                                   float(readBuffer[5]), 0.1))  # 0.1 a placeholder for QM calculated charge on the atom
            if verbosity >= 3:
                print(" {:<3} {: .8f} {: .8f} {: .8f}".format(NumberToSymbol[int(readBuffer[1])], float(readBuffer[3]),
                                                              float(readBuffer[4]), float(readBuffer[5])))
    # Read through ORCA file, read *last* set of cartesian coordinates
    elif program == "orca":
        f = open(filename, 'r')
        for line in f:
            if line.find("CARTESIAN COORDINATES (ANGSTROEM)") != -1:
                if verbosity >= 2:
                    print("\nCartesian Coordinates found")
                del geom[:]
                readBuffer = f.__next__()
                while True:
                    readBuffer = f.__next__()
                    if readBuffer and readBuffer.strip():
                        geom.append(readBuffer)
                        if verbosity >= 3:
                            readBuffer = readBuffer.split()
                            print(" Found atom: {:<3} {: .8f} {: .8f} {: .8f} in current Cartesian Coordinates".format(
                                readBuffer[0], float(readBuffer[1]), float(readBuffer[2]), float(readBuffer[3])))
                    else:
                        break
        if verbosity >= 2:
            print("\nReading of geometry finished.\nAdding atoms to WellFARe molecule: ", molecule.name)
        for i in geom:
            readBuffer = i.split()
            molecule.add_atom(Atom(readBuffer[0], float(readBuffer[1]), float(readBuffer[2]), float(readBuffer[3]),
                                   0.1))  # 0.1 a placeholder for QM computed carge on the atom
            if verbosity >= 3:
                print(" {:<3} {: .8f} {: .8f} {: .8f}".format(readBuffer[0], float(readBuffer[1]), float(readBuffer[2]),
                                                              float(readBuffer[3])))
        f.close()
    # Read through Turbomole aoforce file, read cartesian coordinates
    elif program == "turbomole":
        f = open(filename, 'r')
        for line in f:
            if line.find("atomic coordinates            atom    charge  isotop") != -1:
                if verbosity >= 2:
                    print("\nCartesian Coordinates found")
                del geom[:]
                while True:
                    readBuffer = f.__next__()
                    if readBuffer and readBuffer.strip():
                        geom.append(readBuffer)
                        if verbosity >= 3:
                            readBuffer = readBuffer.split()
                            print(" Found atom: {:<3} {: .8f} {: .8f} {: .8f} in current Cartesian Coordinates".format(
                                NumberToSymbol[int(float(readBuffer[4]))], Bohr2Ang(float(readBuffer[0])),
                                Bohr2Ang(float(readBuffer[1])), Bohr2Ang(float(readBuffer[2]))))
                    else:
                        break
        if verbosity >= 2:
            print("\nReading of geometry finished.\nAdding atoms to WellFARe molecule: ", molecule.name)
        for i in geom:
            readBuffer = i.split()
            molecule.add_atom(Atom(NumberToSymbol[int(float(readBuffer[4]))], Bohr2Ang(float(readBuffer[0])),
                                   Bohr2Ang(float(readBuffer[1])), Bohr2Ang(float(readBuffer[2])),
                                   0.1))  # 0.1 a placeholder for QM computed carge on the atom
            if verbosity >= 3:
                print(" {:<3} {: .8f} {: .8f} {: .8f}".format(NumberToSymbol[int(float(readBuffer[4]))],
                                                              Bohr2Ang(float(readBuffer[0])),
                                                              Bohr2Ang(float(readBuffer[1])),
                                                              Bohr2Ang(float(readBuffer[2]))))
        f.close()
    # Read through xyz, read cartesian coordinates
    elif program == "xyz":
        f = open(filename, 'r')
        # Now examine every line until a line that doesn't conform to the template or EOF is found
        for line in f:
            # Check if line conforms to template
            readBuffer = line.split()
            if len(readBuffer) != 4:
                # If there are not *exactly* four entries on this line, we delete the current geometry
                # and start fresh.
                del geom[:]
            else:
                if isAtmSymb(readBuffer[0]) and isFloat(readBuffer[1]) and isFloat(readBuffer[2]) and isFloat(
                        readBuffer[3]):
                    geom.append(readBuffer)
                    if verbosity >= 3:
                        if len(geom) == 1:
                            print("New structure found, starting to read structure")
                            print(" Found atom: {:<3} {: .8f} {: .8f} {: .8f} in current Cartesian Coordinates".format(
                                readBuffer[0], float(readBuffer[1]), float(readBuffer[2]), float(readBuffer[3])))
                        else:
                            print(" Found atom: {:<3} {: .8f} {: .8f} {: .8f} in current Cartesian Coordinates".format(
                                readBuffer[0], float(readBuffer[1]), float(readBuffer[2]), float(readBuffer[3])))
                else:
                    # If the entries aren't an atomic symbol and 3 coords, we delete the current geometry
                    # ans start fresh.
                    del geom[:]

        if verbosity >= 2:
            print("\nReading of geometry finished.\nAdding atoms to WellFARe molecule: ", molecule.name)
        for i in geom:
            molecule.add_atom(Atom(i[0], float(i[1]), float(i[2]),
                                   float(i[3]), 0.1))  # 0.1 a placeholder for QM calculated charge on the atom
            if verbosity >= 3:
                print(" {:<3} {: .8f} {: .8f} {: .8f}".format(i[0], float(i[1]), float(i[2]),
                                                              float(i[3])))
        f.close()

    # BOND ORDER READING SECTION
    bo = []
    if program == "g09":
        f = open(filename, 'r')
        for line in f:
            if line.find("Atomic Valencies and Mayer Atomic Bond Orders:") != -1:
                bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
                if verbosity >= 2:
                    print("\nAtomic Valencies and Mayer Atomic Bond Orders found, reading data")
                bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
                while True:
                    readBuffer = f.__next__()
                    # Check if the whole line is integers only (Header line)
                    if isInt("".join(readBuffer.split())) == True:
                        # And use this information to label the columns
                        columns = readBuffer.split()
                    # If we get to the Löwdin charges, we're done reading
                    elif readBuffer.find("Lowdin Atomic Charges") != -1:
                        break
                    else:
                        row = readBuffer.split()
                        j = 1
                        for i in columns:
                            j = j + 1
                            bo[int(row[0]) - 1][int(i) - 1] = float(row[j])
        f.close()
        if verbosity >= 3:
            print("\nBond Orders:")
            np.set_printoptions(suppress=True)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(bo)
    elif program == "orca":
        f = open(filename, 'r')
        for line in f:
            if line.find("Mayer bond orders larger than 0.1") != -1:
                bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
                if verbosity >= 2:
                    print("\nMayer bond orders larger than 0.1 found, reading data")
                bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
                while True:
                    readBuffer = f.__next__()
                    # Check if the whole line isn't empty (in that case we're done)
                    if readBuffer and readBuffer.strip():
                        # Break the line into pieces
                        readBuffer = readBuffer[1:].strip()
                        readBuffer = readBuffer.split("B")
                        for i in readBuffer:
                            bondpair1 = int(i[1:4].strip())
                            bondpair2 = int(i[8:11].strip())
                            order = i[-9:].strip()
                            bo[bondpair1][bondpair2] = order
                            bo[bondpair2][bondpair1] = order
                    else:
                        break
        f.close()
        if verbosity >= 3:
            print("\nBond Orders:")
            np.set_printoptions(suppress=True)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(bo)
    elif program == "turbomole":  ######## I don't know if/how Turbomole prints these...
        # f = open(filename, 'r')
        # for line in f:
        #     if line.find("Mayer bond orders larger than 0.1") != -1:
        #         bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
        #         if verbosity >= 2:
        #             print("\nMayer bond orders larger than 0.1 found, reading data")
        #         bo = np.zeros((molecule.num_atoms(), molecule.num_atoms()))
        #         while True:
        #             readBuffer = f.__next__()
        #             # Check if the whole line isn't empty (in that case we're done)
        #             if readBuffer and readBuffer.strip():
        #                 # Break the line into pieces
        #                 readBuffer = readBuffer[1:].strip()
        #                 readBuffer = readBuffer.split("B")
        #                 for i in readBuffer:
        #                     bondpair1 = int(i[1:4].strip())
        #                     bondpair2 = int(i[8:11].strip())
        #                     order = i[-9:].strip()
        #                     bo[bondpair1][bondpair2] = order
        #                     bo[bondpair2][bondpair1] = order
        #             else:
        #                 break
        # f.close()
        if verbosity >= 3:
            print("\nBond Orders:")
            np.set_printoptions(suppress=True)
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print(bo)

    # Test if we actually have Mayer Bond orders
    if np.count_nonzero(bo) != 0 and molecule.num_atoms() != 1:
        if verbosity >= 2:
            print("\nLooking for bonds in WellFARe molecule: ", molecule.name)
            print("(using bond orders with a cutoff of {: .2f}):".format(bondcutoff))
        for i in range(0, molecule.num_atoms()):
            for j in range(i + 1, molecule.num_atoms()):
                if bo[i][j] >= bondcutoff:
                    molecule.add_bond(i, j)
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}) and {:<3} ({:3d}) (Bond order: {: .3f})".format(molecule.atoms[i].symbol, i,
                                                                                            molecule.atoms[j].symbol, j,
                                                                                            bo[i][j]))
    # Else use 130% of the sum of covalent radii as criterion for a bond (user defined: distfactor)
    elif molecule.num_atoms() != 1:
        if verbosity >= 2:
            print("\nLooking for bonds in WellFARe molecule:", molecule.name)
            print("(using covalent radii scaled by {: .2f}):".format(distfactor))
        for i in range(0, molecule.num_atoms()):
            for j in range(i + 1, molecule.num_atoms()):
                if molecule.atm_atm_dist(i, j) <= (
                        SymbolToRadius[molecule.atoms[i].symbol] + SymbolToRadius[
                    molecule.atoms[j].symbol]) * distfactor:
                    molecule.add_bond(i, j)
                    if verbosity >= 3:
                        print(
                            " {:<3} ({:3d}) and {:<3} ({:3d}) (Distance: {:.3f} Å)".format(molecule.atoms[i].symbol, i,
                                                                                           molecule.atoms[j].symbol, j,
                                                                                           molecule.atm_atm_dist(i, j)))
    else:
        if verbosity >= 2:
            print("\nNot looking for bonds in WellFARe molecule:", molecule.name)
            print("(because we only have one atom)")

    # Now that we know where the bonds are, find angles
    if len(molecule.bonds) > 1:
        if verbosity >= 2:
            print("\nLooking for angles in WellFARe molecule: ", molecule.name)
        for i in range(0, len(molecule.bonds)):
            for j in range(i + 1, len(molecule.bonds)):
                if molecule.bonds[i][0] == molecule.bonds[j][0]:
                    molecule.add_angle(molecule.bonds[i][1], molecule.bonds[i][0], molecule.bonds[j][1])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({:6.2f}°)".format(
                            molecule.atoms[molecule.bonds[i][1]].symbol, molecule.bonds[i][1],
                            molecule.atoms[molecule.bonds[i][0]].symbol, molecule.bonds[i][0],
                            molecule.atoms[molecule.bonds[j][1]].symbol, molecule.bonds[j][1],
                            math.degrees(molecule.bond_angle(len(molecule.angles) - 1))))
                if molecule.bonds[i][0] == molecule.bonds[j][1]:
                    molecule.add_angle(molecule.bonds[i][1], molecule.bonds[i][0], molecule.bonds[j][0])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({:6.2f}°)".format(
                            molecule.atoms[molecule.bonds[i][1]].symbol, molecule.bonds[i][1],
                            molecule.atoms[molecule.bonds[i][0]].symbol, molecule.bonds[i][0],
                            molecule.atoms[molecule.bonds[j][0]].symbol, molecule.bonds[j][0],
                            math.degrees(molecule.bond_angle(len(molecule.angles) - 1))))
                if molecule.bonds[i][1] == molecule.bonds[j][0]:
                    molecule.add_angle(molecule.bonds[i][0], molecule.bonds[i][1], molecule.bonds[j][1])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({:6.2f}°)".format(
                            molecule.atoms[molecule.bonds[i][0]].symbol, molecule.bonds[i][0],
                            molecule.atoms[molecule.bonds[i][1]].symbol, molecule.bonds[i][1],
                            molecule.atoms[molecule.bonds[j][1]].symbol, molecule.bonds[j][1],
                            math.degrees(molecule.bond_angle(len(molecule.angles) - 1))))
                if molecule.bonds[i][1] == molecule.bonds[j][1]:
                    molecule.add_angle(molecule.bonds[i][0], molecule.bonds[i][1], molecule.bonds[j][0])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({:6.2f}°)".format(
                            molecule.atoms[molecule.bonds[i][0]].symbol, molecule.bonds[i][0],
                            molecule.atoms[molecule.bonds[i][1]].symbol, molecule.bonds[i][1],
                            molecule.atoms[molecule.bonds[j][0]].symbol, molecule.bonds[j][0],
                            math.degrees(molecule.bond_angle(len(molecule.angles) - 1))))
    else:
        if verbosity >= 2:
            print("\nNot looking for angles in WellFARe molecule: ", molecule.name)
            print("(because there are fewer than 2 bonds identified in the molecule)")

    # Same for dihedrals: Use angles to determine where they are
    if len(molecule.angles) > 1:
        if verbosity >= 2:
            print("\nLooking for dihedrals in WellFARe molecule: ", molecule.name)
        for i in range(0, len(molecule.angles)):
            for j in range(i + 1, len(molecule.angles)):
                if molecule.angles[i][1] == molecule.angles[j][0] and molecule.angles[i][2] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][0], molecule.angles[i][1], molecule.angles[i][2],
                                          molecule.angles[j][2])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({: 7.2f}°)".format(
                            molecule.atoms[molecule.angles[i][0]].symbol, molecule.angles[i][0],
                            molecule.atoms[molecule.angles[i][1]].symbol, molecule.angles[i][1],
                            molecule.atoms[molecule.angles[i][2]].symbol, molecule.angles[i][2],
                            molecule.atoms[molecule.angles[j][2]].symbol, molecule.angles[j][2],
                            math.degrees(molecule.dihedral_angle(len(molecule.dihedrals) - 1))))
                if molecule.angles[i][1] == molecule.angles[j][2] and molecule.angles[i][2] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][0], molecule.angles[i][1], molecule.angles[i][2],
                                          molecule.angles[j][0])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({: 7.2f}°)".format(
                            molecule.atoms[molecule.angles[i][0]].symbol, molecule.angles[i][0],
                            molecule.atoms[molecule.angles[i][1]].symbol, molecule.angles[i][1],
                            molecule.atoms[molecule.angles[i][2]].symbol, molecule.angles[i][2],
                            molecule.atoms[molecule.angles[j][0]].symbol, molecule.angles[j][0],
                            math.degrees(molecule.dihedral_angle(len(molecule.dihedrals) - 1))))
                if molecule.angles[i][1] == molecule.angles[j][0] and molecule.angles[i][0] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][2], molecule.angles[j][0], molecule.angles[j][1],
                                          molecule.angles[j][2])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({: 7.2f}°)".format(
                            molecule.atoms[molecule.angles[i][2]].symbol, molecule.angles[i][2],
                            molecule.atoms[molecule.angles[j][0]].symbol, molecule.angles[j][0],
                            molecule.atoms[molecule.angles[j][1]].symbol, molecule.angles[j][1],
                            molecule.atoms[molecule.angles[j][2]].symbol, molecule.angles[j][2],
                            math.degrees(molecule.dihedral_angle(len(molecule.dihedrals) - 1))))
                if molecule.angles[i][1] == molecule.angles[j][2] and molecule.angles[i][0] == molecule.angles[j][1]:
                    molecule.add_dihedral(molecule.angles[i][2], molecule.angles[j][2], molecule.angles[j][1],
                                          molecule.angles[j][0])
                    if verbosity >= 2:
                        print(" {:<3} ({:3d}), {:<3} ({:3d}), {:<3} ({:3d}) and {:<3} ({:3d}) ({: 7.2f}°)".format(
                            molecule.atoms[molecule.angles[i][2]].symbol, molecule.angles[i][2],
                            molecule.atoms[molecule.angles[j][2]].symbol, molecule.angles[j][2],
                            molecule.atoms[molecule.angles[j][1]].symbol, molecule.angles[j][1],
                            molecule.atoms[molecule.angles[j][0]].symbol, molecule.angles[j][0],
                            math.degrees(molecule.dihedral_angle(len(molecule.dihedrals) - 1))))
    else:
        if verbosity >= 2:
            print("\nNot looking for dihedrals in WellFARe molecule: ", molecule.name)
            print("(because there are fewer than two angles identified in the molecule)")

