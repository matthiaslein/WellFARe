
################################################################################
#                                                                              #
# This is the part of the program where the command line arguments are defined #
#                                                                              #
################################################################################

parser = argparse.ArgumentParser(
    description="WellFAReMICE: Method for the Incremental Construction and Exploration of the Potential Energy Surface",
    epilog="recognised filetypes: gaussian, orca, turbomole, xyz")
# parser.add_argument("-P", "--numproc", type=int, help="number of processes for parallel execution",
#                     default="2")
parser.add_argument("-v", "--verbosity", help="increase output verbosity", type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument("-o", "--output",
                    help="type of output file to be written: xyz file or just cartesian coordinates without header",
                    choices=["xyz", "cart"],
                    default="cart")
parser.add_argument("-g", "--group", help="replacement group",
                    choices=["methyl", "ethenyl", "oh", "oh2", "nh2", "nh3", "cho", "f", "cl", "br", "i"],
                    default="methyl")
parser.add_argument("inputfile", metavar='file',
                    help="input file(s) with molecular structure")
parser.add_argument("-r", "--replace", type=int, nargs='+', help="list of individual hydrogen atoms to replace")
parser.add_argument("-a", "--alloncarbon", type=int, nargs='+',
                    help="list of carbon atoms whose hydrogen atoms should be replaced")
parser.add_argument("-t", "--terminal", type=int, help="number of terminal hydrogen atoms to replace")

args = parser.parse_args()

###############################################################################
#                                                                             #
# The main part of the program starts here                                    #
#                                                                             #
###############################################################################

# Print GPL v3 statement and program header
if args.verbosity >= 1:
    ProgramHeader()

molecule = Molecule("Input Structure", 0)
extractMolecularData(args.inputfile, molecule, verbosity=args.verbosity)

ListOfHydrogens = []

if args.replace is not None:
    if args.verbosity >= 3:
        print("\nAdding hydrogen atoms from the --replace key")
    for i in args.replace:
        if molecule.atoms[i - 1].symbol == "H":
            ListOfHydrogens.append(i)
            if args.verbosity >= 3:
                print("Hydrogen atom ", i, " added to the list of atoms for replacement.")
        else:
            ProgramWarning("Atom " + str(i) + " in the input is not hydrogen!")
if args.alloncarbon is not None:
    if args.verbosity >= 3:
        print("\nAdding hydrogen atoms from the --alloncarbon key")
    for i in args.alloncarbon:
        if molecule.atoms[i - 1].symbol == "C":
            if args.verbosity >= 3:
                print("Adding the hydrogen atoms bonded to carbon atom ", i)
            for j in molecule.bonds:
                at1 = j[0]  # First atom in the bond
                at2 = j[1]  # Second atom in the bond
                if at1 + 1 == i and molecule.atoms[at2].symbol == "H":
                    ListOfHydrogens.append(at2 + 1)
                    if args.verbosity >= 3:
                        print("Hydrogen atom ", at2 + 1, " added to the list of atoms for replacement.")
                elif at2 + 1 == i and molecule.atoms[at1].symbol == "H":
                    ListOfHydrogens.append(at1 + 1)
                    if args.verbosity >= 3:
                        print("Hydrogen atom ", at1 + 1, " added to the list of atoms for replacement.")
        else:
            ProgramWarning("Atom " + str(i) + " in the input is not carbon!")
if args.terminal is not None:
    if args.verbosity >= 3:
        print("\nAdding hydrogen atoms from the --terminal key")
    # Count backwards and add the given number of hydrogen atoms to the list
    counter = 0
    for i in range(molecule.numatoms(), 0, -1):
        if counter < args.terminal and molecule.atoms[i - 1].symbol == "H":
            if args.verbosity >= 3:
                print("Hydrogen atom ", i, " added to the list of atoms for replacement.")
            ListOfHydrogens.append(i)
            counter += 1
elif args.replace is None and args.alloncarbon is None:
    # If no instructions were given at all: default to replacing the terminal 3 hydrogens
    if args.verbosity >= 2:
        print("Defaulting to replacement of the last 3 hydrogen atoms")
    counter = 0
    for i in range(molecule.numatoms(), 0, -1):
        if counter < 3 and molecule.atoms[i - 1].symbol == "H":
            if args.verbosity >= 3:
                print("Hydrogen atom ", i, " added to the list of atoms for replacement.")
            ListOfHydrogens.append(i)
            counter += 1

if ListOfHydrogens == []:
    ProgramError("No atoms selected for replacement")
else:
    # Turn list into set to remove duplicates
    ListOfHydrogens = set(ListOfHydrogens)
    if args.verbosity >= 2:
        print("\nThere are ", len(ListOfHydrogens), "hydrogen atoms that have been selected for replacement.")
        if args.verbosity >= 3:
            print("Hydrogen atoms: ", ListOfHydrogens)

for j, i in enumerate(ListOfHydrogens):
    filesuffix = str(j + 1).zfill(int(math.ceil(math.log(len(ListOfHydrogens), 10))))
    filename, file_extension = os.path.splitext(args.inputfile)
    # Look up which replacement was selected
    if args.group == "methyl":
        newmol = copy.deepcopy(molecule)
        newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
            i) + " replaced"
        newmol.replaceHwithTetrahedral((i - 1))
        msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "ethenyl":
        newmol = copy.deepcopy(molecule)
        newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
            i) + " replaced"
        newmol.replaceHwithEthenyl(i - 1)
        msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "cho":
        # If we replace H by CHO, we create the three possible orientations at the same time
        for l, k in enumerate(["a", "b", "c"]):
            newmol = copy.deepcopy(molecule)
            newmol.name = "Molecule " + filesuffix + " out of " + str(
                len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            newmol.replaceHwithCHO((i - 1), orientation=l)
            msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix + k, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "nh3":
        newmol = copy.deepcopy(molecule)
        newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
            i) + " replaced"
        newmol.replaceHwithTetrahedral((i - 1), replacement="N")
        msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "oh":
        # If we replace H by OH, we create the three possible orientations at the same time
        for l, k in enumerate(["a", "b", "c"]):
            newmol = copy.deepcopy(molecule)
            newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            newmol.replaceHwithTetrahedral((i - 1), replacement="O", addH=1, orientation=l)
            msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix + k, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "nh2":
        # If we replace H by NH₂, we create the three possible orientations at the same time
        for l, k in enumerate(["a", "b", "c"]):
            newmol = copy.deepcopy(molecule)
            newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            newmol.replaceHwithTetrahedral((i - 1), replacement="N", addH=2, orientation=l)
            msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix + k, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "oh2":
        # If we replace H by OH₂, we create the three possible orientations at the same time
        for l, k in enumerate(["a", "b", "c"]):
            newmol = copy.deepcopy(molecule)
            newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
                i) + " replaced"
            newmol.replaceHwithTetrahedral((i - 1), replacement="O", addH=2, orientation=l)
            msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix + k, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "f":
        newmol = copy.deepcopy(molecule)
        newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
            i) + " replaced"
        newmol.replaceHwithTetrahedral((i - 1), replacement="F", addH=0)
        msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "cl":
        newmol = copy.deepcopy(molecule)
        newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
            i) + " replaced"
        newmol.replaceHwithTetrahedral((i - 1), replacement="Cl", addH=0)
        msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "br":
        newmol = copy.deepcopy(molecule)
        newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
            i) + " replaced"
        newmol.replaceHwithTetrahedral((i - 1), replacement="Br", addH=0)
        msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)
    elif args.group == "i":
        newmol = copy.deepcopy(molecule)
        newmol.name = "Molecule " + filesuffix + " out of " + str(len(ListOfHydrogens)) + ", hydrogen atom no. " + str(
        i) + " replaced"
        newmol.replaceHwithTetrahedral((i - 1), replacement="I", addH=0)
        msg = newmol.printMol(output=args.output, file=filename + "-" + filesuffix, comment=newmol.name)
        if args.verbosity >= 3:
            print("\nNew Structure:")
            print(msg)

# Print program footer
if args.verbosity >= 1:
    ProgramFooter()
