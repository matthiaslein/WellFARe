##############################################################################
#                                                                            #
# This is the part of the program where modules are imported                 #
#                                                                            #
##############################################################################

import argparse
import math
from multiprocessing import Pool, cpu_count


from messages import *
from molecule import Molecule, build_molecular_dihedrals, \
    build_molecular_angles, build_bond_orders
from qmparser import extract_molecular_data


def check_dist_diff(first_molecule, second_molecule, a, b, toler):
    # print("Checking atoms {} and {}.".format(i,j))
    dist1 = first_molecule.atm_atm_dist(a, b)
    if dist1 <= 10.0:
        dist2 = second_molecule.atm_atm_dist(a, b)
        distdiff = math.fabs(dist1 - dist2)
    else:
        distdiff = 0.0
    tolerance = (10 ** (-1.0 * toler)) * dist1
    if distdiff != 0.0 and distdiff >= tolerance:
        return False
    return True


def check_dist_list(molecule_a, molecule_b, list_of_pairs, toler):
    # print(list)
    for pair in list_of_pairs:
        if not check_dist_diff(molecule_a, molecule_b, pair[0], pair[1],
                               toler):
            return False
    return True


############################################################################
#                                                                          #
# This is the part of the program where the cmd line arguments are defined #
#                                                                          #
############################################################################

parser = argparse.ArgumentParser(
    description="comparestructures: Determines if two input structures"
                " represent the same molecule.",
    epilog="recognised filetypes: gaussian, orca, turbomole, xyz")
parser.add_argument("-v", "--verbosity", help="increase output verbosity",
                    type=int, choices=[0, 1, 2, 3, 4], default=0)
parser.add_argument("-t", "--tolerance",
                    help="sets a tolerance of 10⁻ⁿ Å for variation of atomic"
                         " distances.",
                    type=int, choices=[1, 2, 3], default=2)
parser.add_argument("-m", "--method",
                    help="method for doing the analysis, either by comparison"
                         " of the local distance matrix, or by comparison of"
                         " the internal coordinates. Note that the default,"
                         " distmat, is much faster, but also incapable of"
                         " distinguishing enantiomers.",
                    choices=["distmat", "internal"],
                    default="distmat")
parser.add_argument("-p", "--numproc", type=int,
                    help="number of processes for parallel execution",
                    default="0")
parser.add_argument("file1", help="input file with first molecular structure")
parser.add_argument("file2", help="input file with second molecular structure")
args = parser.parse_args()

# Default number of cores for processing is requested with "-p 0" and uses
# all available cores.
if args.numproc == 0:
    args.numproc = cpu_count()


###############################################################################
#                                                                             #
# The main part of the program starts here                                    #
#                                                                             #
###############################################################################

def main():
    # Print GPL v3 statement and program header
    prg_start_time = time.time()
    if args.verbosity >= 1:
        msg_program_header("CompareStruct", 1.0)

    molecule1 = Molecule("First input Structure", 0)
    extract_molecular_data(args.file1, molecule1, read_bond_orders=False,
                           verbosity=0)

    molecule2 = Molecule("Second input Structure", 0)
    extract_molecular_data(args.file2, molecule2, read_bond_orders=False,
                           verbosity=0)

    #################################
    # Tests for identity start here #
    #################################

    # We first test if the number of atoms is identical
    if molecule1.num_atoms() != molecule2.num_atoms():
        if args.verbosity >= 2:
            print("\nNumber of atoms differs: {} vs {}.".format(
                molecule1.num_atoms(), molecule2.num_atoms()))
        if args.verbosity >= 1:
            msg_program_footer(prg_start_time)
        sys.exit(-1)
    else:
        if args.verbosity >= 3:
            print(
                msg_timestamp("\nIdentical number of atoms. ", prg_start_time))

    # Next, check if each individual atom in the list is of the same type
    for i in range(0, molecule1.num_atoms()):
        if molecule1.atoms[i].symbol() != molecule2.atoms[i].symbol():
            if args.verbosity >= 2:
                print(
                    "\nAtom types differ: {} vs {} for atom no {}.".format(
                        molecule1.atoms[i].symbol(),
                        molecule2.atoms[i].symbol(),
                        i + 1))
            if args.verbosity >= 1:
                msg_program_footer(prg_start_time)
            sys.exit(-1)
    if args.verbosity >= 3:
        print(msg_timestamp("\nIdentical types of atoms. ", prg_start_time))

    # Now check bond lengths, angles and dihedrals (here), or the distance
    # matrix (below).
    if args.method == "internal":
        # Determine bond distances and check if they're identical
        bonds_mol1 = build_bond_orders(molecule1, verbosity=args.verbosity - 1,
                                       canonical_order=True,
                                       cpu_number=args.numproc)
        if args.verbosity >= 3:
            print(
                msg_timestamp("\n{} bonds identified for molecule 1 ".format(
                    len(molecule1.bonds)), prg_start_time))
        bonds_mol2 = build_bond_orders(molecule2, verbosity=args.verbosity - 1,
                                       canonical_order=False,
                                       cpu_number=args.numproc)
        if args.verbosity >= 3:
            print(
                msg_timestamp("\n{} bonds identified for molecule 2 ".format(
                    len(molecule2.bonds)), prg_start_time))
        if len(bonds_mol1) != len(bonds_mol2):
            if args.verbosity >= 2:
                print("Mismatching number of bonds")
            if args.verbosity >= 1:
                msg_program_footer(prg_start_time)
            sys.exit(-1)
        # Tolerance will be ±0.1 Å, ±0.01 Å or ±0.001 Å
        tol = (10 ** (-1.0 * args.tolerance))
        if args.verbosity >= 3:
            print(
                "\nTolerance for bond length comparison is ±{} Å".format(tol))
        for idx, bond in enumerate(molecule1.bonds):
            if bond == molecule2.bonds[idx]:
                dist_diff = math.fabs(bonds_mol1[idx] - bonds_mol2[idx])
                if dist_diff >= tol:
                    if args.verbosity >= 2:
                        print(
                            "Mismatching distance of bond no {}:"
                            " {: 0.3f} Å vs. {: 0.3f} Å".format(
                                idx + 1, bonds_mol1[idx], bonds_mol2[idx]))
                    if args.verbosity >= tol:
                        msg_program_footer(prg_start_time)
                    sys.exit(-1)

            else:
                if args.verbosity >= 2:
                    print("Mismatching bond no {}".format(idx + 1))
                if args.verbosity >= 1:
                    msg_program_footer(prg_start_time)
                sys.exit(-1)
        if args.verbosity >= 2:
            print(msg_timestamp("Bonds and bond distances match. ",
                                prg_start_time))

        # Determine bond angles and check if they're identical
        angles_mol1 = build_molecular_angles(molecule1,
                                             verbosity=args.verbosity - 1)
        angles_mol2 = build_molecular_angles(molecule2,
                                             verbosity=args.verbosity - 1)
        # Tolerance will be ±5°, ±0.5° or ±0.05°
        tol = (10 ** (-1.0 * args.tolerance)) * 50.0
        if args.verbosity >= 3:
            print("\nTolerance for bond angle comparison is ±{}°".format(tol))
        for idx, bond in enumerate(molecule1.angles):
            if bond == molecule2.angles[idx]:
                dist_diff = math.fabs(angles_mol1[idx] - angles_mol2[idx])
                if dist_diff >= tol:
                    if args.verbosity >= 2:
                        print(
                            "Mismatching value of bond angle no {}:"
                            " {: 0.3f} vs. {: 0.3f}".format(
                                idx + 1, angles_mol1[idx], angles_mol2[idx]))
                    if args.verbosity >= 1:
                        msg_program_footer(prg_start_time)
                    sys.exit(-1)
            else:
                if args.verbosity >= 2:
                    print("Mismatching bond angle no {}.".format(idx + 1))
                if args.verbosity >= 1:
                    msg_program_footer(prg_start_time)
                sys.exit(-1)
        if args.verbosity >= 2:
            print(msg_timestamp("Bond angles and their values match. ",
                                prg_start_time))

        # Determine dihedral angles and check if they're identical
        dihedrals_m1 = build_molecular_dihedrals(molecule1,
                                                 verbosity=args.verbosity - 1)
        dihedrals_m2 = build_molecular_dihedrals(molecule2,
                                                 verbosity=args.verbosity - 1)
        # Tolerance will be ±15°, ±1.5° or ±0.15°
        tol = (10 ** (-1.0 * args.tolerance)) * 150.0
        if args.verbosity >= 3:
            print("\nTolerance for dihedral angle comparison is ±{}°".format(
                tol))
        for idx, bond in enumerate(molecule1.dihedrals):
            if bond == molecule2.dihedrals[idx]:
                # The difference calculation for dihedrals is a bit more
                # complicated because of the -180 to 180° range!
                dist_diff = 180 - math.fabs(
                    math.fabs(dihedrals_m1[idx] - dihedrals_m2[idx]) - 180)
                if dist_diff >= tol:
                    if args.verbosity >= 2:
                        print(
                            "Mismatching dihedral angle value no {}: {: 0.3f}°"
                            " vs. {: 0.3f}°".format(
                                idx + 1, dihedrals_m1[idx],
                                dihedrals_m2[idx]))
                    if args.verbosity >= 1:
                        msg_program_footer(prg_start_time)
                    sys.exit(-1)
            else:
                if args.verbosity >= 2:
                    print("Mismatching dihedral angle no {}.".format(idx + 1))
                if args.verbosity >= 1:
                    msg_program_footer(prg_start_time)
                sys.exit(-1)

        if args.verbosity >= 2:
            print(msg_timestamp("Dihedral angles and their values match. ",
                                prg_start_time))
    else:
        if args.numproc == 1:
            if args.verbosity >= 3:
                print(
                    "\nStarting serial execution on a single processor core.")
            for i in range(0, molecule1.num_atoms()):
                for j in range(i + 1, molecule1.num_atoms()):
                    if check_dist_diff(molecule1, molecule2, i, j,
                                       args.tolerance) is False:
                        if args.verbosity >= 2:
                            print("Mismatch in local distance matrix.")
                        if args.verbosity >= 1:
                            msg_program_footer(prg_start_time)
                        sys.exit(-1)
        else:
            if args.verbosity >= 3:
                print(
                    "\nStarting parallel execution on {} processor"
                    " cores.".format(args.numproc))
            pairs = []
            for i in range(0, molecule1.num_atoms()):
                for j in range(i + 1, molecule1.num_atoms()):
                    pairs.append([i, j])
            chunks = [pairs[i:i + (len(pairs) // args.numproc)] for i in
                      range(0, len(pairs), (len(pairs) // args.numproc))]
            with Pool(processes=args.numproc) as p:
                res = [p.apply_async(check_dist_list, args=(
                     molecule1, molecule2, i, args.tolerance)) for i in chunks]
                results = [p.get() for p in res]
            if False in results:
                if args.verbosity >= 2:
                    print("Mismatch in local distance matrix.")
                if args.verbosity >= 1:
                    msg_program_footer(prg_start_time)
                sys.exit(-1)

        if args.verbosity >= 3:
            print("\nIdentical distance matrices.")

        if args.verbosity >= 2:
            print("\nStructures are sufficiently similar.")

    if args.verbosity >= 1:
        msg_program_footer(prg_start_time)


if __name__ == '__main__':
    main()
