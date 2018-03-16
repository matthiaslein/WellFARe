#####################
#                   #
# Import statements #
#                   #
#####################

import time
import argparse
from multiprocessing import Pool, cpu_count

from atom import Atom
from molecule import Molecule
from qmparser import extract_molecular_data
from messages import *

########################################################
#                                                      #
# This is where the command line arguments are defined #
#                                                      #
########################################################

parser = argparse.ArgumentParser(
    description="WellFARe: Method for the Incremental Construction and Exploration of the Potential Energy Surface",
    epilog="recognised filetypes: gaussian, orca, turbomole, xyz")
parser.add_argument("-P", "--numproc", type=int,
                    help="number of processes for parallel execution",
                    default="0")
parser.add_argument("-v", "--verbosity", help="increase output verbosity",
                    type=int, choices=[0, 1, 2, 3], default=3)
parser.add_argument("inputfile", metavar='file',
                    help="input file(s) with molecular structure")

args = parser.parse_args()

# Default number of cores for processing is requested with "-p 0" and uses
# all available cores.
if args.numproc == 0:
    args.numproc = cpu_count()


############################################
#                                          #
# The main part of the program starts here #
#                                          #
############################################

def main():
    prg_start_time = time.time()

    # Print GPL v3 statement and program header
    if args.verbosity >= 1:
        msg_program_header("WellFARe", 0.9)

    molecule = Molecule("Input Structure", 0)
    extract_molecular_data(args.inputfile, molecule, verbosity=args.verbosity)

    # Print program footer
    if args.verbosity >= 1:
        msg_program_footer(prg_start_time)

if __name__ == '__main__':
    main()