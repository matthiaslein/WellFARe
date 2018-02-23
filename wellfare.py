from molecule import Molecule
from qmparser import extractMolecularData

################################################################################
#                                                                              #
# This is the part of the program where the command line arguments are defined #
#                                                                              #
################################################################################

parser = argparse.ArgumentParser(
    description="WellFAReFF: Method for the Incremental Construction and Exploration of the Potential Energy Surface",
    epilog="recognised filetypes: gaussian, orca, turbomole, xyz")
# parser.add_argument("-P", "--numproc", type=int, help="number of processes for parallel execution",
#                     default="2")
parser.add_argument("-v", "--verbosity", help="increase output verbosity", type=int, choices=[0, 1, 2, 3], default=0)
parser.add_argument("-o", "--output",
                    help="type of output file to be written: xyz file or just cartesian coordinates without header",
                    choices=["xyz", "cart"],
                    default="cart")
parser.add_argument("-g", "--group", help="replacement group", choices=["methyl", "ethenyl", "oh", "oh2", "nh2", "nh3", "cho", "f", "cl", "br", "i"], default="methyl")
parser.add_argument("inputfile", metavar='file',
                    help="input file(s) with molecular structure")
parser.add_argument("-r", "--replace", type=int, nargs='+', help="list of individual hydrogen atoms to replace")
parser.add_argument("-a", "--alloncarbon", type=int, nargs='+',
                    help="list of carbon atoms whose hydrogen atoms should be replaced")
parser.add_argument("-t", "--terminal", type=int, help="number of terminal hydrogen atoms to replace")

args = parser.parse_args()

def main():
    pass

if __name__ == '__main__':
    main()