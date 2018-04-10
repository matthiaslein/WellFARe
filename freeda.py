#############################################################################
#                                                                            #
# This is the part of the program where modules are imported                 #
#                                                                            #
##############################################################################

import argparse
import math
import itertools
import scipy.misc as misc
from multiprocessing import Pool, cpu_count

from messages import *
from molecule import Molecule
from qmparser import extract_molecular_data
from thermochemistry import ThermodynamicState, thermochemical_analysis



###############################################################################
#                                                                             #
# This is the part of the program where the thermochemical comparison is done #
#                                                                             #
###############################################################################

def thermochemical_comparison(reactants, products, temp=298.15, press=101325.0,
                              analysis=False, verbosity=0):
    # Note that temperature and pressure are only used for printing, all the
    # calculations are done in thermochemical_analysis() and *not* here!

    rSumMass = 0.0
    rSum_Ee = 0.0
    rSum_ZPVE = 0.0
    rSum_thermVib = 0.0
    rSum_thermRot = 0.0
    rSum_thermTrans = 0.0
    rSum_kT = 0.0
    rSum_transS = 0.0
    rSum_elecS = 0.0
    rSum_rotS = 0.0
    rSum_vibS = 0.0
    rSum_negTS = 0.0
    for i in reactants:
        rSumMass += i.mass()
        rSum_Ee += i.Ee_QM
        rSum_ZPVE += i.ZPVE
        rSum_thermVib += i.thermVib
        rSum_thermRot += i.thermRot
        rSum_thermTrans += i.thermTrans
        rSum_kT += i.kT
        rSum_transS += i.transS
        rSum_elecS += i.elecS
        rSum_rotS += i.rotS
        rSum_vibS += i.vibS
        rSum_negTS += i.negTS
    rSum_U = rSum_Ee + rSum_ZPVE + rSum_thermVib + rSum_thermRot + rSum_thermTrans
    rSum_S = rSum_transS + rSum_elecS + rSum_rotS + rSum_vibS
    rSum_H = rSum_U + rSum_kT
    rSum_G = rSum_H + rSum_negTS

    pSumMass = 0.0
    pSum_Ee = 0.0
    pSum_ZPVE = 0.0
    pSum_thermVib = 0.0
    pSum_thermRot = 0.0
    pSum_thermTrans = 0.0
    pSum_kT = 0.0
    pSum_transS = 0.0
    pSum_elecS = 0.0
    pSum_rotS = 0.0
    pSum_vibS = 0.0
    pSum_negTS = 0.0
    for i in products:
        pSumMass += i.mass()
        pSum_Ee += i.Ee_QM
        pSum_ZPVE += i.ZPVE
        pSum_thermVib += i.thermVib
        pSum_thermRot += i.thermRot
        pSum_thermTrans += i.thermTrans
        pSum_kT += i.kT
        pSum_transS += i.transS
        pSum_elecS += i.elecS
        pSum_rotS += i.rotS
        pSum_vibS += i.vibS
        pSum_negTS += i.negTS
    pSum_U = pSum_Ee + pSum_ZPVE + pSum_thermVib + pSum_thermRot + pSum_thermTrans
    pSum_S = pSum_transS + pSum_elecS + pSum_rotS + pSum_vibS
    pSum_H = pSum_U + pSum_kT
    pSum_G = pSum_H + pSum_negTS

    diffSumMass = math.fabs(pSumMass - rSumMass)
    diff_Ee = pSum_Ee - rSum_Ee
    diff_ZPVE = pSum_ZPVE - rSum_ZPVE
    diff_thermVib = pSum_thermVib - rSum_thermVib
    diff_thermRot = pSum_thermRot - rSum_thermRot
    diff_thermTrans = pSum_thermTrans - rSum_thermTrans
    diff_U = pSum_U - rSum_U
    diff_kT = pSum_kT - rSum_kT
    diff_H = pSum_H - rSum_H
    diff_elecS = pSum_elecS - rSum_elecS
    diff_transS = pSum_transS - rSum_transS
    diff_rotS = pSum_rotS - rSum_rotS
    diff_vibS = pSum_vibS - rSum_vibS
    diff_S = pSum_S - rSum_S
    diff_negTS = pSum_negTS - rSum_negTS
    diff_G = pSum_G - rSum_G

    # Branch into printing different tables for (a) reactions and (b) just a sum of compounds
    if products != []:

        conv = 2625.5002  # Conversion factor for hartree to kJ mol⁻¹
        print(
            "#############################################################################")
        print("Thermodynamic changes during the reaction at:")
        print("{:.2f} K ({:+.2f} °C, {:+.2f} °F)".format(temp, temp - 273.15, (
                temp * (9 / 5)) - 459.67))
        print("{:.2f} Pa ({:.2f} atm, {:.2f} mmHg)".format(press,
                                                           press / 101325.0,
                                                           press * 0.007500617))
        print(
            "-----------------------------------------------------------------------------")
        print(
            "                                  Reactants        Products    Reaction")
        print(
            "=============================================================================")
        print(
            "                                         in hartrees (h)          in kJ mol⁻¹")
        print(
            "-----------------------------------------------------------------------------")
        print(
            "Electronic energy:             {:> 12.6f}      {:> 12.6f}      {:+10.2f}".format(
                rSum_Ee, pSum_Ee, diff_Ee * conv))
        print(
            "Zero point vibrational energy: {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
                rSum_ZPVE, pSum_ZPVE, diff_ZPVE * conv))
        print(
            "Energy at absolute zero:       {:> 12.6f}      {:> 12.6f}   ΔE={:+10.2f}\n".format(
                rSum_Ee + rSum_ZPVE, pSum_Ee + pSum_ZPVE,
                (diff_Ee + diff_ZPVE) * conv))
        print(
            "Vibrational contribution:      {:> 12.6f}      {:> 12.6f}      {:+10.2f}".format(
                rSum_thermVib, pSum_thermVib, diff_thermVib * conv))
        print(
            "Rotational contributions:      {:> 12.6f}      {:> 12.6f}      {:+10.2f}".format(
                rSum_thermRot, pSum_thermRot, diff_thermRot * conv))
        print(
            "Translational contributions:   {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
                rSum_thermTrans, pSum_thermTrans, diff_thermTrans * conv))
        print(
            "Internal energy (U):           {:> 12.6f}      {:> 12.6f}   ΔU={:+10.2f}\n".format(
                rSum_U, pSum_U, diff_U * conv))
        print(
            "Volume work (pΔV = ΔnkT):      {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
                rSum_kT, pSum_kT, diff_kT * conv))
        print(
            "Enthalpy (H = U + pΔV):        {:> 12.6f}      {:> 12.6f}   ΔH={:+10.2f}\n".format(
                rSum_H, pSum_H, diff_H * conv))
        print(
            "=============================================================================")
        print("                                                in J mol⁻¹ K⁻¹")
        print(
            "-----------------------------------------------------------------------------")
        print(
            "Electronic entropy:          {:> 10.2f}        {:> 10.2f}          {:+10.2f}".format(
                rSum_elecS, pSum_elecS, diff_elecS))
        print(
            "Translational entropy:       {:> 10.2f}        {:> 10.2f}          {:+10.2f}".format(
                rSum_transS, pSum_transS, diff_transS))
        print(
            "Rotational entropy:          {:> 10.2f}        {:> 10.2f}          {:+10.2f}".format(
                rSum_rotS, pSum_rotS, diff_rotS))
        print(
            "Vibrational entropy:         {:> 10.2f}        {:> 10.2f}          {:+10.2f}\n".format(
                rSum_vibS, pSum_vibS, diff_vibS))
        print(
            "Total entropy (S):           {:> 10.2f}        {:> 10.2f}       ΔS={:+10.2f}\n".format(
                rSum_S, pSum_S, diff_S))
        print(
            "=============================================================================")
        print(
            "                                         in hartrees (h)          in kJ mol⁻¹")
        print(
            "-----------------------------------------------------------------------------")
        print(
            "Environment term (-TS):        {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
                rSum_negTS, pSum_negTS, diff_negTS * conv))
        print(
            "=============================================================================")
        print(
            "Gibbs energy (G = H - TS):     {:> 12.6f}      {:> 12.6f}   ΔG={:+10.2f}".format(
                rSum_G,
                pSum_G,
                diff_G * conv))
        print(
            "#############################################################################")
        # Sanity check: if the sum of masses isn't the same, there's a problem
        if diffSumMass >= 0.01:
            msg_program_warning(
                "There don't seem to be the same numbers/types of atoms")
    else:
        conv = 2625.5002  # Conversion factor for hartree to kJ mol⁻¹
        print(
            "#############################################################################")
        if len(reactants) == 1:
            print("Thermodynamics for the input compounds at:")
        else:
            print(
                "Combined thermodynamics for all {} input compounds at:".format(
                    len(reactants)))
        print("{:.2f} K ({:+.2f} °C, {:+.2f} °F)".format(temp, temp - 273.15, (
                temp * (9 / 5)) - 459.67))
        print("{:.2f} Pa ({:.2f} atm, {:.2f} mmHg)".format(press,
                                                           press / 101325.0,
                                                           press * 0.007500617))
        print(
            "=============================================================================")
        print("                                         in hartrees (h)")
        print(
            "-----------------------------------------------------------------------------")
        print("Electronic energy:                      {:> 12.6f}".format(
            rSum_Ee))
        if len(reactants) == 1 and analysis == True:
            print("\n Contributions to the ZPVE:")
            for i, j in enumerate(reactants[0].ZPVEList):
                print(
                    " Vibration {:4} ({:6.0f} cm⁻¹):  {:> 12.6f}".format(i + 1,
                                                                         reactants[
                                                                             0].frequencies[
                                                                             i],
                                                                         j))
            print(
                "Total zero point vibrational energy:    {:> 12.6f}\n".format(
                    rSum_ZPVE))
        else:
            print(
                "Zero point vibrational energy:          {:> 12.6f}\n".format(
                    rSum_ZPVE))
        print("Energy at absolute zero:                {:> 12.6f}\n".format(
            rSum_Ee + rSum_ZPVE))

        print("Translational contributions:            {:> 12.6f}".format(
            rSum_thermTrans))
        if len(reactants) == 1:
            print(
                "Rotational contributions (σ = {:2}):      {:> 12.6f}".format(
                    reactants[0].sigmaRot, rSum_thermRot))
        else:
            print("Rotational contributions:               {:> 12.6f}".format(
                rSum_thermRot))
        if len(reactants) == 1 and analysis == True:
            print("\n Vibrational contributions:")
            for i, j in enumerate(reactants[0].thermVibList):
                print(
                    " Vibration {:4} ({:6.0f} cm⁻¹):  {:> 12.6f}".format(i + 1,
                                                                         reactants[
                                                                             0].frequencies[
                                                                             i],
                                                                         j))
        if len(reactants) == 1:
            print(
                "Total vibrational contribution:         {:> 12.6f}\n".format(
                    rSum_thermVib))
        else:
            print("Vibrational contribution:               {:> 12.6f}".format(
                rSum_thermVib))
        print("Internal energy (U):                    {:> 12.6f}\n".format(
            rSum_U))
        print("Volume work (pΔV = ΔnkT):               {:> 12.6f}\n".format(
            rSum_kT))
        print("Enthalpy (H = U + pΔV):                 {:> 12.6f}\n".format(
            rSum_H))
        print(
            "=============================================================================")
        print("                                        in J mol⁻¹ K⁻¹")
        print(
            "-----------------------------------------------------------------------------")
        print("Electronic entropy:                   {:> 10.2f}".format(
            rSum_elecS))
        print("Translational entropy:                {:> 10.2f}".format(
            rSum_transS))
        print("Rotational entropy:                   {:> 10.2f}".format(
            rSum_rotS))
        if len(reactants) == 1 and analysis == True:
            print("\n Vibrational contributions to the entropy:")
            for i, j in enumerate(reactants[0].VibSList):
                print(
                    " Vibration {:4} ({:6.0f} cm⁻¹):  {:> 10.2f}".format(i + 1,
                                                                         reactants[
                                                                             0].frequencies[
                                                                             i],
                                                                         j))
            print("Total vibrational entropy:            {:> 10.2f}\n".format(
                rSum_vibS))
        else:
            print("Vibrational entropy:                  {:> 10.2f}\n".format(
                rSum_vibS))
        print("Total entropy (S):                    {:> 10.2f}\n".format(
            rSum_S))
        print(
            "=============================================================================")
        print("                                         in hartrees (h)")
        print(
            "-----------------------------------------------------------------------------")
        print("Environment term (-TS):                 {:> 12.6f}\n".format(
            rSum_negTS))
        print(
            "=============================================================================")
        print(
            "Gibbs energy (G = H - TS):              {:> 12.6f}".format(
                rSum_G))
        print(
            "#############################################################################")




############################################################################
#                                                                          #
# This is the part of the program where the cmd line arguments are defined #
#                                                                          #
############################################################################

parser = argparse.ArgumentParser(
    description="freeda: Free Energy Decomposition Analysis",
    epilog="recognised filetypes: gaussian, orca, turbomole")
parser.add_argument("-P", "--numproc", type=int, help="number of processes "
                                                      "for parallel execution",
                    default="0")
parser.add_argument("--temperature", type=float, help="Temperature (in Kelvin)"
                                                      " for the thermochemical"
                                                      " analysis",
                    default="298.15")
parser.add_argument("--pressure", type=float,
                    help="Pressure (in Pa) for the thermochemical analysis",
                    default="101325.0")
parser.add_argument("--scalefreq", type=float,
                    help="Scale harmonic frequencies by a constant factor",
                    default="1.0")
parser.add_argument("-a", "--analysis",
                    help="For single compounds, analyse contributions"
                         " from each vibrational mode", action='store_true')
parser.add_argument("-r", "--reactants", metavar='file', nargs='+',
                    help="input file(s) with qc data of the reactant(s)",
                    required=True)
parser.add_argument("-p", "--products", metavar='file', nargs='+',
                    help="input file(s) with qc data of the product(s)")
parser.add_argument("-v", "--verbosity", help="increase output verbosity",
                    type=int, choices=[0, 1, 2, 3], default=0)

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
        msg_program_header("Freeda", 1.0)

    list_of_reactants = []
    if type(args.reactants) is str:
        list_of_reactants.append(Molecule(args.reactants, 0))
        extract_molecular_data(args.reactants, list_of_reactants[-1],
                               verbosity=args.verbosity, read_coordinates=True,
                               read_bond_orders=True, build_angles=True,
                               build_dihedrals=True, cpu_number=args.numproc)
        thermochemical_analysis(list_of_reactants[-1], temp=args.temperature,
                                press=args.pressure, scalefreq=args.scalefreq,
                                verbosity=args.verbosity)
    else:
        for i in args.reactants:
            list_of_reactants.append(Molecule(i, 0))
            extract_molecular_data(i, list_of_reactants[-1],
                                   verbosity=args.verbosity,
                                   read_coordinates=True,
                                   read_bond_orders=True, build_angles=True,
                                   build_dihedrals=True,
                                   cpu_number=args.numproc)
            thermochemical_analysis(list_of_reactants[-1],
                                    temp=args.temperature,
                                    press=args.pressure,
                                    scalefreq=args.scalefreq,
                                    verbosity=args.verbosity)

    list_of_products = []
    if args.products is not None and type(args.products) is str:
        list_of_products.append(Molecule(args.products, 0))
        extract_molecular_data(args.products, list_of_products[-1],
                               verbosity=args.verbosity, read_coordinates=True,
                               read_bond_orders=True, build_angles=True,
                               build_dihedrals=True, cpu_number=args.numproc)
        thermochemical_analysis(list_of_products[-1], temp=args.temperature,
                                press=args.pressure, scalefreq=args.scalefreq,
                                verbosity=args.verbosity)
    elif args.products is not None:
        for i in args.products:
            list_of_products.append(Molecule(i, 0))
            extract_molecular_data(i, list_of_products[-1],
                                   verbosity=args.verbosity,
                                   read_coordinates=True,
                                   read_bond_orders=True, build_angles=True,
                                   build_dihedrals=True,
                                   cpu_number=args.numproc)
            thermochemical_analysis(list_of_products[-1], temp=args.temperature,
                                    press=args.pressure,
                                    scalefreq=args.scalefreq,
                                    verbosity=args.verbosity)
    else:
        if args.verbosity >= 1:
            print("No products read")

    thermochemical_comparison(list_of_reactants, list_of_products,
                              temp=args.temperature, press=args.pressure,
                              analysis=args.analysis, verbosity=args.verbosity)

    if args.verbosity >= 1:
        msg_program_footer(prg_start_time)


if __name__ == '__main__':
    main()
