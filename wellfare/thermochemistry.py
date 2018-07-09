import math

import numpy as np

# from typing import Optional
#
# from molecule import Molecule
from wellfare.conversions import *
from wellfare.constants import *
from wellfare.messages import *


###############################################################################
#                                                                             #
# This is the part of the program where the vibrational analysis is done      #
#                                                                             #
###############################################################################

def thermochemical_analysis(molecule, temp=298.15, press=101325.0,
                            scalefreq=1.0, internal="truhlar", cutoff=100.0,
                            verbosity=0):
    # Constants
    # k_boltz() = 1.38064852E-23  # Boltzmann constant in JK⁻¹
    # h_planck() = 6.626070040E-34  # Planck constant in Js
    # r_gas() = 8.3144598  # Gas constant in J K⁻¹ mol⁻¹
    # c_light() = 2.9979258E10  # Speed of light in cm s⁻¹

    # Print T and p conditions
    if verbosity >= 2:
        print("\n Thermal analysis for WellFARe molecule: ", molecule.name)
        print(" Temperature: {:> 12.2f} K".format(temp))
        print(" Pressure   : {:> 12.2f} Pa".format(press))

    # Electronic contribution to the entropy
    Qelec = molecule.mult
    molecule.elecS = r_gas() * math.log(Qelec)
    if verbosity >= 2:
        print(
            "\n Electronic partition function:                  {:> 12.5E}".format(
                Qelec))
        print(
            " Electronic contribution to the entropy:         {:> 9.2f} J mol⁻¹ K⁻¹".format(
                molecule.elecS))

    # Translational Partition Function
    Qtrans = (((2 * math.pi * (
            molecule.mass() / 6.0221409E26) * k_boltz() * temp) / (
                       h_planck() ** 2)) ** (3 / 2)) * (
                     (k_boltz() * temp) / press)
    molecule.transS = r_gas() * (math.log(Qtrans) + 1 + (3 / 2))
    Etrans = (3 / 2) * r_gas() * temp
    molecule.thermTrans = Etrans / 2625500.2
    if verbosity >= 2:
        print(
            "\n Translational partition function:               {:> 12.5E}".format(
                Qtrans))
        print(
            " Translational contribution to the total energy: {:> 12.2f} J mol⁻¹".format(
                Etrans))
        print(
            " Translational contribution to the entropy:      {:> 12.2f} J mol⁻¹ K⁻¹".format(
                molecule.transS))

    # Rotational Partition Function
    # We need to distinguish three cases: (1) single atom, (2) linear molecule, (3) non-linear molecule
    # For cases (2) and (3), we also need to determine the correct symmetry number sigma_r

    if molecule.num_atoms() == 1:
        if verbosity >= 3:
            print(" No rotational contribution for single atoms.")
        # Case (1): single atom
        q_rot = 1.0
        molecule.rotS = 0.0
        e_rot = 0.0
    else:
        # Now we need to calculate the moments of inertia to determine if the molecule is linear or not
        # (and to guesstimate the symmetry number)
        # The molecular center of mass
        xValue = 0.0
        yValue = 0.0
        zValue = 0.0
        for i in molecule.atoms:
            xValue = xValue + (i.mass * i.coord[0])
            yValue = yValue + (i.mass * i.coord[1])
            zValue = zValue + (i.mass * i.coord[2])
        xValue = xValue / (molecule.mass())
        yValue = yValue / (molecule.mass())
        zValue = zValue / (molecule.mass())

        # Translate whole molecule into the center of mass reference frame
        for i in molecule.atoms:
            i.coord[0] = i.coord[0] - xValue
            i.coord[1] = i.coord[1] - yValue
            i.coord[2] = i.coord[2] - zValue

        # Build inertia tensor
        inertia_tensor = []
        Ixx = 0.0
        Ixy = 0.0
        Ixz = 0.0
        Iyx = 0.0
        Iyy = 0.0
        Iyz = 0.0
        Izx = 0.0
        Izy = 0.0
        Izz = 0.0
        for i in molecule.atoms:
            Ixx = Ixx + (
                    i.mass * (
                    (ang_to_bohr(i.coord[1]) * ang_to_bohr(i.coord[1])) + (
                    ang_to_bohr(i.coord[2]) * ang_to_bohr(i.coord[2]))))
            Ixy = Ixy - i.mass * ang_to_bohr(i.coord[0]) * ang_to_bohr(
                i.coord[1])
            Ixz = Ixz - i.mass * ang_to_bohr(i.coord[0]) * ang_to_bohr(
                i.coord[2])
            Iyx = Iyx - i.mass * ang_to_bohr(i.coord[1]) * ang_to_bohr(
                i.coord[0])
            Iyy = Iyy + (
                    i.mass * (
                    (ang_to_bohr(i.coord[0]) * ang_to_bohr(i.coord[0])) + (
                    ang_to_bohr(i.coord[2]) * ang_to_bohr(i.coord[2]))))
            Iyz = Iyz - i.mass * ang_to_bohr(i.coord[1]) * ang_to_bohr(
                i.coord[2])
            Izx = Izx - i.mass * ang_to_bohr(i.coord[2]) * ang_to_bohr(
                i.coord[0])
            Izy = Izy - i.mass * ang_to_bohr(i.coord[2]) * ang_to_bohr(
                i.coord[1])
            Izz = Izz + (
                    i.mass * (
                    (ang_to_bohr(i.coord[0]) * ang_to_bohr(i.coord[0])) + (
                    ang_to_bohr(i.coord[1]) * ang_to_bohr(i.coord[1]))))
        inertia_tensor.append([Ixx, Ixy, Ixz])
        inertia_tensor.append([Iyx, Iyy, Iyz])
        inertia_tensor.append([Izx, Izy, Izz])
        inertia_tensor = np.matrix(inertia_tensor)

        # Diagonalise inertia tensor
        inertia_moments, inertial_axes = np.linalg.eig(inertia_tensor)

        # Orthogonalise eigenvectors (only sometimes necessary)...
        inertial_axes, r = np.linalg.qr(inertial_axes)

        # Sort moments from highest to lowest
        idx = inertia_moments.argsort()[::-1]
        inertia_moments = inertia_moments[idx]
        inertial_axes = inertial_axes[:, idx]
        if verbosity >= 3:
            print("\n Moments of inertia in atomic units:")
            print(inertia_moments)

        if inertia_moments[0] - inertia_moments[1] < 0.0005:
            # Case (2): linear molecule
            if verbosity >= 3:
                print("\n Rotational contribution for a linear molecule.")
            phi = (h_planck() ** 2) / (8 * (math.pi ** 2) * (
                    inertia_moments[0] * 4.6468689E-48) * k_boltz())
            q_rot = (1 / molecule.sigmaRot) * (temp / phi)
            molecule.rotS = r_gas() * (math.log(q_rot) + 1)
            e_rot = r_gas() * temp
        else:
            # Case (3): nonlinear molecule
            if verbosity >= 3:
                print("\n Rotational contribution for a nonlinear molecule.")
            phi_x = (h_planck() ** 2) / (8 * (math.pi ** 2) * (
                    inertia_moments[0] * 4.6468689E-48) * k_boltz())
            phi_y = (h_planck() ** 2) / (8 * (math.pi ** 2) * (
                    inertia_moments[1] * 4.6468689E-48) * k_boltz())
            phi_z = (h_planck() ** 2) / (8 * (math.pi ** 2) * (
                    inertia_moments[2] * 4.6468689E-48) * k_boltz())
            q_rot = (math.sqrt(math.pi) / molecule.sigmaRot) * (
                    math.sqrt(temp ** 3) / math.sqrt(phi_x * phi_y * phi_z))
            molecule.rotS = r_gas() * (math.log(q_rot) + (3 / 2))
            e_rot = (3 / 2) * r_gas() * temp

    if verbosity >= 2:
        print(
            "\n Rotational partition function:                  {:> 12.5E}".format(
                q_rot))
        print(
            " Rotational contribution to the total energy:    {:> 12.2f} J mol⁻¹".format(
                e_rot))
        print(
            " Rotational contribution to the entropy:         {:> 12.2f} J mol⁻¹ K⁻¹\n".format(
                molecule.rotS))

    if molecule.num_atoms() != 1 and (
            molecule.H_QM != [] or molecule.frequencies != []):
        # If we didn't read frequencies from file, calculate them from force constants
        if molecule.frequencies == []:
            if verbosity >= 2:
                print("\n Vibrational analysis:")
            # First, we need to form the mass-weighted force-constant matrix
            molecule.H_mw = np.zeros((len(molecule.H_QM), len(molecule.H_QM)))
            for i in range(0, molecule.num_atoms()):
                for j in range(0, molecule.num_atoms()):
                    for h in range(0, 3):
                        for l in range(0, 3):
                            molecule.H_mw[3 * i + h][3 * j + l] = \
                                molecule.H_QM[3 * i + h][
                                    3 * j + l] / math.sqrt(
                                    symbol_to_au_mass[
                                        molecule.atm_symbol(i)] *
                                    symbol_to_au_mass[
                                        molecule.atm_symbol(j)])
            frequencies, normalModes = np.linalg.eig(molecule.H_mw)

            # We need to distinguish two cases: (1) linear molecule with 3N-5 vibrations and (2) non-linear
            # molecules with 3N-6 vibrations.
            listOfFreqs = []
            if inertia_moments[0] - inertia_moments[1] < 0.0005 and inertia_moments[2] < 0.0005:
                # Case (1): linear molecule
                if verbosity >= 3:
                    print(
                        "\n Vibrational contribution for a linear molecule (3N-5)")
                if verbosity >= 3:
                    print(
                        "\n Frequencies that correspond to rotation and translation (lowest 5):")
                    for i in np.sort(frequencies)[:5]:
                        conversion = 219474.63  # atomic units to cm⁻¹ in a harmonic oscillator
                        if i < 0.0:
                            sign = -1
                        else:
                            sign = 1
                        i = sign * math.sqrt(abs(i)) * conversion
                        print("{:> 9.2f} cm⁻¹".format(i))

                for i in np.sort(frequencies)[5:]:
                    conversion = 219474.63  # atomic units to cm⁻¹ in a harmonic oscillator
                    if i < 0.0:
                        sign = -1
                    else:
                        sign = 1
                    i = sign * math.sqrt(abs(i)) * conversion

                    # We don't add imaginary modes to the thermochemical analysis
                    if sign == 1:
                        listOfFreqs.append(i)
                # Then save the list to the molecule
                molecule.frequencies = listOfFreqs

            else:
                # Case (2): nonlinear molecule
                if verbosity >= 3:
                    print(
                        "\n Vibrational contribution for a nonlinear molecule (3N-6)")
                if verbosity >= 3:
                    print(
                        "\n Frequencies that correspond to rotation and translation (lowest 6):")
                    for i in np.sort(frequencies)[:6]:
                        conversion = 219474.63  # atomic units to cm⁻¹ in a harmonic oscillator
                        if i < 0.0:
                            sign = -1
                        else:
                            sign = 1
                        i = sign * math.sqrt(abs(i)) * conversion
                        print("{:> 9.2f} cm⁻¹".format(i))
                for i in np.sort(frequencies)[6:]:
                    conversion = 219474.63  # atomic units to cm⁻¹ in a harmonic oscillator
                    if i < 0.0:
                        sign = -1
                    else:
                        sign = 1
                    i = sign * math.sqrt(abs(i)) * conversion

                    # We don't add imaginary modes to the thermochemical analysis
                    if sign == 1:
                        listOfFreqs.append(i)
                # Then save the list to the molecule
                molecule.frequencies = listOfFreqs
            if verbosity >= 2:
                print(
                    "\n Frequencies (before contamination with rotation and translation has been removed):")
                if verbosity >= 3:
                    print(" (imaginary modes will be ignored)")
                for i in listOfFreqs:
                    print("{:> 9.2f} cm⁻¹".format(i))
        else:
            # If we've read the frequencies from file, just use those
            listOfFreqs = molecule.frequencies

        # Check if we have been asked to scale all frequencies
        if scalefreq != 1.0:
            if verbosity >= 1:
                print("\n Scaling all frequencies by a factor of {}".format(
                    scalefreq))
            for i in range(0, len(listOfFreqs)):
                if verbosity >= 1:
                    print(
                        " Vibration {} Before: {:> 9.2f} cm⁻¹, After: {:> 9.2f} cm⁻¹".format(
                            i + 1, listOfFreqs[i],
                            listOfFreqs[
                                i] * scalefreq))
                listOfFreqs[i] = listOfFreqs[i] * scalefreq

        # Following Truhlar's advice, very low frequencies (below
        # 100 cm⁻¹) should be fixed to exactly 100 cm⁻¹. We check
        # if this option is selected and then proceed:
        if internal == "truhlar":
            if verbosity >= 1:
                print("\n Fixing all low frequencies to {:> 6.1f}"
                      " cm⁻¹".format(cutoff))
            for i in range(0, len(listOfFreqs)):
                if 0.0 < listOfFreqs[i] < cutoff:
                    if verbosity >= 1:
                        print(
                            " Vibration {} Before: {:> 6.1f} cm⁻¹,"
                            " After: {:> 6.1f} cm⁻¹".format(i + 1, listOfFreqs[i],cutoff))
                    listOfFreqs[i] = cutoff
        # Alternatively, all low-lying modes can be simply ignored completely
        # Here, we make assumptions about which modes may have an outsize
        # contribution and completely delete the corresponding frequencies.
        elif internal == "ignorelow":
            if verbosity >= 1:
                print("\n Discarding all low frequencies below {:> 6.1f}"
                      " cm⁻¹".format(cutoff))
            # We're going through this list in reverse order, so we can delete
            # matching items straight away without messing up the list index
            for i in range(len(listOfFreqs)-1,-1,-1):
                if 0.0 < listOfFreqs[i] < cutoff:
                    if verbosity >= 1:
                        print(
                            " Vibration {} ({:> 6.1f} cm⁻¹),"
                            " discarded".format(i + 1, listOfFreqs[i]))
                        del listOfFreqs[i]

        if verbosity >= 2:
            print(
                "\n Frequencies (after scaling and internal rotor treatment):")
            if verbosity >= 3:
                print(" (imaginary modes will be ignored)")
            for i in listOfFreqs:
                print("{:> 9.2f} cm⁻¹".format(i))

        # Then, create a list that contains all vibrational temperatures (makes the summation of the
        # partition function simpler/more convenient)
        listOfVibTemps = []
        if verbosity >= 2:
            print("\n Vibrational temperatures:")
        ZPVE = 0.0
        for i in listOfFreqs:
            nu = i * c_light()
            phi = h_planck() * nu / k_boltz()
            # Still need to check for imaginary modes, since we might have just read the frequencies
            # instead of calculating them ourselves.
            if i > 0.0:
                molecule.ZPVEList.append(
                    (h_planck() * nu * 6.0221409E23) / 2.0 / 2625500.2)
                ZPVE += h_planck() * nu * 6.0221409E23
                listOfVibTemps.append(phi)
            if verbosity >= 2:
                print("{:> 9.2f} K".format(phi))
        molecule.ZPVE = (
                                ZPVE / 2.0) / 2625500.2  # Converted from Joules to Hartree straight away
        # Now, assemble the partition function, vibrational entropy and the vibrational contribution to
        # the total thermal energy along with the zero point vibrational energy
        Qvib = 1.0
        Svib = 0.0
        Evib = 0.0
        if verbosity >= 2 and internal == "grimme":
            print("\n Interpolation between entropic contributions from"
                  " harmonic oscillators and free rotors for low frequency"
                  " modes:")
        for j, i in enumerate(listOfVibTemps):
            Qvib = Qvib * (math.exp(-1.0 * i / (2.0 * temp))) / (
                    1 - math.exp(-1.0 * i / (temp)))
            if internal == "grimme":
                # Frequency of the vibration = freq of free rotor
                omega = listOfFreqs[j] * c_light()
                # Moment of the free rotor
                mu = h_planck()/(8*math.pi*math.pi*omega)
                # Reduced value for the actual moment to make sure its value
                # is within reasonable bounds
                B_av = (inertia_moments[0] + inertia_moments[1] +
                        inertia_moments[2]) / 3
                mu_prime = (mu * B_av)/(mu + B_av)
                # Vibrational entropy as usual
                vibrational_entropy = ((i / temp) / (
                        math.exp(i / (temp)) - 1.0)) - math.log(
                    1.0 - math.exp(-1.0 * i / (temp)))
                vibrational_entropy = vibrational_entropy * r_gas()
                # Rotational entropy of the free rotor
                rotational_entropy = (8 * (
                            math.pi ** 3) * mu_prime * k_boltz() * temp) / (
                                                 h_planck() * h_planck())
                rotational_entropy = math.log(math.sqrt(rotational_entropy))
                rotational_entropy = (0.5 + rotational_entropy) * r_gas()
                # Weighting function to balance between vib and rot entropy
                w = 1/(1+(((cutoff* c_light())/omega)**4))
                if verbosity >= 3:
                    print(" Interpolation for vibration {}: {}% Svib, {}% Srot.".format(j,int(w*100),int((1-w)*100)))
                SvibContribution = (w * vibrational_entropy) + (
                            (1 - w) * rotational_entropy)
            else:
                SvibContribution = ((i / temp) / (
                        math.exp(i / (temp)) - 1.0)) - math.log(
                    1.0 - math.exp(-1.0 * i / (temp)))
                SvibContribution = SvibContribution * r_gas()
            Svib += SvibContribution
            molecule.VibSList.append(SvibContribution * r_gas())
            EvibContribution = i * (0.5 + (1 / (math.exp(i / temp) - 1.0)))
            Evib += EvibContribution * r_gas()
            molecule.thermVibList.append(
                math.fabs((EvibContribution * r_gas()) - (
                        molecule.ZPVEList[j] * 2625500.2)) / 2625500.2)

    else:
        # If there are no vibrations, i.e. for a single atom, the results are straightforward:
        if verbosity >= 2:
            print(
                "\n No vibrational analysis. This is either"
                " a single atom, or we didn't find any data.")
        Qvib = 1.0
        Svib = 0.0
        molecule.vibS = 0.0
        Evib = 0.0

    molecule.vibS = Svib
    molecule.thermRot = (e_rot / 2625500.2)
    molecule.thermVib = (Evib / 2625500.2) - molecule.ZPVE
    Etherm = molecule.ZPVE + molecule.thermVib + molecule.thermRot + molecule.thermTrans
    Etot = molecule.qm_energy + molecule.ZPVE + molecule.thermVib + molecule.thermRot + molecule.thermTrans
    molecule.kT = (
            k_boltz() / 4.3597482E-18 * temp)  # kT contributon to the enthalpy
    Htot = Etot + molecule.kT  # Enthalpy: H = Etot + kB T in Hartree
    Stot = (
            molecule.elecS + molecule.transS + molecule.rotS + molecule.vibS)  # Total entropy
    molecule.negTS = (
            -1.0 * temp * Stot / 2625500.2)  # Total entropic contribution TS
    Gtot = Htot + molecule.negTS
    if molecule.frequencies == [] and molecule.H_QM == []:
        msg_program_warning(
            "\nCareful! We don't have any vibrational data for this"
            " compound!\nDon't use any of the derived thermodynamic"
            " values ΔE, ΔU, ΔH, ΔS or ΔG!\n")
    if verbosity >= 2:
        print(
            "\n Vibrational partition function:"
            "                  {:> 12.5E}\n".format(
                Qvib))
        print(
            " Vibrational contribution to the entropy:"
            "         {:> 12.2f} J mol⁻¹ K⁻¹\n".format(
                molecule.vibS))
        print("  Contributions to the ZPVE:")
        for i, j in enumerate(molecule.ZPVEList):
            print(
                "  Vibration {:4}:"
                "                        {:> 12.2f} J mol⁻¹"
                    .format(i + 1, j * 2625500.2))
        print(
            " Total zero-point vibrational energy (ZPVE):"
            "            {:> 12.2f} J mol⁻¹\n"
                .format(molecule.ZPVE * 2625500.2))
        print("  Contributions to the vibrations:")
        for i, j in enumerate(molecule.thermVibList):
            print(
                "  Vibration {:4}:"
                "                        {:> 12.2f} J mol⁻¹"
                    .format(i + 1, j * 2625500.2))
        print(
            " Total vibrational contribution to the energy:"
            "    {:> 12.2f} J mol⁻¹"
                .format(molecule.thermVib * 2625500.2))

    if verbosity >= 1:
        print("\n Summary of the thermochemistry for: ", molecule.name)
        if verbosity >= 2:
            print(
                " The temperature is {:> 6.2f} K and"
                " the pressure is {:> 10.2f} Pa for this analysis"
                    .format(temp, press))
        print(
            " Electronic Energy (Ee):"
            "                                 {:> 12.6f} h"
                .format(molecule.qm_energy))
        print(
            " Electronic Energy (Ee + ZPVE):"
            "                          {:> 12.6f} h"
                .format(molecule.qm_energy + molecule.ZPVE))
        if verbosity >= 2:
            print(
                "   Translational contribution:"
                "                           {:> 12.6f} h"
                    .format(molecule.thermTrans))
            print(
                "   Rotational contribution:"
                "                              {:> 12.6f} h"
                    .format(molecule.thermRot))
            print(
                "   Zero point vibrational contribution:"
                "                  {:> 12.6f} h"
                    .format(molecule.ZPVE))
            print(
                "   Finite temperature vibrational contribution:"
                "          {:> 12.6f} h"
                    .format(molecule.thermVib))
            print(
                "  Total thermal contributions to the energy (Etherm):"
                "    {:> 12.6f} h"
                    .format(Etherm))
        print(
            " Internal energy (U = Ee + Etherm):"
            "                      {:> 12.6f} h"
                .format(Etot))
        if verbosity >= 2:
            print(
                "  kT (=pV) contribution to the enthalpy:"
                "                 {:> 12.6f} h"
                    .format(molecule.kT))
        print(
            " Enthalpy (H = U + kT):"
            "                                  {:> 12.6f} h"
                .format(Htot))
        if verbosity >= 3:
            print(
                "   Electronic entropy:"
                "                               {:> 12.2f} J mol⁻¹ K⁻¹"
                    .format(molecule.elecS))
            print(
                "   Translational entropy:"
                "                            {:> 12.2f} J mol⁻¹ K⁻¹"
                    .format(molecule.transS))
            print(
                "   Rotational entropy:"
                "                               {:> 12.2f} J mol⁻¹ K⁻¹"
                    .format(molecule.rotS))
            print(
                "   Vibrational entropy:"
                "                              {:> 12.2f} J mol⁻¹ K⁻¹"
                    .format(molecule.vibS))
        if verbosity >= 2:
            print(
                "  Total entropy:"
                "                                     {:> 12.2f} J mol⁻¹ K⁻¹"
                    .format(Stot))
            print(
                " Total entropic contribution to the Gibbs energy (-TS)"
                "   {:> 12.6f} h\n"
                    .format(molecule.negTS))
        print(
            " Total Gibbs energy (G = H - TS)"
            "                         {:> 12.6f} h\n"
                .format(Gtot))
