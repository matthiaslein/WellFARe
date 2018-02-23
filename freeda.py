
################################################################################
#                                                                              #
# This is the part of the program where the vibrational analysis is done       #
#                                                                              #
################################################################################

def thermochemicalAnalysis(molecule, temp=298.15, press=101325.0, scalefreq=1.0, verbosity=0):
    # Constants
    kBoltz = 1.38064852E-23  # Boltzmann constant in JK⁻¹
    hPlanck = 6.626070040E-34  # Planck constant in Js
    RGas = 8.3144598  # Gas constant in J K⁻¹ mol⁻¹
    Clight = 2.9979258E10  # Speed of light in cm s⁻¹
    # Clight = 29979258         # Speed of light in m s⁻¹

    # Print T and p conditions
    if verbosity >= 2:
        print("\n Thermal analysis for WellFARe molecule: ", molecule.name)
        print(" Temperature: {:> 12.2f} K".format(temp))
        print(" Pressure   : {:> 12.2f} Pa".format(press))

    # Electronic contribution to the entropy
    Qelec = molecule.mult
    molecule.elecS = RGas * math.log(Qelec)
    if verbosity >= 2:
        print("\n Electronic partition function:                  {:> 12.5E}".format(Qelec))
        print(" Electronic contribution to the entropy:         {:> 9.2f} J mol⁻¹ K⁻¹".format(molecule.elecS))

    # Translational Partition Function
    Qtrans = (((2 * math.pi * (molecule.mass() / 6.0221409E26) * kBoltz * temp) / (hPlanck ** 2)) ** (3 / 2)) * (
        (kBoltz * temp) / press)
    molecule.transS = RGas * (math.log(Qtrans) + 1 + (3 / 2))
    Etrans = (3 / 2) * RGas * temp
    molecule.thermTrans = Etrans / 2625500.2
    if verbosity >= 2:
        print("\n Translational partition function:               {:> 12.5E}".format(Qtrans))
        print(" Translational contribution to the total energy: {:> 12.2f} J mol⁻¹".format(Etrans))
        print(" Translational contribution to the entropy:      {:> 12.2f} J mol⁻¹ K⁻¹".format(molecule.transS))

    # Rotational Partition Function
    # We need to distinguish three cases: (1) single atom, (2) linear molecule, (3) non-linear molecule
    # For cases (2) and (3), we also need to determine the correct symmetry number sigma_r

    if molecule.numatoms() == 1:
        if verbosity >= 3:
            print(" No rotational contribution for single atoms.")
        # Case (1): single atom
        Qrot = 1.0
        molecule.rotS = 0.0
        Erot = 0.0
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
        inertiaTensor = []
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
                    (Ang2Bohr(i.coord[1]) * Ang2Bohr(i.coord[1])) + (Ang2Bohr(i.coord[2]) * Ang2Bohr(i.coord[2]))))
            Ixy = Ixy - i.mass * Ang2Bohr(i.coord[0]) * Ang2Bohr(i.coord[1])
            Ixz = Ixz - i.mass * Ang2Bohr(i.coord[0]) * Ang2Bohr(i.coord[2])
            Iyx = Iyx - i.mass * Ang2Bohr(i.coord[1]) * Ang2Bohr(i.coord[0])
            Iyy = Iyy + (
                i.mass * (
                    (Ang2Bohr(i.coord[0]) * Ang2Bohr(i.coord[0])) + (Ang2Bohr(i.coord[2]) * Ang2Bohr(i.coord[2]))))
            Iyz = Iyz - i.mass * Ang2Bohr(i.coord[1]) * Ang2Bohr(i.coord[2])
            Izx = Izx - i.mass * Ang2Bohr(i.coord[2]) * Ang2Bohr(i.coord[0])
            Izy = Izy - i.mass * Ang2Bohr(i.coord[2]) * Ang2Bohr(i.coord[1])
            Izz = Izz + (
                i.mass * (
                    (Ang2Bohr(i.coord[0]) * Ang2Bohr(i.coord[0])) + (Ang2Bohr(i.coord[1]) * Ang2Bohr(i.coord[1]))))
        inertiaTensor.append([Ixx, Ixy, Ixz])
        inertiaTensor.append([Iyx, Iyy, Iyz])
        inertiaTensor.append([Izx, Izy, Izz])
        inertiaTensor = np.matrix(inertiaTensor)

        # Diagonalise inertia tensor
        inertiaMoments, inertialAxes = np.linalg.eig(inertiaTensor)

        # Orthogonalise eigenvectors (only sometimes necessary)...
        inertialAxes, r = np.linalg.qr(inertialAxes)

        # Sort moments from highest to lowest
        idx = inertiaMoments.argsort()[::-1]
        inertiaMoments = inertiaMoments[idx]
        inertialAxes = inertialAxes[:, idx]
        if verbosity >= 3:
            print("\n Moments of inertia in atomic units:")
            print(inertiaMoments)

        if inertiaMoments[0] - inertiaMoments[1] < 0.0005:
            # Case (2): linear molecule
            if verbosity >= 3:
                print("\n Rotational contribution for a linear molecule.")
            Phi = (hPlanck ** 2) / (8 * (math.pi ** 2) * (inertiaMoments[0] * 4.6468689E-48) * kBoltz)
            Qrot = (1 / molecule.sigmaRot) * (temp / Phi)
            molecule.rotS = RGas * (math.log(Qrot) + 1)
            Erot = RGas * temp
        else:
            # Case (3): nonlinear molecule
            if verbosity >= 3:
                print("\n Rotational contribution for a nonlinear molecule.")
            Phix = (hPlanck ** 2) / (8 * (math.pi ** 2) * (inertiaMoments[0] * 4.6468689E-48) * kBoltz)
            Phiy = (hPlanck ** 2) / (8 * (math.pi ** 2) * (inertiaMoments[1] * 4.6468689E-48) * kBoltz)
            Phiz = (hPlanck ** 2) / (8 * (math.pi ** 2) * (inertiaMoments[2] * 4.6468689E-48) * kBoltz)
            Qrot = (math.sqrt(math.pi) / molecule.sigmaRot) * (math.sqrt(temp ** 3) / math.sqrt(Phix * Phiy * Phiz))
            molecule.rotS = RGas * (math.log(Qrot) + (3 / 2))
            Erot = (3 / 2) * RGas * temp

    if verbosity >= 2:
        print("\n Rotational partition function:                  {:> 12.5E}".format(Qrot))
        print(" Rotational contribution to the total energy:    {:> 12.2f} J mol⁻¹".format(Erot))
        print(" Rotational contribution to the entropy:         {:> 12.2f} J mol⁻¹ K⁻¹\n".format(molecule.rotS))

    if molecule.numatoms() != 1 and (molecule.frequencies != [] or molecule.H_QM != []):
        # If we didn't read frequencies from file, calculate them from force constants
        if molecule.frequencies == []:
            if verbosity >= 2:
                print("\n Vibrational analysis:")
            # First, we need to form the mass-weighted force-constant matrix
            molecule.H_mw = np.zeros((len(molecule.H_QM), len(molecule.H_QM)))
            for i in range(0, molecule.numatoms()):
                for j in range(0, molecule.numatoms()):
                    for h in range(0, 3):
                        for l in range(0, 3):
                            molecule.H_mw[3 * i + h][3 * j + l] = molecule.H_QM[3 * i + h][3 * j + l] / math.sqrt(
                                SymbolToAUMass[molecule.atoms[i].symbol] * SymbolToAUMass[molecule.atoms[j].symbol])
            frequencies, normalModes = np.linalg.eig(molecule.H_mw)

            # We need to distinguish two cases: (1) linear molecule with 3N-5 vibrations and (2) non-linear
            # molecules with 3N-6 vibrations.
            listOfFreqs = []
            if inertiaMoments[0] - inertiaMoments[1] < 0.0005:
                # Case (1): linear molecule
                if verbosity >= 3:
                    print("\n Vibrational contribution for a linear molecule (3N-5)")
                if verbosity >= 3:
                    print("\n Frequencies that correspond to rotation and translation (lowest 5):")
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
                    print("\n Vibrational contribution for a nonlinear molecule (3N-6)")
                if verbosity >= 3:
                    print("\n Frequencies that correspond to rotation and translation (lowest 6):")
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
                print("\n Frequencies (before contamination with rotation and translation has been removed):")
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
                print("\n Scaling all frequencies by a factor of {}".format(scalefreq))
            for i in range(0, len(listOfFreqs)):
                if verbosity >= 1:
                    print(" Vibration {} Before: {:> 9.2f} cm⁻¹, After: {:> 9.2f} cm⁻¹".format(i + 1, listOfFreqs[i],
                                                                                               listOfFreqs[
                                                                                                   i] * scalefreq))
                listOfFreqs[i] = listOfFreqs[i] * scalefreq

        # Then, create a list that contains all vibrational temperatures (makes the summation of the
        # partition function simpler/more convenient)
        listOfVibTemps = []
        if verbosity >= 2:
            print("\n Vibrational temperatures:")
        ZPVE = 0.0
        for i in listOfFreqs:
            nu = i * Clight
            phi = hPlanck * nu / kBoltz
            # Still need to check for imaginary modes, since we might have just read the frequencies
            # instead of calculating them ourselves.
            if i > 0.0:
                molecule.ZPVEList.append((hPlanck * nu * 6.0221409E23) / 2.0 / 2625500.2)
                ZPVE += hPlanck * nu * 6.0221409E23
                listOfVibTemps.append(phi)
            if verbosity >= 2:
                print("{:> 9.2f} K".format(phi))
        molecule.ZPVE = (ZPVE / 2.0) / 2625500.2  # Converted from Joules to Hartree straight away
        # Now, assemble the partition function, vibrational entropy and the vibrational contribution to
        # the total thermal energy along with the zero point vibrational energy
        Qvib = 1.0
        Svib = 0.0
        Evib = 0.0
        for j, i in enumerate(listOfVibTemps):
            Qvib = Qvib * (math.exp(-1.0 * i / (2.0 * temp))) / (1 - math.exp(-1.0 * i / (temp)))
            SvibContribution = ((i / temp) / (math.exp(i / (temp)) - 1.0)) - math.log(1.0 - math.exp(-1.0 * i / (temp)))
            Svib += SvibContribution * RGas
            molecule.VibSList.append(SvibContribution * RGas)
            EvibContribution = i * (0.5 + (1 / (math.exp(i / temp) - 1.0)))
            Evib += EvibContribution * RGas
            molecule.thermVibList.append(
                math.fabs((EvibContribution * RGas) - (molecule.ZPVEList[j] * 2625500.2)) / 2625500.2)

    else:
        # If there are no vibrations, i.e. for a single atom, the results are straightforward:
        if verbosity >= 2:
            print("\n No vibrational analysis. This is either a single atom, or we didn't find any data.")
        Qvib = 1.0
        Svib = 0.0
        molecule.vibS = 0.0
        Evib = 0.0

    molecule.vibS = Svib
    molecule.thermRot = (Erot / 2625500.2)
    molecule.thermVib = (Evib / 2625500.2) - molecule.ZPVE
    Etherm = molecule.ZPVE + molecule.thermVib + molecule.thermRot + molecule.thermTrans
    Etot = molecule.Ee_QM + molecule.ZPVE + molecule.thermVib + molecule.thermRot + molecule.thermTrans
    molecule.kT = (kBoltz / 4.3597482E-18 * temp)  # kT contributon to the enthalpy
    Htot = Etot + molecule.kT  # Enthalpy: H = Etot + kB T in Hartree
    Stot = (molecule.elecS + molecule.transS + molecule.rotS + molecule.vibS)  # Total entropy
    molecule.negTS = (-1.0 * temp * Stot / 2625500.2)  # Total entropic contribution TS
    Gtot = Htot + molecule.negTS
    if molecule.frequencies == [] and molecule.H_QM == []:
        ProgramWarning(
            "\nCareful! We don't have any vibrational data for this compound!\nDon't use any of the derived thermodynamic values ΔE, ΔU, ΔH, ΔS or ΔG!\n")
    if verbosity >= 2:
        print("\n Vibrational partition function:                  {:> 12.5E}\n".format(Qvib))
        print(" Vibrational contribution to the entropy:         {:> 12.2f} J mol⁻¹ K⁻¹\n".format(molecule.vibS))
        print("  Contributions to the ZPVE:")
        for i, j in enumerate(molecule.ZPVEList):
            print("  Vibration {:4}:                        {:> 12.2f} J mol⁻¹".format(i + 1, j * 2625500.2))
        print(" Total zero-point vibrational energy (ZPVE):            {:> 12.2f} J mol⁻¹\n".format(
            molecule.ZPVE * 2625500.2))
        print("  Contributions to the vibrations:")
        for i, j in enumerate(molecule.thermVibList):
            print("  Vibration {:4}:                        {:> 12.2f} J mol⁻¹".format(i + 1, j * 2625500.2))
        print(" Total vibrational contribution to the energy:    {:> 12.2f} J mol⁻¹".format(
            molecule.thermVib * 2625500.2))

    if verbosity >= 1:
        print("\n Summary of the thermochemistry for: ", molecule.name)
        if verbosity >= 2:
            print(" The temperature is {:> 6.2f} K and the pressure is {:> 10.2f} Pa for this analysis".format(temp,
                                                                                                               press))
        print(" Electronic Energy (Ee):                                 {:> 12.6f} h".format(molecule.Ee_QM))
        print(" Electronic Energy (Ee + ZPVE):                          {:> 12.6f} h".format(
            molecule.Ee_QM + molecule.ZPVE))
        if verbosity >= 2:
            print("   Translational contribution:                           {:> 12.6f} h".format(molecule.thermTrans))
            print("   Rotational contribution:                              {:> 12.6f} h".format(molecule.thermRot))
            print("   Zero point vibrational contribution:                  {:> 12.6f} h".format(molecule.ZPVE))
            print("   Finite temperature vibrational contribution:          {:> 12.6f} h".format(molecule.thermVib))
            print("  Total thermal contributions to the energy (Etherm):    {:> 12.6f} h".format(Etherm))
        print(" Internal energy (U = Ee + Etherm):                      {:> 12.6f} h".format(Etot))
        if verbosity >= 2:
            print("  kT (=pV) contribution to the enthalpy:                 {:> 12.6f} h".format(molecule.kT))
        print(" Enthalpy (H = U + kT):                                  {:> 12.6f} h".format(Htot))
        if verbosity >= 3:
            print("   Electronic entropy:                               {:> 12.2f} J mol⁻¹ K⁻¹".format(molecule.elecS))
            print("   Translational entropy:                            {:> 12.2f} J mol⁻¹ K⁻¹".format(
                molecule.transS))
            print("   Rotational entropy:                               {:> 12.2f} J mol⁻¹ K⁻¹".format(
                molecule.rotS))
            print("   Vibrational entropy:                              {:> 12.2f} J mol⁻¹ K⁻¹".format(
                molecule.vibS))
        if verbosity >= 2:
            print("  Total entropy:                                     {:> 12.2f} J mol⁻¹ K⁻¹".format(Stot))
            print(" Total entropic contribution to the Gibbs energy (-TS)   {:> 12.6f} h\n".format(molecule.negTS))
        print(" Total Gibbs energy (G = H - TS)                         {:> 12.6f} h\n".format(Gtot))



################################################################################
#                                                                              #
# This is the part of the program where the thermochemical comparison is done  #
#                                                                              #
################################################################################

def thermochemicalComparison(reactants, products, temp=298.15, press=101325.0, analysis=False, verbosity=0):
    # Note that temperature and pressure are only used for printing, all the calculations
    # are done in thermochemicalAnalysis() and *not* here!

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
        print("#############################################################################")
        print("Thermodynamic changes during the reaction at:")
        print("{:.2f} K ({:+.2f} °C, {:+.2f} °F)".format(temp, temp - 273.15, (temp * (9 / 5)) - 459.67))
        print("{:.2f} Pa ({:.2f} atm, {:.2f} mmHg)".format(press, press / 101325.0, press * 0.007500617))
        print("-----------------------------------------------------------------------------")
        print("                                  Reactants        Products    Reaction")
        print("=============================================================================")
        print("                                         in hartrees (h)          in kJ mol⁻¹")
        print("-----------------------------------------------------------------------------")
        print("Electronic energy:             {:> 12.6f}      {:> 12.6f}      {:+10.2f}".format(
            rSum_Ee, pSum_Ee, diff_Ee * conv))
        print("Zero point vibrational energy: {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
            rSum_ZPVE, pSum_ZPVE, diff_ZPVE * conv))
        print("Energy at absolute zero:       {:> 12.6f}      {:> 12.6f}   ΔE={:+10.2f}\n".format(
            rSum_Ee + rSum_ZPVE, pSum_Ee + pSum_ZPVE, (diff_Ee + diff_ZPVE) * conv))
        print("Vibrational contribution:      {:> 12.6f}      {:> 12.6f}      {:+10.2f}".format(
            rSum_thermVib, pSum_thermVib, diff_thermVib * conv))
        print("Rotational contributions:      {:> 12.6f}      {:> 12.6f}      {:+10.2f}".format(
            rSum_thermRot, pSum_thermRot, diff_thermRot * conv))
        print("Translational contributions:   {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
            rSum_thermTrans, pSum_thermTrans, diff_thermTrans * conv))
        print("Internal energy (U):           {:> 12.6f}      {:> 12.6f}   ΔU={:+10.2f}\n".format(
            rSum_U, pSum_U, diff_U * conv))
        print("Volume work (pΔV = ΔnkT):      {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
            rSum_kT, pSum_kT, diff_kT * conv))
        print("Enthalpy (H = U + pΔV):        {:> 12.6f}      {:> 12.6f}   ΔH={:+10.2f}\n".format(
            rSum_H, pSum_H, diff_H * conv))
        print("=============================================================================")
        print("                                                in J mol⁻¹ K⁻¹")
        print("-----------------------------------------------------------------------------")
        print("Electronic entropy:          {:> 10.2f}        {:> 10.2f}          {:+10.2f}".format(
            rSum_elecS, pSum_elecS, diff_elecS))
        print("Translational entropy:       {:> 10.2f}        {:> 10.2f}          {:+10.2f}".format(
            rSum_transS, pSum_transS, diff_transS))
        print("Rotational entropy:          {:> 10.2f}        {:> 10.2f}          {:+10.2f}".format(
            rSum_rotS, pSum_rotS, diff_rotS))
        print("Vibrational entropy:         {:> 10.2f}        {:> 10.2f}          {:+10.2f}\n".format(
            rSum_vibS, pSum_vibS, diff_vibS))
        print("Total entropy (S):           {:> 10.2f}        {:> 10.2f}       ΔS={:+10.2f}\n".format(
            rSum_S, pSum_S, diff_S))
        print("=============================================================================")
        print("                                         in hartrees (h)          in kJ mol⁻¹")
        print("-----------------------------------------------------------------------------")
        print("Environment term (-TS):        {:> 12.6f}      {:> 12.6f}      {:+10.2f}\n".format(
            rSum_negTS, pSum_negTS, diff_negTS * conv))
        print("=============================================================================")
        print(
            "Gibbs energy (G = H - TS):     {:> 12.6f}      {:> 12.6f}   ΔG={:+10.2f}".format(rSum_G,
                                                                                              pSum_G,
                                                                                              diff_G * conv))
        print("#############################################################################")
        # Sanity check: if the sum of masses isn't the same, there's a problem
        if diffSumMass >= 0.01:
            ProgramWarning("There don't seem to be the same numbers/types of atoms")
    else:
        conv = 2625.5002  # Conversion factor for hartree to kJ mol⁻¹
        print("#############################################################################")
        if len(reactants) == 1:
            print("Thermodynamics for the input compounds at:")
        else:
            print("Combined thermodynamics for all {} input compounds at:".format(len(reactants)))
        print("{:.2f} K ({:+.2f} °C, {:+.2f} °F)".format(temp, temp - 273.15, (temp * (9 / 5)) - 459.67))
        print("{:.2f} Pa ({:.2f} atm, {:.2f} mmHg)".format(press, press / 101325.0, press * 0.007500617))
        print("=============================================================================")
        print("                                         in hartrees (h)")
        print("-----------------------------------------------------------------------------")
        print("Electronic energy:                      {:> 12.6f}".format(
            rSum_Ee))
        if len(reactants) == 1 and analysis == True:
            print("\n Contributions to the ZPVE:")
            for i, j in enumerate(reactants[0].ZPVEList):
                print(" Vibration {:4} ({:6.0f} cm⁻¹):  {:> 12.6f}".format(i+1,reactants[0].frequencies[i], j))
            print("Total zero point vibrational energy:    {:> 12.6f}\n".format(rSum_ZPVE))
        else:
            print("Zero point vibrational energy:          {:> 12.6f}\n".format(rSum_ZPVE))
        print("Energy at absolute zero:                {:> 12.6f}\n".format(
            rSum_Ee + rSum_ZPVE))

        print("Translational contributions:            {:> 12.6f}".format(
            rSum_thermTrans))
        if len(reactants) == 1:
            print("Rotational contributions (σ = {:2}):      {:> 12.6f}".format(reactants[0].sigmaRot, rSum_thermRot))
        else:
            print("Rotational contributions:               {:> 12.6f}".format(rSum_thermRot))
        if len(reactants) == 1 and analysis == True:
            print("\n Vibrational contributions:")
            for i, j in enumerate(reactants[0].thermVibList):
                print(" Vibration {:4} ({:6.0f} cm⁻¹):  {:> 12.6f}".format(i+1,reactants[0].frequencies[i], j))
        if len(reactants) == 1:
            print("Total vibrational contribution:         {:> 12.6f}\n".format(
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
        print("=============================================================================")
        print("                                        in J mol⁻¹ K⁻¹")
        print("-----------------------------------------------------------------------------")
        print("Electronic entropy:                   {:> 10.2f}".format(
            rSum_elecS))
        print("Translational entropy:                {:> 10.2f}".format(
            rSum_transS))
        print("Rotational entropy:                   {:> 10.2f}".format(
            rSum_rotS))
        if len(reactants) == 1 and analysis == True:
            print("\n Vibrational contributions to the entropy:")
            for i, j in enumerate(reactants[0].VibSList):
                print(" Vibration {:4} ({:6.0f} cm⁻¹):  {:> 10.2f}".format(i+1,reactants[0].frequencies[i], j))
            print("Total vibrational entropy:            {:> 10.2f}\n".format(rSum_vibS))
        else:
            print("Vibrational entropy:                  {:> 10.2f}\n".format(rSum_vibS))
        print("Total entropy (S):                    {:> 10.2f}\n".format(
            rSum_S))
        print("=============================================================================")
        print("                                         in hartrees (h)")
        print("-----------------------------------------------------------------------------")
        print("Environment term (-TS):                 {:> 12.6f}\n".format(
            rSum_negTS))
        print("=============================================================================")
        print(
            "Gibbs energy (G = H - TS):              {:> 12.6f}".format(rSum_G))
        print("#############################################################################")



# class Molecule:
#     """A molecule with a name, charge and a list of atoms"""
#
#     def __init__(self, name, charge=0, multiplicity=1):
#         """ (Molecule, str, int) -> NoneType
#
#     Create a Molecule named name with charge charge and no atoms
#     (int) Multiplicity mult is automatically set to the lowest
#     possible value (1 or 2) and the lists of bonds, angles and
#     dihedrals are empty.
#     """
#
#         self.name = name
#         self.charge = charge
#         self.mult = multiplicity
#         self.atoms = []  # Initially an empty list
#         self.bonds = []  # Initially an empty list
#         self.angles = []  # Initially an empty list
#         self.dihedrals = []  # Initially an empty list
#         self.H_QM = np.zeros((3, 3))  # Force constants, Array size arbitrary, just a placeholder for type
#         self.H_mw = np.zeros((3, 3))  # Mass weighted force constants, Array size arbitrary, just a placeholder for type
#         self.frequencies = []  # List of vibrational frequencies in cm ⁻¹
#         self.sigmaRot = 1  # Rotational symmetry number
#         self.Ee_QM = 0.0  # electronic energy in Hartree
#         self.ZPVE = 0.0  # Zero Point Vibrational Energy in Hartree
#         self.ZPVEList = []  # List of individual vibrations' contributions to the ZPVE in Hartree
#         self.thermVib = 0.0  # Finite temperature vibrational contribution in Hartree
#         self.thermVibList = []  # List of contributions from individual vibrations in Hartree
#         self.thermRot = 0.0  # Thermal contribution from rotations in Hartree
#         self.thermTrans = 0.0  # Thermal contribution from translation in Hartree
#         self.kT = 0.0  # Thermal contribution (from pV = kT) to Enthalpy in Hartree
#         self.transS = 0.0  # Translational entropy in J mol⁻¹ K⁻¹
#         self.elecS = 0.0  # Electronic entropy in J mol⁻¹ K⁻¹
#         self.rotS = 0.0  # Rotational entropy in J mol⁻¹ K⁻¹
#         self.vibS = 0.0  # Vibrational entropy in J mol⁻¹ K⁻¹
#         self.VibSList = []  # List of individual vibrations' contributions to the entropy in J mol⁻¹ K⁻¹
#         self.negTS = 0.0  # Thermal contribution (from -TS) to Gibbs energy in Hartree


################################################################################
#                                                                              #
# This is the part of the program where the command line arguments are defined #
#                                                                              #
################################################################################

parser = argparse.ArgumentParser(
    description="WellFAReFEDA: Wellington Fast Assessment of Reactions - Free Energy Decomposition Analysis",
    epilog="recognised filetypes: gaussian, orca, turbomole")
# parser.add_argument("-P", "--numproc", type=int, help="number of processes for parallel execution",
#                     default="2")
parser.add_argument("--temperature", type=float, help="Temperature (in Kelvin) for the thermochemical analysis",
                    default="298.15")
parser.add_argument("--pressure", type=float, help="Pressure (in Pa) for the thermochemical analysis",
                    default="101325.0")
parser.add_argument("--scalefreq", type=float, help="Scale harmonic frequencies by a constant factor",
                    default="1.0")
parser.add_argument("-a", "--analysis", help="For single compounds, analyse contributions from each vibrational mode",
                    action='store_true')
parser.add_argument("-r", "--reactants", metavar='file', nargs='+',
                    help="input file(s) with qc data of the reactant(s)",
                    required=True)
parser.add_argument("-p", "--products", metavar='file', nargs='+', help="input file(s) with qc data of the product(s)")
parser.add_argument("-v", "--verbosity", help="increase output verbosity", type=int, choices=[0, 1, 2, 3], default=0)

args = parser.parse_args()

###############################################################################
#                                                                             #
# The main part of the program starts here                                    #
#                                                                             #
###############################################################################

# Print GPL v3 statement and program header
ProgramHeader()

ListOfReactants = []
if type(args.reactants) is str:
    ListOfReactants.append(Molecule(args.reactants, 0))
    extractMolecularData(args.reactants, ListOfReactants[-1], verbosity=args.verbosity, bondcutoff=0.55)
    thermochemicalAnalysis(ListOfReactants[-1], temp=args.temperature, press=args.pressure, scalefreq=args.scalefreq,
                           verbosity=args.verbosity)
else:
    for i in args.reactants:
        ListOfReactants.append(Molecule(i, 0))
        extractMolecularData(i, ListOfReactants[-1], verbosity=args.verbosity, bondcutoff=0.55)
        thermochemicalAnalysis(ListOfReactants[-1], temp=args.temperature, press=args.pressure,
                               scalefreq=args.scalefreq,
                               verbosity=args.verbosity)

ListOfProducts = []
if args.products is not None and type(args.products) is str:
    ListOfProducts.append(Molecule(args.products, 0))
    extractMolecularData(args.products, ListOfProducts[-1], verbosity=args.verbosity, bondcutoff=0.55)
    thermochemicalAnalysis(ListOfProducts[-1], temp=args.temperature, press=args.pressure, scalefreq=args.scalefreq,
                           verbosity=args.verbosity)
elif args.products is not None:
    for i in args.products:
        ListOfProducts.append(Molecule(i, 0))
        extractMolecularData(i, ListOfProducts[-1], verbosity=args.verbosity, bondcutoff=0.55)
        thermochemicalAnalysis(ListOfProducts[-1], temp=args.temperature, press=args.pressure, scalefreq=args.scalefreq,
                               verbosity=args.verbosity)
else:
    if args.verbosity >= 1:
        print("No products read")

thermochemicalComparison(ListOfReactants, ListOfProducts, temp=args.temperature, press=args.pressure,
                         analysis=args.analysis, verbosity=args.verbosity)

ProgramFooter()
