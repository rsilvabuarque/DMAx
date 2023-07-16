"""Module defining core LAMMPS input set generators."""

from DMAx.sets.base import LammpsInputGenerator
        
class TwoptInputGenerator(LammpsInputGenerator):
    def create2ptInput(self, temperature=300, pressure=1, runtime1=1000000, runtime2=200000, timestep_unit_fs=1, master_inputs=None, **kwargs):
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        os.chdir(calc_dir)
        if not master_inputs:
            self.createLammpsInput(**{"calc_dir": calc_dir, "temperature": temperature})
            self.slurm_modifier(**{"temperature": temperature, "pressure": pressure})
            subprocess.call("rm *singlepoint", shell=True)
            in_line = int(subprocess.check_output("grep -n 'NVT production dynamics' in.* | cut -d : -f 1", shell=True))
            subprocess.call("head -n {} in.* > in_head".format(in_line), shell=True)
            subprocess.call("cat in_head $MASTER_FILES/in.master_2pt > in.{}".format(self.suffix), shell=True)
            subprocess.call("rm *in_head*", shell=True)
        else:
            subprocess.call('cp {} in.{}'.format(master_inputs["in"], self.suffix), shell=True)
            subprocess.call('cp {} data.{}'.format(master_inputs["data"], self.suffix), shell=True)
            # For the next command line to work, the lines (variable             input string in.*) and (variable             sname string *) must have the * replaced by the word 'master'
            subprocess.call("sed -i 's/master/{}/g' in.*".format(self.suffix), shell=True)
        # Select the runtimes for the NVT Production Dynamics and for the following 2pt NVT
        subprocess.call("sed -i 's/runtime1/{}/g' in.*".format(runtime1), shell=True)
        subprocess.call("sed -i 's/runtime2/{}/g' in.*".format(runtime2), shell=True)
        # Increasing the maximum number of neighbors since it seems to be a problem for some supercells
        subprocess.call("sed -i 's/check yes/check yes one 3000/g' in.*", shell=True)

    def temp_analysis_2pt(self, temperature_range, **kwargs):
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        os.chdir(calc_dir)
        if not "temperatures" in os.listdir("."):
            os.mkdir("temperatures")
        os.chdir("temperatures")
        for temp in temperature_range:
            if not "{}K".format(temp) in os.listdir("."):
                os.mkdir("{}K".format(temp))
            os.chdir("{}K".format(temp))
            prefix = "{}_2pt_{}K".format(self.suffix, temp)
            print("Creating inputs for " + prefix)
            self.create2ptInput(freq, temperature=temp, **{"calc_dir": "."})
            os.chdir("../")

class DMAInputGenerator(LammpsInputGenerator):
    def createLammpsInputDMA(self, frequency, temperature=300, pressure=1, timestep_unit_fs=1, oscillation_amplitude_percentage=0.04, numcycles=10, datapoints_per_cycle=500, tilt_analysis=False, nve=False, stressdir=None, master_inputs=None, **kwargs):
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        stressdir = kwargs.get("stressdir", None)
        os.chdir(calc_dir)
        timestep = timestep_unit_fs
        period = (1 / frequency) / (timestep * 1e-15)
        if not master_inputs:
            self.createLammpsInput(**{"calc_dir": calc_dir, "temperature": temperature})
            self.slurm_modifier(**{"temperature": temperature, "pressure": pressure})
            subprocess.call("rm *singlepoint", shell=True)
            in_line = int(subprocess.check_output("grep -n 'NVT production dynamics' in.* | cut -d : -f 1", shell=True))
            subprocess.call("head -n {} in.* > in_head".format(in_line), shell=True)
            subprocess.call("cat in_head $MASTER_FILES/in.master_mech_loss > in.{}".format(self.suffix), shell=True)
            subprocess.call("rm *in_head*", shell=True)
        else:
            subprocess.call('cp {} in.{}'.format(master_inputs["in"], self.suffix), shell=True)
            subprocess.call('cp {} data.{}'.format(master_inputs["data"], self.suffix), shell=True)
            # For the next command line to work, the lines (variable             input string in.*) and (variable             sname string *) must have the * replaced by the word 'master'
            subprocess.call("sed -i 's/master/{}/g' in.*".format(self.suffix), shell=True)
            # Finding magnitude and direction of longest lattice vector and determining the oscillation amplitude amd direction
        grepCRYSTX = subprocess.check_output("grep 'CRYSTX' {}".format(self.structure_file), shell=True).decode("utf-8").split()
        latparam_dict = {"x": float(grepCRYSTX[1]), "y": float(grepCRYSTX[2]), "z": float(grepCRYSTX[3])}
        biggestlatparam = max(latparam_dict.values())
        biggestlatdir = max(latparam_dict, key=latparam_dict.get)
        dirlist = list(latparam_dict.keys())
        if stressdir != None:
            changing_var1 = stressdir
            changing_var2 = stressdir
        else:
            changing_var1 = biggestlatdir
            changing_var2 = dirlist[0] + dirlist[1]
        if tilt_analysis == False:
            dirlist.remove(changing_var1)
            subprocess.call("sed -i -e 's/stressdir/{}/g ; s/dir1/{}/g ; s/dir2/{}/g' in.*".format(changing_var1, dirlist[0], dirlist[1]), shell=True)
        else:
            subprocess.call("sed -i -e 's/stressdir wiggle osc_amp freq_input dir1 volume dir2 volume/{} wiggle osc_amp freq_input/g' in.*".format(changing_var2), shell=True)
        if nve:
            subprocess.call("sed -i -e 's/fix                  2 all nvt temp ${rtemp} ${rtemp} 100.0 tloop 10 ploop 10/fix                  2 all nve/g' in.*", shell=True)
        subprocess.call("sed -i 's/osc_amp/{}/g' in.*".format(biggestlatparam * float(oscillation_amplitude_percentage)), shell=True)
        subprocess.call("sed -i 's/freq_input/{}/g' in.*".format(round(period)), shell=True)
        dump_frequency = period / datapoints_per_cycle
        tdamp = 100
        # Adjusting the calculation timestep to adapt for high frequencies
        runtime_multiplier = 1
        while dump_frequency < 1:
            timestep /= 10
            dump_frequency *= 10
            runtime_multiplier *= 10
            tdamp /= 2
        runtime = round(period * runtime_multiplier * int(numcycles))
        dump_frequency = round(dump_frequency)
        subprocess.call("sed -i 's/prod_timestep/{}/g' in.*".format(timestep), shell=True)
        # Determining the runtime to fit the desired number of cycles
        subprocess.call("sed -i 's/runtime/{}/g' in.*".format(runtime), shell=True)
        # Determining the dump frequency to fit N points per cycle
        subprocess.call("sed -i 's/dump_freq/{}/g' in.*".format(dump_frequency), shell=True)
        # Increasing the maximum number of neighbors since it seems to be a problem for some supercells
        subprocess.call("sed -i 's/check yes/check yes one 3000/g' in.*", shell=True)
        # Reducing the temperature update time to avoid issues at high frequencies
        subprocess.call("sed -i 's/tdamp/{}/g' in.*".format(tdamp), shell=True)

    def noise_filter_CLI(self, **kwargs):
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        os.chdir(calc_dir)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        if not "noise_filter_run" in os.listdir(calc_dir):
            os.mkdir("noise_filter_run")
        os.chdir("noise_filter_run")
        self.createLammpsInput(**{"calc_dir": calc_dir, "temperature": temperature})
        subprocess.call("sed -i 's/run                  5000000 # run for 15 ns/run                  500000 # run for 0.5 ns/g' in.*", shell=True)
        self.slurm_modifier(**{"temperature": temperature, "pressure": pressure})
        subprocess.call("rm *singlepoint", shell=True)
        os.chdir("../")

    def error_analysis_DMA(self, frequencies=[25e9], num_error_calcs=5, noise_filter_run=False, **kwargs):
        oscillation_amplitude_percentage = kwargs.get('oscillation_amplitude_percentage', 0.04)
        numcycles = kwargs.get('numcycles', 10)
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        stressdir = kwargs.get("stressdir", None)
        os.chdir(calc_dir)
        if not "frequencies" in os.listdir("."):
            os.mkdir("frequencies")
        os.chdir("frequencies")
        if noise_filter_run:
            self.noise_filter_CLI(**{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        for freq in frequencies:
            for i in range(num_error_calcs):
                prefix = "{}_mech_loss_{}K_{}_{}".format(self.suffix, temperature, frequency_dirname_parser(freq), i)
                print("Creating inputs for " + prefix)
                if not "{}_{}".format(frequency_dirname_parser(freq), i) in os.listdir("."):
                    os.mkdir("{}_{}".format(frequency_dirname_parser(freq), i))
                os.chdir("{}_{}".format(frequency_dirname_parser(freq), i))
                self.createLammpsInputDMA(freq, oscillation_amplitude_percentage=oscillation_amplitude_percentage, numcycles=numcycles, temperature=temperature, pressure=pressure, **{"calc_dir": ".", "stressdir": stressdir})
                os.chdir("../")
                
class DMAConvAnalysisInputGenerator(DMAInputGenerator):
    """
    LAMMPS input generator for the convergence analysis of parameters for Dynamic Mechanical Analysis.
    """
    def numCyclesConvAnalysisCLI(self, std_freq_Hz=25e9, custom_numcycles=None, standard_numcycles=5, bulk_modulus_GPa=None, strain_direction=None, noise_filter_run=True, **kwargs):
        """
        Method for generating the LAMMPS input files for a runtime convergence for Dynamic Mechanical Analysis.
        Parameters
        ----------
        std_freq : float
            Standard frequency (in Hz) to run all simulations. Default to 25GHz.
        custom_numcycles: int
            Custom number of cycles to test for convergence.
        bulk_modulus_GPa: float
            Estimated Bulk Modulus of the observed structure. If provided, will be used to optimize the selected strain range to better fit the material's crystallinity.
        strain_direction: str
            Strain direction of the simulation. If not provided, will default to the longest axis of the unit cell.
        noise_calc: bool
            Whether to add a calculation without any strain to remove noise in post-processing.
        """
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        os.chdir(calc_dir)
        if custom_numcycles:
            numcycles = custom_numcycles
        else:
            numcycles = standard_numcycles
        if bulk_modulus_GPa:
            if bulk_modulus_GPa < 100:
                numcycles = 10
            else:
                numcycles = 5
        if not "frequencies" in os.listdir("."):
            os.mkdir("frequencies")
        os.chdir("frequencies")
        if noise_filter_run:
            self.noise_filter_CLI(std_freq_Hz, numcycles, **{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        freq_dirname = frequency_dirname_parser(std_freq_Hz)
        os.mkdir(freq_dirname)
        os.chdir(freq_dirname)
        os.mkdir("{}cycles".format(numcycles))
        os.chdir("{}cycles".format(numcycles))
        self.createLammpsInputDMA(std_freq_Hz, numcycles=numcycles, temperature=temperature, **{"calc_dir": ".", "stressdir": strain_direction})

    def strainSizeConvAnalysisCLI(self, std_freq_Hz=25e9, custom_strain_range=None, bulk_modulus_GPa=None, strain_direction=None, standard_numcycles=5, standard_numstrains=10, noise_filter_run=True, crystalline=False, **kwargs):
        """
        Method for generating the LAMMPS input files for a strain size convergence for Dynamic Mechanical Analysis. The strain percentages are proportional to the size of the unit cell in the direction of the strain
        Parameters
        ----------
        std_freq : float
            Standard frequency (in Hz) to run all simulations. Default to 25GHz.
        custom_strain_range: list
            Custom list of strains to test for convergence. Values must be in percentage (Ex.: 1%, 2% -> strain_range=[1, 2]). If not provided, will default to standard selection.
        bulk_modulus_GPa: float
            Estimated Bulk Modulus of the observed structure. If provided, will be used to optimize the selected strain range to better fit the material's crystallinity.
        strain_direction: str
            Strain direction of the simulation. If not provided, will default to the longest axis of the unit cell.
        """
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        os.chdir(self.calc_dir)
        if not "frequencies" in os.listdir("."):
            os.mkdir("frequencies")
        os.chdir("frequencies")
        if noise_filter_run:
            self.noise_filter_CLI(std_freq_Hz, standard_numcycles, **{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        freq_dirname = frequency_dirname_parser(std_freq_Hz)
        os.mkdir(freq_dirname)
        os.chdir(freq_dirname)
        if crystalline:
            strain_range_pc = np.array([round(i, 3) for i in np.logspace(log10(0.001), log10(10), num=standard_numstrains, endpoint=True, base=10)])
        else:
            strain_range_pc = np.array([round(i, 3) for i in np.logspace(log10(0.1), log10(100), num=standard_numstrains, endpoint=True, base=10)])
        if custom_strain_range:
            strain_range_pc = custom_strain_range
        if bulk_modulus_GPa:
            if bulk_modulus_GPa < 100:
                strain_range_pc = np.array([10, 15, 20, 25, 30, 35])
            else:
                strain_range_pc = np.array([0.01, 0.1, 1, 2, 5, 10])
        for strain in strain_range_pc:
            os.mkdir("{}pc_strain".format(strain))
            os.chdir("{}pc_strain".format(strain))
            self.createLammpsInputDMA(std_freq_Hz, numcycles=standard_numcycles, oscillation_amplitude_percentage=strain/100, temperature=temperature, **{"calc_dir": ".", "stressdir": strain_direction})
            os.chdir("../")

class DMAMasterCurveInputGenerator(DMAInputGenerator):
    def MasterCurveCLI(self, polymer_glass_temperature, standard_numcycles, standard_strain_pc, num_frequencies=20, num_temperatures=10, temperature_interval=10, custom_frequency_range=None, custom_temperature_range=None, noise_filter_run=True, **kwargs):
        """
        Method for generating the LAMMPS input files for generating a mechanical loss master curve.
        ----------
        polymer_glass_temperature : float
            Glass temperature transition of the studied polymer.
        standard_numcycles: int
            Standard number of periods to run all simulations.
        standard_strain_pc: float
            Standard strain to run all simulations. Values must be in percentage (Ex.: 1%, 2% -> strain_range=[1, 2]).
        strain_direction: str
            Strain direction of the simulation. If not provided, will default to the longest axis of the unit cell.
        """
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        stressdir = kwargs.get("stressdir", None)
        os.chdir(calc_dir)
        if not "temperatures" in os.listdir("."):
            os.mkdir("temperatures")
        os.chdir("temperatures")
        if custom_frequency_range:
            frequency_range = custom_freq_range
        else:
            frequency_range = [round(i, 3) for i in np.logspace(log10(10e9), log10(500e9), num=num_frequencies, endpoint=True, base=10)]
        if custom_temperature_range:
            temperature_range = custom_temperature_range
        else:
            temperature_range = np.arange(num_temperatures) * temperature_interval
            temperature_range -= (num_temperatures // 2) * temperature_interval
            temperature_range += polymer_glass_temperature
        for temp in temperature_range:
            if not "{}K".format(temp) in os.listdir("."):
                os.mkdir("{}K".format(temp))
            os.chdir("{}K".format(temp))
            if noise_filter_run:
                self.noise_filter_CLI(min(frequency_range), standard_numcycles, **{ "calc_dir": ".", "temperature": temp, "pressure": pressure})
            for freq in frequency_range:
                prefix = "{}_mech_loss_{}K_{}".format(self.suffix, temp, frequency_dirname_parser(freq))
                print("Creating inputs for " + prefix)
                if not "{}".format(frequency_dirname_parser(freq)) in os.listdir("."):
                    os.mkdir("{}".format(frequency_dirname_parser(freq)))
                os.chdir("{}".format(frequency_dirname_parser(freq)))
                self.createLammpsInputDMA(freq, oscillation_amplitude_percentage=standard_strain_pc/100, numcycles=standard_numcycles, temperature=temp, **{"calc_dir": ".", "stressdir": stressdir})
                os.chdir("../")
            os.chdir("../")
