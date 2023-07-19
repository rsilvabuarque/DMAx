"""Jobs used to generate the master curve."""

from jobflow import Flow, job, Maker
from dataclasses import dataclass

from LAMMPSInputGenerator import DMAMasterCurveInputGenerator, LammpsInputGenerator, submit_slurm_files
from LAMMPSDMAParser import DMAMasterCurveDataParser, master_curve_parser, master_curve_plotter

@dataclass
class DmaMasterCurveMaker(Maker):
    """Maker to generate the master curve of the structure"""
    # TODO: edit account perhaps to be default csd626? make a logic checking if they have permission or sum
    # TODO: add this attribute - input_set_generator: LammpsInputGenerator = field(default_factory=DMAConvAnalysisInputGenerator)
    
    name: str = "convergence analysis"
    polymer_glass_temperature: float = 399
    standard_numcycles: int = 4
    standard_strain: float = 0.11
    num_frequencies: int = 20
    num_temperatures: int = 10
    temperature_interval: int = 10
    custom_frequency_range: list = None
    noise_filter_run: bool = True
    forcefield: str = "UFF"
    suffix: str = None
    account: str = None
    nodes: int = 1
    ntasks_per_node: int = 16
    time: str ='24:0:0'
    temperature: float = 300
    pressure: float = 1
    

    # TODO: make input_generator: DMAMasterCurveInputGenerator but as a better looking parameter (see examples)
    
    @job 
    def generate_curve_inputs(generator_object):
        """
        Generate master curve inputs for supercomputer

        Parameters
        ___
        generator_object : DMAMasterCurveInputGenerator object
        """
        MasterCurveObj.MasterCurveCLI(self.polymer_glass_temperature, self.standard_numcycles, self.standard_strain, num_frequencies=self.num_frequencies, num_temperatures=self.num_temperatures, temperature_interval=self.temperature_interval, custom_frequency_range=elf.custom_frequency_range, noise_filter_run=self.noise_filter_run, **kwargs)

    @job
    def run_curve_simulations(slurm_path):
        """
        Submit generater master curve inputs to supercomputer

        Parameters
        ___
        slurm_path: str
        A string representing the path of the directory to search for SLURM files.

        """ 
        submit_slurm_files(slurm_path)

    @job
    def parse_curve_outputs(parser_object):
        """
        Parse master curve onto graph

        Returns
        ----


        Parameters
        ___
        slurm_path: str
        A string representing the path of the directory to search for SLURM files.

        """ 
        *parser_data, = parser_object.master_curve_parser()
        parser_object.master_curve_plotter(parser_data)
    
    
    def make(self, structure_file, calc_dir, polymer_glass_temperature, standard_numcycles, standard_strain, **kwargs):
    
    stressdir = kwargs.get("stressdir", None)
    generator_obj = DMAMasterCurveInputGenerator(structure_file, calc_dir, forcefield=self.forcefield, suffix=self.suffix, account=self.account, nodes=self.nodes, ntasks_per_node=self.ntasks_per_node, time=self.time, temperature=self.temperature, pressure=self.pressure)
    parser_obj = DMAMasterCurveDataParser(generator_obj.calc_dir)
    
    jobs = []
    
    job1 = generate_curve_inputs(generator_obj)
    job2 = run_curve_simulations(generator_obj.calc_dir)
    job3 = parse_curve_outputs(parser_obj)
    
    jobs += [job1, job2, job3]
    
    return Flow(jobs=jobs, name=self.name)
    
    