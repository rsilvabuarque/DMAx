"""Definition of base LAMMPS job maker."""

from typing import Callable
from dataclasses import dataclass, field

import os
import subprocess

from jobflow import Maker, Response, job

from pymatgen.core.structure import Structure

from DMAx.sets.base import LammpsInputGenerator


#def lammps_job(method: Callable):
    """
    Decorate the ``make`` method of LAMMPS job makers.

    This is a thin wrapper around :obj:`~jobflow.core.job.Job` that configures common
    settings for all LAMMPS jobs. [what does it ensure: "For example, it ensures that large data objects
    (band structures, density of states, LOCPOT, CHGCAR, etc) are all stored in the
    atomate2 data store."] [does it need to configure a 'output schema': "It also configures the output schema to be a VASP
    :obj:`.TaskDoc`."]

    Any makers that return LAMMPS jobs (not flows) should decorate the ``make`` method
    with @vasp_job. For example:

    .. code-block:: python

        class MyLammpsMaker(BaseLammpsMaker):
            @lammps_job
            def make(structure):
                # code to run LAMMPS job.
                pass

    Parameters
    ----------
    method : callable
        A BaseLammpsMaker.make method. This should not be specified directly and is
        implied by the decorator.

    Returns
    -------
    callable
        A decorated version of the make function that will generate LAMMPS jobs.
    """
    # for ? we can put configre common settings for all LAMMPS jobs since, when it is tagged on make, it is guranteed to be submitted in the supercomputer
    # priority: ADVANCED
    # ex for vasp they did (found in atomate2/src/atomate2/vasp/jobs/base.py: 
        # 
        # return job(method, data=_DATA_OBJECTS, output_schema=TaskDoc)
        # 
        # where data and output_schema are things specific to all Lammps workflows
            # data is defined in their script where BaseVaspMaker is
            # output_schema is what they use for whatever that is 
    #TODO: add @lammps_job to any make() method that is or inherits BaseLammpsMaker
#    return job(method, ?)

@dataclass
class BaseLammpsMaker(Maker):
    """
    Base LAMMPS job maker.

    Parameters
    ----------
    name : str
        The job name.
    input_set_generator : .LAMMPSInputGenerator
        A generator used to make the input set.
    """

    name: str = "base lammps job"
    input_set_generator: LAMMPSInputGenerator = field(default_factory=LAMMPSInputGenerator)
    # write_input_set_kwargs: dict = field(default_factory=dict)
    # copy_lammps_kwargs: dict = field(default_factory=dict)
    # run_lammps_kwargs: dict = field(default_factory=dict)
    # task_document_kwargs: dict = field(default_factory=dict)
    # stop_children_kwargs: dict = field(default_factory=dict)
    # write_additional_data: dict = field(default_factory=dict)
    
    # @lammps_job
    def make(self, structure_file: str, calc_dir: str):
        """
        Run a LAMMPS calculation.

        Parameters
        ----------
        structure_file : BGF|MSI|MOL2
        The input structure for the calculation. Can also take the formats below with listed limitations.
        PDB|PQR|XYZ - connections will be generated automatically (if no CONECT tables).
            NOTE: Forcefield types are assumed to be the same as the atom names
        CHARMM_PSF - Also need to specify CHARMM CRD file (in quotes) or have xxx.crd file in same folder
        AMBER_PRMTOP - Also need to specify AMBER CRD|RST file (in quotes) or have xxx.crd|xxx.rst7 file in same folder
        GROMACS_GRO - Also need to specify TOP|XPLOR file (in quotes) xxx.top|xxx.xplor file in the same folder
        calc_dir: str
        Path to where input files should be written.
        """
        # write lammps input files
        
        # run lammps
        
        # parse lammps outputs
        return Response()
    
    def submit_slurm_files(dir_path, avoid_containing_log=True):
    """
    Recursively submit all SLURM files found in a directory and its subdirectories,
    but do so inside the directories where the SLURM files are found.

    Arguments:
    dir_path -- A string representing the path of the directory to search for SLURM files.
    avoid_containing_log -- A boolean flag to avoid submitting files in directories containing a .log file.
    """

    # Iterate over each item in the given directory
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)

        # If the item is a directory, recursively call this function on it
        if os.path.isdir(item_path):
            submit_slurm_files(item_path, avoid_containing_log)

        # If the item is a file with a ".slurm" extension, submit it to SLURM
        elif os.path.isfile(item_path) and item.endswith(".slurm"):
            # Check if avoid_containing_log flag is True and if a ".log" file exists in the same directory
            if avoid_containing_log and any(f.endswith(".log") for f in os.listdir(dir_path)):
                print(f"Skipping {item_path} as it is in a directory containing a .log file.")
            else:
                # Use the subprocess module to run the sbatch command on the SLURM file
                # But do so inside the directory where the SLURM file is found
                slurm_dir = os.path.dirname(item_path)
                subprocess.run(["sbatch", os.path.basename(item_path)], cwd=slurm_dir)
