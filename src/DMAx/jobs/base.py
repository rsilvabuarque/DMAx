"""Definition of base LAMMPS job maker."""

# import Structure
from typing import Callable
from dataclasses import dataclass, field

from jobflow import Maker, Response, job

from atomate2.vasp.sets.base import LammpsInputGenerator


def lammps_job(method: Callable):
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
    return job(method, ?)

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
    more?
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
    def make(self, structure: Structure, prev_lammps_dir: str | Path | None = None):
        """
        Run a LAMMPS calculation.

        Parameters
        ----------
        structure : Structure
            A pymatgen structure object.
        prev_vasp_dir : str or Path or None
            A previous LAMMPS calculation directory to copy output files from.
        """
        # copy previous inputs
        # write vasp input files
        # write any additional data
        # run vasp
        # parse vasp outputs
        # decide whether child jobs should proceed 
        # gzip folder
        return Response(
            ...
        )

