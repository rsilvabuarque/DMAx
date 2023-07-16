"""Settings for DMAx."""

from pathlib import Path

from pydantic import BaseSettings, Field, root_validator

_DEFAULT_CONFIG_FILE_PATH = "~/.dmax.yaml"
CLI_PATH = ""
SLR_MASTER_PATH = ""
MECH_LOSS_MASTER_PATH = ""
NOISE_FILTER_MASTER_PATH = ""


#TODO: for createLammpsInput.pl, resolve which supercomputer they need access to OR get Prof Pascals permissions changed
#TODO: for slurm_script.master, in.master_mech_loss, and in.master_noise_filter_run, check Ricardo's preference on whether these can be published to DMAx

# perlmutter paths
/global/homes/t/tpascal/scripts/createLammpsInput.pl
/global/homes/b/bfune/DMAx/src/DMAx/master_files/slurm_script.master
/global/homes/b/bfune/DMAx/src/DMAx/master_files/in.master_mech_loss
/global/homes/b/bfune/DMAx/src/DMAx/master_files/in.master_noise_filter_run

#expanse paths
/expanse/lustre/projects/csd626/tpascal/scripts/createLammpsInput.pl
/expanse/lustre/projects/csd626/ricardosb/master_files/slurm_script.master
/expanse/lustre/projects/csd626/ricardosb/master_files/in.master_mech_loss
/expanse/lustre/projects/csd626/ricardosb/master_files/in.master_noise_filter_run


__all__ = ["DMAxSetting"]

class DMAxSettings(BaseSettings):
    """
    Setting for DMAx.
    
    The default way to modify these is to modify ~/.dmax.yaml. Alternatively,
    the environment variable DMAX_CONFIG_FILE can be set to point to a yaml file
    with DMAx settings.

    Lastly, the variables can be modified directly through environment variables by
    using the "DMAX" prefix. E.g. DMAX_SCRATCH_DIR = path/to/scratch.
    """
    
    CONFIG_FILE: str = Field(
        _DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from."
    )
    
    CALC_DIR: str = Field(
        "calc_dir", description="Path to where input files should be written."
    )

    # LAMMPS specific settings
    LAMMPS_CMD str = Field(
        "lammps_std", description="Command to run standard version of LAMMPS."
        
    class Config:
        """Pydantic config settings."""
        
        env_prefix = "dmax_"
    
    @root_validator(pre=True)
    def load_default_settings(cls, values):
        """
        Load settings from file or environment variables.

        Loads settings from a root file if available and uses that as defaults in
        place of built-in defaults.

        This allows setting of the config file path through environment variables.
        CHATGPT: Overall, the load_default_settings method allows loading default settings from a configuration file, overriding any existing attribute values, and providing a mechanism to set the configuration file path through environment variables.
        """
        from monty.serialization import loadfn

        config_file_path: str = values.get("CONFIG_FILE", _DEFAULT_CONFIG_FILE_PATH)

        new_values = {}
        if Path(config_file_path).expanduser().exists():
            new_values.update(loadfn(Path(config_file_path).expanduser()))

        new_values.update(values)
        return new_values