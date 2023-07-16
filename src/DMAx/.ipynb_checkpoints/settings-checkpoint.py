"""Settings for DMAx."""

from pydantic import BaseSettings, Field

_DEFAULT_CONFIG_FILE_PATH = "~/.dmax.yaml"

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
    
    # general settings
    CONFIG_FILE: str = Field(
        _DEFAULT_CONFIG_FILE_PATH, description="File to load alternative defaults from."
    )
    
    
    # LAMMPS specific settings
    LAMMPS_CMD str = Field(
        "lammps_std", description="Command to run standard version of LAMMPS."
    class Config:
        """Pydantic config settings."""
        
        env_prefix = "dmax_"
    
