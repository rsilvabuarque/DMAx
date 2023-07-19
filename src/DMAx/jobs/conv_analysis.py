"""Jobs used to perform the convergence analysis."""

from jobflow import Flow, Response, job

from DMAx.jobs.base import BaseLammpsMaker
from DMAx.sets.base import LammpsInputGenerator
from DMAx.sets.base import DMAConvAnalysisInputGenerator

"""
ATTEMPT 1:

@dataclass
class DmaConvAnalysisDataMaker(BaseLammpsMaker):
    """Maker to perform the convergence analysis"""
    
    name: str = "convergence analysis"
    input_set_generator: LammpsInputGenerator = field(default_factory=DMAConvAnalysisInputGenerator)

    @job 
    def generate_conv_analysis_inputs(
        
    ):

    @job
    def run_conv_analysis(

    ):

    @job
    def parse_conv_analysis_outputs(

    ):
"""

# ATTEMPT 2
@dataclass
class DmaConvAnalysisDataMaker():
    """Maker to perform the convergence analysis"""
    
    name: str = "convergence analysis"
    input_set_generator: LammpsInputGenerator = field(default_factory=DMAConvAnalysisInputGenerator)

    @job 
    def generate_conv_analysis_inputs(
        
    ):

    @job
    def run_conv_analysis(

    ):

    @job
    def parse_conv_analysis_outputs(

    ):
    
    @job
    def make(self, 
    