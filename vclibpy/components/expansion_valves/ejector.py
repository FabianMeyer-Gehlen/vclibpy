"""
Module with semi-physical model of the transcritical CO2 ejector
"""

import logging
import math

import numpy

import vclibpy.media.ref_prop
from vclibpy.components.component import ThreePortComponent
from vclibpy.media import ThermodynamicState, MedProp

logger = logging.getLogger(__name__)


class Ejector(ThreePortComponent):
    """
    Ejector model according to Liu and Groll 2013:
    'Study of ejector efficiencies in refrigeration cycles'

    Assumptions:

    - flow inside ejector is steady and one-dimensional
    - the motive nozzle is a converging nozzle
    - at the motive nozzle exit the flow is sonic (choked flow)
    - isentropic efficiencies of subcomponents are given
    - ejector walls are adiabatic
    - inlet-flow velocity is neglected
    - gravitational effects are neglected
    - at the mixing chamber outlet the two mass flows are assumed mixed perfectly


    For more information on the model refer to the paper
    """

    def __init__(self,
                 d_throat: float,
                 d_mixing: float,
                 c_m: float = 0.73,  # Efficiency constant for the mixing chamber of a given ejector - 0.73 is the value for the ejector used by Zhu
                 phi_n: float = 0.95,  # Isentropic efficiency of primary nozzle according to Zhu 2018: Theoretical model
                 phi_d: float = 0.9,  # Isentropic efficiency of diffuser according to Zhu 2018: Theoretical model
                 v_step_min = 0.00001,
                 v_step_max = 0.1,
                 **kwargs):
        """Initialize class with kwargs"""
        self.max_err = kwargs.pop("max_err", 0.5)
        self.min_iteration_step = kwargs.pop("min_iteration_step", 1)
        self.show_iteration = kwargs.get("show_iteration", False)
        self.use_quick_solver = kwargs.pop("use_quick_solver", True)
        self.max_num_iterations = kwargs.pop("max_num_iterations", int(1e5))
        self.step_max = kwargs.pop("step_max", 10000)
        super().__init__()
        self.d_throat = d_throat
        self.d_mixing = d_mixing
        self.state_throat: ThermodynamicState = None
        self.state_primary_mixing: ThermodynamicState = None  # Thermodynamic state of primary flow at mixing chamber
        self.state_mixing: ThermodynamicState = None  # Thermodynamic state of mixed flow at mixing chamber
