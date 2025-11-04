"""
Module with semi-physical ejector model according to Liu and Groll 2013
"""
from vclibpy.components.expansion_valves.ejector import Ejector
import numpy as np
from scipy.optimize import fsolve

class EjectorLiu(Ejector):
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

    Args:
        d_throat (float): Diameter of the motive nozzle throat in mm (has to be between 1.8 and 2.7 mm for empirical correlations to work).
        d_mixing (float): Diameter of the mixing chamber in mm (has to be 4 mm for empirical correlations to work).
        **kwargs: Additional keyword arguments for the iteration.
    """

    def __init__(self,
                 d_throat: float = 1.8,
                 d_mixing: float = 4,
                 **kwargs):
        """Initialize class with kwargs"""
        self.max_err = kwargs.pop("max_err", 0.5)
        self.show_iteration = kwargs.get("show_iteration", False)
        self.use_quick_solver = kwargs.pop("use_quick_solver", True)
        self.max_num_iterations = kwargs.pop("max_num_iterations", int(1e5))
        self.newton_relaxation_factor = kwargs.pop("newton_relaxation_factor", 0.5)
        self.newton_step_size = kwargs.pop("newton_step_size", 1e-6)
        super().__init__()
        self.d_throat = d_throat
        self.d_mixing = d_mixing


