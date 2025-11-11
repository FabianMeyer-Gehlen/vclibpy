"""
Module with classes for ejector models.
"""

import logging
import abc
from vclibpy.components.component import ThreePortComponent
from vclibpy.media import ThermodynamicState

logger = logging.getLogger(__name__)


class Ejector(ThreePortComponent, abc.ABC):
    """
    Base class for an ejector.
    """

    def __init__(self):
        super().__init__()
        self.state_primary_throat: ThermodynamicState = None  # Thermodynamic state of primary flow at motive nozzle throat
        self.state_primary_mixing: ThermodynamicState = None  # Thermodynamic state of primary flow at mixing chamber
        self.state_secondary_mixing: ThermodynamicState = None # Thermodynamic state of secondary flow at mixing chamber
        self.state_mixing: ThermodynamicState = None  # Thermodynamic state of mixed flow at mixing chamber

    @abc.abstractmethod
    def calculate_motive_nozzle(self, p_motive: float, h_motive: float, p_suction: float):
        """
        Calculate mass flow and state inside motive nozzle throat

        Args:
            p_motive (float): Pressure at motive nozzle inlet
            p_suction (float): Pressure at suction nozzle inlet
            h_motive (float): specific Enthalpy at motive nozzle inlet

        Returns:
            None
        """
        raise NotImplementedError

    # @abc.abstractmethod
    # def calculate_suction_nozzle(self):
    #     """
    #     Calculate mass flow and state inside suction nozzle
    #
    #     Returns:
    #         None
    #     """
    #     raise NotImplementedError

    # @abc.abstractmethod
    # def calculate_mixing_chamber(self):
    #     """
    #     Calculate mass flow and state inside mixing chamber
    #
    #     Returns:
    #         None
    #     """
    #     raise NotImplementedError
    #
    # @abc.abstractmethod
    # def calculate_diffusor(self):
    #     """
    #     Calculate state inside diffusor
    #
    #     Returns:
    #         None
    #     """
    #     raise NotImplementedError