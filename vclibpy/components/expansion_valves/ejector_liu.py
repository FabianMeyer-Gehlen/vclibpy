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
        self.max_err = kwargs.pop("max_err", 0.005)
        self.show_iteration = kwargs.get("show_iteration", False)
        self.use_quick_solver = kwargs.pop("use_quick_solver", True)
        self.max_num_iterations = kwargs.pop("max_num_iterations", int(1e5))
        self.newton_relaxation_factor = kwargs.pop("newton_relaxation_factor", 1.0)  # Starting value for the relaxation factor in the Newton-Raphson method
        self.newton_step_size = kwargs.pop("newton_step_size", 1e-6)  # Relative step size for the numerical derivative in the Newton-Raphson method
        self.step_max = kwargs.pop("step_max", 1000000)  # Maximum step size for pressure correction during Newton-Raphson method in Pa
        super().__init__()
        self.d_throat = d_throat
        self.d_mixing = d_mixing


    def calculate_motive_nozzle(self, p_motive: float, p_suction: float, h_motive: float):
        """
        Calculate state and velocity inside motive nozzle throat

        Returns:
            None
        """

        # Set state at motive nozzle inlet from given parameters
        self.state_primary = self.med_prop.calc_state("PH", p_motive, h_motive)

        # Check if given parameters are in the valid range for the empirical correlations
        if not 8e6 <= p_motive <= 14e6:
            raise ValueError(f"p_motive ({p_motive:.3f} Pa) out of range (8 MPa to 14 MPa). Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")
        if not 2.5e6 <= p_suction <= 5e6:
            raise ValueError(f"p_suction ({p_suction:.3f} Pa) out of range (2.5 MPa to 5 MPa). Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")
        if not 40 + 273.15 <= self.state_primary.T <= 60 + 273.15:
            raise ValueError(f"T_motive ({self.state_primary.T-273.15:.3f} °C) out of range (40 °C to 60 °C). Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")
        if not 1.8 <= self.d_throat <= 2.7:
            raise ValueError(f"d_throat ({self.d_throat:.3f} mm) out of range (1.8 mm to 2.7 mm). Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")
        if not self.d_mixing == 4:
            raise ValueError(f"d_mixing ({self.d_mixing:.3f} mm) has to be 4 mm. Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")

        # Calculation of the isentropic motive nozzle efficiency according to Liu and Grolls empirical correlation
        pi = p_motive / p_suction
        a = self.d_throat / self.d_mixing
        eta_is_motive = -36.137 - 4.160*pi + 1.161*pi**2 - 0.106*pi**3 + 212.320*a - 355.359*a**2 + 196.035*a**3

        # Plausibility check
        if not 0<= eta_is_motive <= 1:
           raise ValueError("eta_is_motive must be between 0 and 1")

        # Initial guess for p_throat
        p_throat: list[float] = []
        p_throat.append(p_suction + (p_motive-p_suction)*0.5)  #ToDo find better start value for p_throat
        p_throat.append(p_throat[0] * (1 + self.newton_step_size))
        rel_err = []  # relative error in percent
        num_iterations = 0  # Number of iterations

        while True:
            num_iterations += 1
            if num_iterations >= self.max_num_iterations:
                raise RuntimeError("Maximum number of iterations for motive nozzle calculation exceeded. Stopping")

            # arrays to store calculated values at throat for current pressure [0] and infinitesimal pressure step [1]
            h_throat: list[float] = [-1.0, -1.0]
            v_throat: list[float] = [-1.0, -1.0]
            q_throat: list[float] = [-1.0, -1.0]
            c_throat: list[float] = [-1.0, -1.0]

            for i in range (0, 2):
                # Calculate the enthalpy at the throat from isentropic efficiency
                h_throat[i] = self.state_primary.h - eta_is_motive * (self.state_primary.h - self.med_prop.calc_state("PS", p_throat[i], self.state_primary.s).h)
                # From this the velocity at the throat can be calculated using an energy balance
                v_throat[i] = (2*( self.state_primary.h - h_throat[i]) )**0.5  # velocity at throat from energy balance
                # Also the speed of sound can be calculated
                q_throat[i] = self.med_prop.calc_state("PH", p_throat[i], h_throat[i]).q
                if 0 <= q_throat[i] <= 1:
                    c_throat[i] = self.med_prop.get_two_phase_speed_of_sound(p_throat[i], q_throat[i])  # speed of sound at throat
                else:
                    c_throat[i] = self.med_prop.get_speed_of_sound(self.med_prop.calc_state("PH", p_throat[i], h_throat[i]))  # speed of sound at throat

            # Check the error between calculated velocity and speed of sound. If it is small enough we can calculate all needed values and end the iteration
            rel_err.append((v_throat[0] - c_throat[0])/c_throat[0]*100)
            print(rel_err[-1], p_throat)
            print(self.newton_relaxation_factor)

            # Calculate the residual for the Newton-Raphson method
            res = v_throat[0] - c_throat[0]  # Ziel: res -> 0 (v == c)
            if 'prev_res' not in locals():
                prev_res = res

            # Correcting the relaxation factor depending on the last step
            if np.sign(res) != np.sign(prev_res) or abs(res) > abs(prev_res):  # If the sign of the residual changed or the error increased, reduce the relaxation factor to prevent oscillations
                self.newton_relaxation_factor = max(0.1, self.newton_relaxation_factor * 0.8)
            else:
                self.newton_relaxation_factor = min(1.0, self.newton_relaxation_factor * 1.1)
            prev_res = res

            # Check if the error is small enough to stop the iteration
            if abs(rel_err[-1]) < self.max_err:
                self.state_primary_throat = self.med_prop.calc_state("PH", p_throat[0], h_throat[0])
                self.m_flow_primary = self.state_primary_throat.d * np.pi * 1/4*(self.d_throat*1e-3)**2 * v_throat[0]
                if not 0.1 <= self.m_flow_primary <= 0.25:
                    raise ValueError(f"Calculated mass flow rate ({self.m_flow_primary:.3f} kg/s) is outside of validity range for ejector model (0.1-0.5 kg/s). Check input parameters.")
                break
            else:  # If the error is still to large, the local differential can be calculated and the next pressure step determined
                differential = ((v_throat[1] - c_throat[1]) - (v_throat[0] - c_throat[0])) / (p_throat[1] - p_throat[0])
                p_step = (v_throat[0] - c_throat[0]) / differential * self.newton_relaxation_factor
                if abs(p_step) >= self.step_max:
                    p_step = self.step_max * np.sign(p_step)
                p_throat[0] = p_throat[0] - p_step
                p_throat[1] = p_throat[0] * (1 + self.newton_step_size)


        #ToDO check for subcritical flow