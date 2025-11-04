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
        self.newton_relaxation_factor = kwargs.pop("newton_relaxation_factor", 0.8)
        self.newton_step_size = kwargs.pop("newton_step_size", 1e-7)
        self.step_max = kwargs.pop("step_max", 1000000)
        super().__init__()
        self.d_throat = d_throat
        self.d_mixing = d_mixing


    def calculate_motive_nozzle(self, p_motive: float, p_suction: float, h_motive: float):
        """
        Calculate state and velocity inside motive nozzle throat

        Returns:
            None
        """


        # Check if given parameters are in the valid range for the empirical correlations
        if not 8e6 <= p_motive <= 14e6:
            raise ValueError(f"p_motive ({p_motive:.3f} Pa) out of range (8 MPa to 14 MPa). Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")
        if not 2.5e6 <= p_suction <= 5e6:
            raise ValueError(f"p_suction ({p_suction:.3f} Pa) out of range (2.5 MPa to 5 MPa). Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")
        if not 1.8 <= self.d_throat <= 2.7:
            raise ValueError(f"d_throat ({self.d_throat:.3f} mm) out of range (1.8 mm to 2.7 mm). Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")
        if not self.d_mixing == 4:
            raise ValueError(f"d_mixing ({self.d_mixing:.3f} mm) has to be 4 mm. Unable to calculate motive nozzle efficiency using the correlation of Liu and Groll.")

        # Set state at motive nozzle inlet from given parameters
        self.state_primary = self.med_prop.calc_state("PH", p_motive, h_motive)

        # Calculation of the isentropic motive nozzle efficiency according to Liu and Grolls empirical correlation
        pi = p_motive / p_suction
        a = self.d_throat / self.d_mixing
        eta_is_motive = -36.137 - 4.160*pi + 1.161*pi**2 - 0.106*pi**3 + 212.320*a - 355.359*a**2 + 196.035*a**3

        # Plausibility check
        if not 0<= eta_is_motive <= 1:
           raise ValueError("eta_is_motive must be between 0 and 1")

        # Initial guess for p_throat
        p_throat = p_suction + (p_motive-p_suction)*0.2  #ToDo find better start value for p_throat
        rel_err = []  # relative error in percent
        num_iterations = 0  # Number of iterations

        while True:
            num_iterations += 1
            if num_iterations >= self.max_num_iterations:
                raise RuntimeError("Maximum number of iterations for motive nozzle calculation exceeded. Stopping")

            # Calculate the enthalpy at the throat from isentropic efficiency
            h_throat = self.state_primary.h - eta_is_motive * (self.state_primary.h - self.med_prop.calc_state("PS", p_throat, self.state_primary.s).h)
            # From this the velocity at the throat can be calculated using an energy balance
            v_throat = (2*( self.state_primary.h - h_throat) )**0.5  # velocity at throat from energy balance
            # Also the speed of sound can be calculated
            q_throat = self.med_prop.calc_state("PH", p_throat, h_throat).q
            if 0 <= q_throat <= 1:
                c_throat = self.med_prop.get_two_phase_speed_of_sound(p_throat, q_throat)  # speed of sound at throat
            else:
                c_throat = self.med_prop.get_speed_of_sound(self.med_prop.calc_state("PH", p_throat, h_throat))  # speed of sound at throat

            # Check the error between calculated velocity and speed of sound. If it is small enough we can calculate all needed values and end the iteration
            rel_err.append((v_throat - c_throat)/c_throat*100)
            print(f"relative error: {rel_err[-1]}, p_throat: {p_throat}")
            if abs(rel_err[-1]) < self.max_err:
                self.state_primary_throat = self.med_prop.calc_state("PH", p_throat, h_throat)
                self.m_flow_primary = self.state_primary_throat.d * 1/4*self.d_throat**2 * v_throat
                break
            else:  # the same can be done again to calculate the local derivative for Newton's method
                p_throat_2 = p_throat * (1 + self.newton_step_size)
                h_throat_2 = self.state_primary.h - eta_is_motive * (self.state_primary.h - self.med_prop.calc_state("PS", p_throat_2, self.state_primary.s).h)
                v_throat_2 = (2 * (self.state_primary.h - h_throat_2)) ** 0.5
                q_throat_2 = self.med_prop.calc_state("PH", p_throat_2, h_throat_2).q
                if 0 <= q_throat_2 <= 1:
                    c_throat_2 = self.med_prop.get_two_phase_speed_of_sound(p_throat_2, q_throat_2)
                else:
                    c_throat_2 = self.med_prop.get_speed_of_sound(self.med_prop.calc_state("PH", p_throat_2, h_throat_2))

                # Now the local differential can be calculated and the next pressure step determined
                # Now the local differential can be calculated and the next pressure step determined
                differential = ((v_throat_2 - c_throat_2) - (v_throat - c_throat)) / (p_throat_2 - p_throat)

                # Ausgabe der Formel mit aktuellen Werten
                print(f"differential = ((v_throat_2 - c_throat_2) - (v_throat - c_throat)) / (p_throat_2 - p_throat)")
                print(f" = (({v_throat_2:.6f} - {c_throat_2:.6f}) - ({v_throat:.6f} - {c_throat:.6f})) / ({p_throat_2:.6f} - {p_throat:.6f}) = {differential:.6e}")

                p_step = (v_throat - c_throat) / differential * self.newton_relaxation_factor
                print(f"p_step: {p_step}")
                if abs(p_step) >= self.step_max:
                    p_step = self.step_max * np.sign(p_step)
                p_throat = p_throat - p_step


        #ToDO check for subcritical flow
        #ToDO check for boundaries of eta_motive at the end to verify applicability of the correlation (m flows)