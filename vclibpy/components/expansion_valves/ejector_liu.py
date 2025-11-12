"""
Module with semi-physical ejector model according to Liu and Groll 2013
"""

from vclibpy.components.expansion_valves.ejector import Ejector
import numpy as np
import matplotlib.pyplot as plt

class EjectorLiu(Ejector):
    """
    Ejector model according to Liu and Groll 2013:
    'Study of ejector efficiencies in refrigeration cycles'
    Additional information on the model was gathered from the ARTI report:
    'Recovery of throttling losses by a two-phase ejector in a vapor compression cycle' from Liu and Groll 2008

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
        d_throat (float): Diameter of the motive nozzle throat in mm (has to be between 1.8 and 2.7 mm for empirical correlations to work. Suggested ratio of d_throat to d_mixing by Barta et al. 2021 is 0.45).
        d_suction (float): Diameter of the suction nozzle in mm (suggested ratio of d_throat to d_suction by Barta et al. 2021 is 0.33).
        d_mixing (float): Diameter of the mixing chamber in mm (has to be 4 mm for empirical correlations to work).
        d_diff (float): Diameter of the diffusor outlet in mm (suggested value by Barta et al. 2021 is 12 mm).
        **kwargs: Additional keyword arguments for the iteration.
    """

    def __init__(self,
                 d_throat: float = 1.8,
                 d_suction: float = 5.45,
                 d_mixing: float = 4,
                 d_diff: float = 12,
                 **kwargs):
        """Initialize class with kwargs"""
        self.max_err = kwargs.pop("max_err_newton", 0.0001)
        self.show_iteration = kwargs.get("show_iteration", False)
        self.max_num_iterations = kwargs.pop("max_num_iterations", int(1e2))
        self.newton_relaxation_factor = kwargs.pop("newton_relaxation_factor", 1)  # Starting value for the relaxation factor in the Newton-Raphson method
        self.newton_step_size = kwargs.pop("newton_step_size", 1e-6)  # Relative step size for the numerical derivative in the Newton-Raphson method
        self.step_max = kwargs.pop("step_max", 1000000)  # Maximum step size for pressure correction during Newton-Raphson method in Pa
        super().__init__()
        self.d_throat = d_throat
        self.d_suction = d_suction
        self.d_mixing = d_mixing
        self.v_throat: float = -1.0  # Velocity at motive nozzle throat
        self.v_suction: float = -1.0  # Velocity at suction nozzle outlet
        self.v_mix: float = -1.0  # Velocity at mixing chamber outlet
        self.d_diff = d_diff


    def calculate_motive_nozzle(self, p_motive: float, h_motive: float, p_suction: float):
        """
        Calculate state and velocity inside motive nozzle throat

        Returns:
            None
        """

        # Set state at motive nozzle inlet from given parameters
        self.state_primary = self.med_prop.calc_state("PH", p_motive, h_motive)
        newton_relaxation_factor = self.newton_relaxation_factor  # Reset relaxation factor for Newton-Raphson method

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
        eta_is_motive = -36.1367305 - 4.15962963*pi + 1.16131867*pi**2 - 0.106090279*pi**3 + 212.320405*a - 355.359177*a**2 + 196.035242*a**3

        # Plausibility check
        if not 0<= eta_is_motive <= 1:
           raise ValueError("eta_is_motive must be between 0 and 1")

        # Initial guess for p_throat
        p_throat: list[float] = []
        p_throat.append(p_suction + (p_motive-p_suction)*0.3)  #ToDo find better start value for p_throat
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
                v_throat[i] = (2*(self.state_primary.h - h_throat[i]) )**0.5  # velocity at throat from energy balance
                # Also the speed of sound can be calculated
                q_throat[i] = self.med_prop.calc_state("PH", p_throat[i], h_throat[i]).q
                if 0 <= q_throat[i] <= 1:
                    c_throat[i] = self.med_prop.get_two_phase_speed_of_sound(p_throat[i], q_throat[i])  # speed of sound at throat
                else:
                    c_throat[i] = self.med_prop.get_speed_of_sound(self.med_prop.calc_state("PH", p_throat[i], h_throat[i]))  # speed of sound at throat

            # Check the error between calculated velocity and speed of sound. If it is small enough we can calculate all needed values and end the iteration
            rel_err.append((v_throat[0] - c_throat[0])/c_throat[0]*100)

            # Calculate the residual for the Newton-Raphson method
            res = v_throat[0] - c_throat[0]
            if 'prev_res' not in locals():
                prev_res = res

            # Correcting the relaxation factor depending on the last step
            if np.sign(res) != np.sign(prev_res) or abs(res) > abs(prev_res):  # If the sign of the residual changed or the error increased, reduce the relaxation factor to prevent oscillations
                newton_relaxation_factor = max(0.1, newton_relaxation_factor * 0.8)
            else:
                newton_relaxation_factor = min(1.0, newton_relaxation_factor * 1.1)
            prev_res = res

            # Check if the error is small enough to stop the iteration
            if abs(rel_err[-1]) < self.max_err:
                self.state_primary_throat = self.med_prop.calc_state("PH", p_throat[0], h_throat[0])
                self.m_flow_primary = self.state_primary_throat.d * np.pi * 1/4*(self.d_throat*1e-3)**2 * v_throat[0]
                self.v_throat = v_throat[0]
                print(f"Motive nozzle converged in {num_iterations} iterations with v_throat={v_throat[0]:.2f} m/s")
                if not 0.1 <= self.m_flow_primary <= 0.25:
                    raise ValueError(f"Calculated mass flow rate ({self.m_flow_primary:.3f} kg/s) is outside of validity range for ejector model (0.1-0.25 kg/s). Check input parameters.")
                break
            else:  # If the error is still to large, the local differential can be calculated and the next pressure step determined
                differential = ((v_throat[1] - c_throat[1]) - (v_throat[0] - c_throat[0])) / (p_throat[1] - p_throat[0])
                p_step = (v_throat[0] - c_throat[0]) / differential * newton_relaxation_factor
                if abs(p_step) >= self.step_max:
                    p_step = self.step_max * np.sign(p_step)
                p_throat[0] = p_throat[0] - p_step
                p_throat[1] = p_throat[0] * (1 + self.newton_step_size)


        #ToDO check for subcritical flow

    def calculate_suction_nozzle(self, entrainment_ratio: float, p_suction: float, h_suction: float):
        """
        Calculate state and velocity inside suction nozzle

        Returns:
            None
        """

        # Set state at suction nozzle inlet from given parameters
        self.state_secondary = self.med_prop.calc_state("PH", p_suction, h_suction)

        newton_relaxation_factor = self.newton_relaxation_factor  # Reset relaxation factor for Newton-Raphson method

        # Calculate mass flow rate of secondary flow from entrainment ratio and primary mass flow rate
        self.m_flow_secondary = entrainment_ratio * self.m_flow_primary

        if not 2.5e6 <= p_suction <= 5e6:
            raise ValueError(f"p_suction ({p_suction:.3f} Pa) out of range (2.5 MPa to 5 MPa). Unable to calculate suction nozzle efficiency using the correlation of Liu and Groll.")
        if not 15 + 273.15 <= self.state_secondary.T <= 26 + 273.15:
            raise ValueError(f"T_suction ({self.state_secondary.T-273.15:.3f} °C) out of range (15 °C to 26 °C). Unable to calculate suction nozzle efficiency using the correlation of Liu and Groll.")
        if not 0.05 <= self.m_flow_secondary <= 0.07:
            raise ValueError(f"Calculated mass flow rate ({self.m_flow_secondary:.3f} kg/s) is outside of validity range for ejector model (0.05-0.07 kg/s). Check input parameters.")

        # Calculation of the isentropic suction nozzle efficiency according to Liu and Grolls empirical correlation
        pi = self.state_primary.p / p_suction
        z = entrainment_ratio * pi**0.02
        eta_is_suction = (-3173.171 + 934.102*pi - 314.4712*pi**2 + 79.52134*pi**3 - 12.22236*pi**4 + 0.814459*pi**5 +
                          694222.1*entrainment_ratio - 2956145*entrainment_ratio**2 + 7950453*entrainment_ratio**3 - 11432720*entrainment_ratio**4 + 6689155*entrainment_ratio**5 -
                          649905.1*z + 2647000*z**2 - 6885025*z**3 + 9627161*z**4 - 5490126*z**5)

        # Plausibility check
        if not 0<= eta_is_suction <= 1:
              raise ValueError("eta_is_suction must be between 0 and 1")

        # Initial guess for p_suction_exit
        p_suction_exit: list[float] = []
        p_suction_exit.append(p_suction * 0.995)  #ToDo find better start value for p_suction_exit
        p_suction_exit.append(p_suction_exit[0] * (1 + self.newton_step_size))
        rel_err = []  # relative error in percent
        num_iterations = 0  # Number of iterations

        A_suction = np.pi/4 *((self.d_suction*1e-3)**2-(self.d_throat*1e-3)**2)  # Cross-sectional area of suction nozzle  #ToDO check with Barta if this is correct, oder if the area should be an annulus

        while True:
            num_iterations += 1
            if num_iterations >= self.max_num_iterations:
                raise RuntimeError("Maximum number of iterations for suction nozzle calculation exceeded. Stopping")

            # arrays to store calculated values at suction nozzle exit for current pressure [0] and infinitesimal pressure step [1]
            h_suction_exit: list[float] = [-1.0, -1.0]
            v_suction_exit: list[float] = [-1.0, -1.0]
            v_conservation_mass: list[float] = [-1.0, -1.0]

            for i in range (0, 2):
                # calculate enthalpy at suction nozzle exit from isentropic efficiency
                h_suction_exit[i] = self.state_secondary.h - eta_is_suction * (self.state_secondary.h - self.med_prop.calc_state("PS", p_suction_exit[i], self.state_secondary.s).h)
                # calculate velocity at suction nozzle exit from energy balance
                v_suction_exit[i] = (2*(self.state_secondary.h - h_suction_exit[i]) )**0.5
                # calculate velocity at suction nozzle exit from mass flow conservation
                v_conservation_mass[i] = self.m_flow_secondary / (self.med_prop.calc_state("PH", p_suction_exit[i], h_suction_exit[i]).d * A_suction)

            rel_err.append((v_suction_exit[0] - v_conservation_mass[0])/v_conservation_mass[0]*100)

            # Calculate the residual for the Newton-Raphson method
            res = v_suction_exit[0] - v_conservation_mass[0]
            if 'prev_res' not in locals():
                prev_res = res

            # Correcting the relaxation factor depending on the last step
            if np.sign(res) != np.sign(prev_res) or abs(res) > abs(prev_res):  # If the sign of the residual changed or the error increased, reduce the relaxation factor to prevent oscillations
                newton_relaxation_factor = max(0.1, newton_relaxation_factor * 0.8)
            else:
                newton_relaxation_factor = min(1.0, newton_relaxation_factor * 1.1)
            prev_res = res

            # c=self.med_prop.get_two_phase_speed_of_sound(p_suction_exit[0], self.med_prop.calc_state("PH", p_suction_exit[0], h_suction_exit[0]).q)
            c = self.med_prop.get_speed_of_sound(self.med_prop.calc_state("PH", p_suction_exit[0], h_suction_exit[0]))

            print(f"Iteration {num_iterations}: p_suction_exit={p_suction_exit[0]:.2f} Pa, v_suction_exit={v_suction_exit[0]:.2f} m/s, v_conservation_mass={v_conservation_mass[0]:.2f} m/s, rel_err={rel_err[-1]:.5f} %, c = {c:.2f} m/s") if self.show_iteration else None

            # Check if the error is small enough to stop the iteration
            if abs(rel_err[-1]) < self.max_err:
                self.state_secondary_mixing = self.med_prop.calc_state("PH", p_suction_exit[0], h_suction_exit[0])
                self.v_suction = v_suction_exit[0]
                print(f"Suction nozzle converged in {num_iterations} iterations with v_suction_exit={v_suction_exit[0]:.2f} m/s")
                break
            else:  # If the error is still to large, the local differential can be calculated and the next pressure step determined
                differential = (((v_suction_exit[1] - v_conservation_mass[1]) - (v_suction_exit[0] - v_conservation_mass[0])) /
                                (p_suction_exit[1] - p_suction_exit[0]))
                p_step = (v_suction_exit[0] - v_conservation_mass[0]) / differential * newton_relaxation_factor

                if abs(p_step) >= self.step_max:
                    p_step = self.step_max * np.sign(p_step)

                p_suction_exit[0] = p_suction_exit[0] - p_step

                if p_suction_exit[0] >= self.state_secondary.p:
                    p_suction_exit[0] = self.state_secondary.p * 0.995  # prevent non-physical pressure values
                    newton_relaxation_factor *= 0.8  # reduce relaxation factor to prevent oscillations

                p_suction_exit[1] = p_suction_exit[0] * (1 + self.newton_step_size)


    def calculate_mixing_chamber(self, entrainment_ratio: float):
        """
        Calculate state and velocity inside mixing chamber

        Returns:
            None
        """

        #  Calculation of mixing chamber efficiency according to Liu and Grolls empirical correlation
        z = (self.d_throat/self.d_mixing)**0.1 * (1+entrainment_ratio)**0.35
        eta_mixing = -6869.077 + 19308.18*z - 18089.31*z**2 + 5649.417*z**3

        newton_relaxation_factor = self.newton_relaxation_factor  # Reset relaxation factor for Newton-Raphson method

        # Total mass flow at mixing chamber and diffusor
        self.m_flow_outlet = self.m_flow_primary + self.m_flow_secondary

        # Calculate geometric parameters
        A_mix = np.pi * 1/4*(self.d_mixing*1e-3)**2  # Cross-sectional area of mixing chamber
        A_throat = np.pi * 1/4*(self.d_throat*1e-3)**2  # Cross-sectional area of motive nozzle throat
        A_suction = A_mix - A_throat  # Cross-sectional area of suction nozzle  #ToDO check with Barta if this is correct, oder if the area should be an annulus

        rel_err: list[tuple[float, float, float]] = []  # relative error in percent
        var: list[tuple[float, float, float]] = []  # store variables for each iteration
        num_iterations = 0  # Number of iterations

        # Set starting values for p_mix, h_mix and v_mix
        p_mix = (self.state_primary.p + self.state_secondary.p) / 2  # Starting value for mixing chamber pressure
        h_mix = 3e5  # Starting value for mixing chamber enthalpy
        v_mix = (self.v_throat + self.v_suction) / 2  # Starting value for mixing chamber velocity

        if self.show_iteration:
            fig_iterations, ax_iterations = plt.subplots(3, 2, sharex=True)
            plot_last = -100

        while True:
            num_iterations += 1
            print(f"Iteration {num_iterations}: p_mix={p_mix:.2f} Pa, h_mix={h_mix:.2f} J/kg, v_mix={v_mix:.2f} m/s")
            if num_iterations >= self.max_num_iterations:
                raise RuntimeError("Maximum number of iterations for mixing chamber calculation exceeded. Stopping")

            # Calculate density at mixing chamber outlet from current guess values
            rho_mix = self.med_prop.calc_state("PH", p_mix, h_mix).d

            # Calculate the three equations for mass, momentum and energy conservation. They should become 0 at the solution
            eq1 = ((self.m_flow_primary + self.m_flow_secondary - rho_mix * A_mix * v_mix) / self.m_flow_primary)  # Mass conservation
            eq2 = ((self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat +
                    self.state_secondary_mixing.p * (A_mix - A_throat) +
                    eta_mixing * self.state_secondary_mixing.d * (A_mix - A_throat) * self.v_suction ** 2 -
                    p_mix * A_mix - rho_mix * A_mix * v_mix ** 2) /
                   (self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat))  # Momentum conservation
            eq3 = ((self.m_flow_primary * (self.state_primary_throat.h + 0.5 * self.v_throat ** 2) +
                    self.m_flow_secondary * (self.state_secondary_mixing.h + 0.5 * self.v_suction ** 2) -
                    self.m_flow_outlet * (h_mix + 0.5 * v_mix ** 2)) /
                   (self.m_flow_primary * (self.state_primary_throat.h + 0.5 * self.v_throat ** 2)))  # Energy conservation

            rel_err.append((eq1, eq2, eq3))
            var.append((p_mix, h_mix, v_mix))

            print("Errors: ", rel_err[-1])

            if self.show_iteration:
                for ax in ax_iterations.flatten():
                    ax.clear()
                iterations = list(range(len(rel_err)))[plot_last:]
                rel_err_slice = rel_err[plot_last:]
                var_slice = var[plot_last:]
                if rel_err_slice and var_slice:
                    p_vals, h_vals, v_vals = zip(*var_slice)
                    err1, err2, err3 = zip(*rel_err_slice)
                else:
                    p_vals = h_vals = v_vals = ()
                    err1 = err2 = err3 = ()
                ax_iterations[0, 0].set_ylabel("p_mix in Pa")
                ax_iterations[0, 1].set_ylabel("err_eq1")
                ax_iterations[1, 0].set_ylabel("h_mix in J/kg")
                ax_iterations[1, 1].set_ylabel("err_eq2")
                ax_iterations[2, 0].set_ylabel("v_mix in m/s")
                ax_iterations[2, 1].set_ylabel("err_eq3")
                ax_iterations[0, 0].scatter(iterations, p_vals)
                ax_iterations[0, 1].scatter(iterations, err1)
                ax_iterations[1, 0].scatter(iterations, h_vals)
                ax_iterations[1, 1].scatter(iterations, err2)
                ax_iterations[2, 0].scatter(iterations, v_vals)
                ax_iterations[2, 1].scatter(iterations, err3)
                plt.draw()
                plt.pause(1e-5)

            # Check if all errors are small enough to stop the iteration
            if max(abs(x) for x in rel_err[-1]) <= self.max_err:
                self.state_mixing = self.med_prop.calc_state("PH", p_mix, h_mix)
                self.v_mix = v_mix
                print(f"Mixing chamber converged in {num_iterations} iterations with v_mix={v_mix:.2f} m/s")
                print(eq1, eq2, eq3)
                break

            # If still here, the errors are too large, and we have to do another iteration
            # Calculate Jacobian matrix numerically to determine next steps for p_mix, h_mix and v_mix
            state_mix = self.med_prop.calc_state("PH", p_mix, h_mix)

            drho_dp = self.med_prop.get_partial_derivative("D", "P", "H", state_mix)
            drho_dh = self.med_prop.get_partial_derivative("D", "H", "P", state_mix)

            J = np.zeros((3, 3))  # Jacobian matrix

            # d(eq1)/d(p_mix, h_mix, v_mix)
            J[0,0] = -A_mix * v_mix * drho_dp / self.m_flow_primary
            J[0,1] = -A_mix * v_mix * drho_dh / self.m_flow_primary
            J[0,2] = -A_mix * state_mix.d / self.m_flow_primary
            # d(eq2)/d(p_mix, h_mix, v_mix)
            denom_mom = self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat
            J[1,0] = -A_mix * (1 + v_mix ** 2 * drho_dp) / denom_mom
            J[1,1] = -A_mix * v_mix ** 2 * drho_dh / denom_mom
            J[1,2] = -2 * A_mix * state_mix.d * v_mix / denom_mom
            # d(eq3)/d(p_mix, h_mix, v_mix)
            denom_en = self.m_flow_primary * (self.state_primary_throat.h + 0.5 * self.v_throat ** 2)
            J[2,0] = 0
            J[2,1] = -self.m_flow_outlet / denom_en
            J[2,2] = -self.m_flow_outlet * v_mix / denom_en

            # print(J)
            # print(np.linalg.cond(J))

            # Spaltenskalierung zur Konditionsverbesserung
            col_scale = np.maximum(np.linalg.norm(J, axis=0), 1e-12)
            J_s = J / col_scale

            # print(J_s)
            # print(np.linalg.cond(J_s))

            f_vec = np.array(rel_err[-1], dtype=float)
            corrections = -np.linalg.solve(J_s, f_vec)
            # print(f"corrections: {corrections}")

            if 'prev_err' not in locals():
                prev_err = rel_err[-1]

            # If the sign of any residual changed or the error increased, reduce the relaxation factor to prevent oscillations
            # if np.any(np.sign(rel_err[-1])!=np.sign(prev_err)) or np.any(np.abs(rel_err[-1])>np.abs(prev_err)):
            #     newton_relaxation_factor = max(0.1, newton_relaxation_factor * 0.8)
            #     print(f"Reducing relaxation factor to {newton_relaxation_factor}")
            # else:
            #     newton_relaxation_factor = min(1.0, newton_relaxation_factor * 1.1)
            #     print(f"Increasing relaxation factor to {newton_relaxation_factor}")

            prev_err = rel_err[-1]

            # Now we can update p_mix, h_mix and v_mix with the calculated corrections
            # Prevent non-physical pressure values
            if p_mix + corrections[0] > self.state_primary.p:
                corrections[0] = self.state_primary.p - p_mix - 1e-6
            elif p_mix + corrections[0] < self.state_secondary.p:
                corrections[0] = self.state_secondary.p - p_mix + 1e-6
            p_mix += corrections[0] * newton_relaxation_factor
            h_mix += corrections[1] * newton_relaxation_factor
            if corrections[2] < -v_mix:  # Prevent negative velocities
                corrections[2] = -v_mix + 1e-6
            elif corrections[2] + v_mix > self.v_throat:
                corrections[2] = self.v_throat - v_mix - 1e-6
            v_mix += corrections[2] * newton_relaxation_factor


    def calculate_diffusor(self):
        """
        Calculate state inside diffusor

        Returns:
            None
        """
        # Calculation of correlation for pressure recovery coefficient proposed by Owen et al. 1992 used by Liu and Groll 2013
        rho_mix_g = self.med_prop.calc_state("PQ", self.state_mixing.p, 1).d
        rho_mix_l = self.med_prop.calc_state("PQ", self.state_mixing.p, 0).d
        c_t = (0.85 * self.state_mixing.d * (1-(self.d_mixing/self.d_diff)**4) *
               (self.state_mixing.q**2/rho_mix_g + (1-self.state_mixing.q)**2/rho_mix_l))

        # Pressure at diffusor outlet from momentum conservation
        p_diffusor = c_t * 1/2 * self.state_mixing.d*self.v_mix**2 + self.state_mixing.p

        # Enthalpy at diffusor outlet from energy conservation
        h_diffusor = ((self.state_primary.h*self.m_flow_primary + self.state_secondary.h*self.m_flow_secondary)/
                      (self.m_flow_primary + self.m_flow_secondary))

        self.state_outlet = self.med_prop.calc_state("PH", p_diffusor, h_diffusor)