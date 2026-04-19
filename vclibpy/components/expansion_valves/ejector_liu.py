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
        p_throat.append(p_suction + (p_motive-p_suction)*0.99)  #ToDo find better start value for p_throat
        p_throat.append(p_throat[0] * (1 + self.newton_step_size))
        rel_err = []  # relative error in percent
        num_iterations = 0  # Number of iterations

        # Setup for plotting if desired
        if self.show_iteration:
            p_throat_hist: list[float] = []  # history for plotting
            fig_m, ax_m = plt.subplots(2, 1, sharex=True)
            ax_m[0].set_ylabel("p_throat [Pa]")
            ax_m[1].set_ylabel("rel_err [%]")
            ax_m[1].set_xlabel("iteration")
            plt.ion()
            plt.show(block=False)

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
                newton_relaxation_factor = max(0.1, newton_relaxation_factor * 0.7)
            else:
                newton_relaxation_factor = min(1.0, newton_relaxation_factor * 1.1)
            prev_res = res

            # Print iteration data and plot if desired
            if self.show_iteration:
                print(f"Iteration {num_iterations}: p_throat={p_throat[0]:.2f} Pa, v_throat={v_throat[0]:.2f} m/s, c_throat={c_throat[0]:.2f} m/s, rel_err={rel_err[-1]:.5f} %"
                      f"\nentropy_throat={self.med_prop.calc_state('PH', p_throat[0], h_throat[0]).s:.2f} J/(kg*K), newton_relaxation_factor={newton_relaxation_factor:.3f}, eta_is_motive={eta_is_motive:.5f}")
                try:
                    p_throat_hist.append(float(p_throat[0]))
                    ax_m[0].clear()
                    ax_m[1].clear()
                    ax_m[0].plot(range(1, len(p_throat_hist) + 1), p_throat_hist, marker='o')
                    ax_m[1].plot(range(1, len(rel_err) + 1), rel_err, marker='o')
                    ax_m[0].set_ylabel("p_throat [Pa]")
                    ax_m[1].set_ylabel("rel_err [%]")
                    ax_m[1].set_xlabel("iteration")
                    plt.pause(1e-5)
                except Exception:
                    pass

            # Check if the error is small enough to stop the iteration
            if abs(rel_err[-1]) < self.max_err:
                self.state_primary_throat = self.med_prop.calc_state("PH", p_throat[0], h_throat[0])
                self.m_flow_primary = self.state_primary_throat.d * np.pi * 1/4*(self.d_throat*1e-3)**2 * v_throat[0]
                self.v_throat = v_throat[0]
                print(f"Motive nozzle converged in {num_iterations} iterations with v_throat={v_throat[0]:.2f} m/s")
                print(f"State Primary in: {self.state_primary}"
                        f"\nState Primary throat: {self.state_primary_throat}"
                        f"\nMass flow Primary: {self.m_flow_primary:.3f} kg/s")
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

        A_suction = np.pi/4 *((self.d_suction*1e-3)**2)  # Cross-sectional area of suction nozzle  #ToDO check with Barta if this is correct, oder if the area should be an annulus

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

            #print(f"Iteration {num_iterations}: p_suction_exit={p_suction_exit[0]:.2f} Pa, v_suction_exit={v_suction_exit[0]:.2f} m/s, v_conservation_mass={v_conservation_mass[0]:.2f} m/s, rel_err={rel_err[-1]:.5f} %, c = {c:.2f} m/s") if self.show_iteration else None

            # Check if the error is small enough to stop the iteration
            if abs(rel_err[-1]) < self.max_err:
                self.state_secondary_mixing = self.med_prop.calc_state("PH", p_suction_exit[0], h_suction_exit[0])
                self.v_suction = v_suction_exit[0]
                print(f"Suction nozzle converged in {num_iterations} iterations with v_suction_exit={v_suction_exit[0]:.2f} m/s")
                print(f"State Secondary in: {self.state_secondary}"
                      f"\nState Secondary mixing: {self.state_secondary_mixing}")
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

        rel_err: list[tuple[float, float]] = []  # relative error in percent
        var: list[tuple[float, float, float]] = []  # store variables for each iteration
        num_iterations = 0  # Number of iterations

        # Set starting values for p_mix, h_mix and v_mix
        p_mix = self.state_secondary.p + 2e5 # Starting value for mixing chamber pressure
        v_mix = (self.v_throat + self.v_suction) / 2  # Starting value for mixing chamber velocity
        h_mix = (self.m_flow_primary*self.state_primary.h + self.m_flow_secondary*self.state_secondary.h)/(self.m_flow_primary+self.m_flow_secondary) - v_mix**2/2  # Starting value for mixing chamber enthalpy

        if self.show_iteration:
            fig_iterations, ax_iterations = plt.subplots(3, 2, sharex=True)
            plot_last = -100  # only plot the last 100 iterations

        while True:
            num_iterations += 1
            # Reference energy term (constant for iteration)
            energy_ref = (self.m_flow_primary * self.state_primary.h + self.m_flow_secondary * self.state_secondary.h) / self.m_flow_outlet

            # Explicit velocity from energy equation: v_mix = sqrt(2 * (energy_ref - h_mix))
            diff_energy = energy_ref - h_mix
            if diff_energy <= 0:
                # Prevent non-physical negative argument in sqrt; damp h_mix implicitly
                v_mix = 1e-6
            else:
                v_mix = np.sqrt(2.0 * diff_energy)

            print(f"Iteration {num_iterations}: p_mix={p_mix:.2f} Pa, h_mix={h_mix:.2f} J/kg, v_mix={v_mix:.2f} m/s")
            if num_iterations >= self.max_num_iterations:
                raise RuntimeError("Maximum number of iterations for mixing chamber calculation exceeded. Stopping")

            # Density from current guess
            rho_mix = self.med_prop.calc_state("PH", p_mix, h_mix).d

            # Residuals (should go to 0)
            eq1 = (self.m_flow_primary + self.m_flow_secondary - rho_mix * A_mix * v_mix) / self.m_flow_primary  # mass
            eq2 = (
                self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat +
                self.state_secondary_mixing.p * (A_mix - A_throat) +
                eta_mixing * self.state_secondary_mixing.d * (A_mix - A_throat) * self.v_suction ** 2 -
                p_mix * A_mix - rho_mix * A_mix * v_mix ** 2
            ) / (self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat)  # momentum

            rel_err.append((eq1, eq2))
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
                    err1, err2 = zip(*rel_err_slice)
                else:
                    p_vals = h_vals = v_vals = ()
                    err1 = err2 = ()
                ax_iterations[0, 0].set_ylabel("p_mix Pa")
                ax_iterations[0, 1].set_ylabel("err_eq1")
                ax_iterations[1, 0].set_ylabel("h_mix J/kg")
                ax_iterations[1, 1].set_ylabel("err_eq2")
                ax_iterations[2, 0].set_ylabel("v_mix m/s")
                ax_iterations[2, 1].set_ylabel("unused")
                ax_iterations[0, 0].scatter(iterations, p_vals)
                ax_iterations[0, 1].scatter(iterations, err1)
                ax_iterations[1, 0].scatter(iterations, h_vals)
                ax_iterations[1, 1].scatter(iterations, err2)
                ax_iterations[2, 0].scatter(iterations, v_vals)
                plt.draw()
                plt.pause(1e-5)

                # Error surface: absolute total error = |eq1| + |eq2| over a grid of (p_mix, h_mix)
                # Compute and plot once per iteration (lightweight grid; adjust as needed)
                try:
                    if not hasattr(self, "_fig_errsurf"):
                        self._fig_errsurf, self._ax_errsurf = plt.subplots()
                        self._fig_errsurf.canvas.manager.set_window_title("Mixing chamber absolute total error surface")

                    p_min = float(self.state_secondary_mixing.p)
                    p_max = float(self.state_primary_throat.p)
                    # Enthalpy bounds around physically relevant range
                    h_lo = float(min(self.state_secondary_mixing.h, self.state_primary_throat.h))
                    h_hi = float(max(self.state_primary.h, self.state_secondary.h))
                    # Expand slightly to visualize edges
                    h_pad = 0.02 * max(1.0, abs(h_hi - h_lo))
                    h_lo -= h_pad
                    h_hi += h_pad

                    Np, Nh = 60, 60
                    p_grid = np.linspace(p_min, p_max, Np)
                    h_grid = np.linspace(h_lo, h_hi, Nh)
                    err_grid = np.full((Nh, Np), np.nan, dtype=float)

                    denom_mom = self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat

                    for ih, hh in enumerate(h_grid):
                        # Velocity from energy equation for each h
                        diff_e = energy_ref - hh
                        if diff_e <= 0:
                            # non-physical; leave as NaN
                            continue
                        v_loc = np.sqrt(2.0 * diff_e)
                        for ip, pp in enumerate(p_grid):
                            try:
                                rho_loc = self.med_prop.calc_state("PH", pp, hh).d
                                eq1_g = (self.m_flow_primary + self.m_flow_secondary - rho_loc * A_mix * v_loc) / self.m_flow_primary
                                eq2_g = (
                                    self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat +
                                    self.state_secondary_mixing.p * (A_mix - A_throat) +
                                    eta_mixing * self.state_secondary_mixing.d * (A_mix - A_throat) * self.v_suction ** 2 -
                                    pp * A_mix - rho_loc * A_mix * v_loc ** 2
                                ) / denom_mom
                                err_grid[ih, ip] = abs(eq1_g) + abs(eq2_g)
                            except Exception:
                                # invalid state; keep NaN
                                pass

                    ax = self._ax_errsurf
                    ax.clear()
                    # Use contourf with logarithmic-like levels if dynamic range is wide
                    valid = np.isfinite(err_grid)
                    if np.any(valid):
                        vmin = np.nanpercentile(err_grid, 5)
                        vmax = np.nanpercentile(err_grid, 95)
                        levels = np.linspace(vmin, vmax, 20) if vmax > vmin else 20
                        cf = ax.contourf(p_grid, h_grid, err_grid, vmin=0.0, levels=levels, cmap="viridis")
                        # Overlay current iterate
                        ax.scatter([p_mix], [h_mix], c="r", s=30, label="current iterate")
                        ax.set_xlabel("p_mix [Pa]")
                        ax.set_ylabel("h_mix [J/kg]")
                        ax.set_title("Absolute total error surface: |eq1| + |eq2|")
                        if not hasattr(self, "_errsurf_cbar"):
                            self._errsurf_cbar = self._fig_errsurf.colorbar(cf, ax=ax)
                        else:
                            self._errsurf_cbar.update_normal(cf)
                        ax.legend(loc="best")
                        self._fig_errsurf.tight_layout()
                        plt.pause(1e-5)
                except Exception as _:
                    # Keep iteration robust if plotting or grid evaluation fails
                    pass

            # Convergence check
            if max(abs(x) for x in rel_err[-1]) <= self.max_err:
                self.state_mixing = self.med_prop.calc_state("PH", p_mix, h_mix)
                self.v_mix = v_mix
                print(f"Mixing chamber converged in {num_iterations} iterations with v_mix={v_mix:.2f} m/s")
                print(eq1, eq2)
                break

            # Jacobian (2x2) with v_mix(h_mix), dv/dh = -1/v_mix (if v_mix > 0)
            state_mix = self.med_prop.calc_state("PH", p_mix, h_mix)
            drho_dp = self.med_prop.get_partial_derivative("D", "P", "H", state_mix)
            drho_dh = self.med_prop.get_partial_derivative("D", "H", "P", state_mix)
            if v_mix > 0:
                dv_dh = -1.0 / v_mix
            else:
                dv_dh = 0.0  # safeguard

            J = np.zeros((2, 2))
            # d(eq1)/d(p_mix)
            J[0, 0] = -A_mix * v_mix * drho_dp / self.m_flow_primary
            # d(eq1)/d(h_mix)
            J[0, 1] = -A_mix * (v_mix * drho_dh + rho_mix * dv_dh) / self.m_flow_primary

            denom_mom = self.state_primary_throat.p * A_throat + eta_mixing * self.m_flow_primary * self.v_throat
            # d(eq2)/d(p_mix)
            J[1, 0] = (-A_mix - A_mix * v_mix ** 2 * drho_dp) / denom_mom
            # d(eq2)/d(h_mix)
            J[1, 1] = (-A_mix * v_mix ** 2 * drho_dh + 2 * rho_mix * A_mix) / denom_mom  # using dv/dh = -1/v_mix

            # Scaling
            col_scale =  np.maximum(np.linalg.norm(J, axis=0), 1e-12)
            J_s = J / col_scale

            print(J_s)
            print(np.linalg.cond(J_s))

            f_vec = np.array(rel_err[-1], dtype=float)
            corrections = -np.linalg.solve(J_s, f_vec)
            corrections /= col_scale

            if 'prev_err' not in locals():
                prev_err = rel_err[-1]

            if np.any(np.sign(rel_err[-1]) != np.sign(prev_err)) or np.any(abs(np.array(rel_err[-1])) > abs(np.array(prev_err))):  # If the sign of any residual changed or any error increased, reduce the relaxation factor to prevent oscillations
                newton_relaxation_factor = max(0.1, newton_relaxation_factor * 0.8)
                print("Reducing relaxation factor to ", newton_relaxation_factor)
            else:
                newton_relaxation_factor = min(1.0, newton_relaxation_factor * 1.1)
                print("Increasing relaxation factor to ", newton_relaxation_factor)

            prev_err = rel_err[-1]

            # Bounds on pressure
            if p_mix + corrections[0] > self.state_primary.p:
                corrections[0] = self.state_primary.p - p_mix - 1e-6
            elif p_mix + corrections[0] < self.state_secondary_mixing.p:
                corrections[0] = self.state_secondary_mixing.p - p_mix + 1e-6

            if corrections[1] > diff_energy:
                corrections[1] = diff_energy - 1e-6  # prevent non-physical negative argument in sqrt
            elif corrections[1] < self.state_primary_throat.h - h_mix:
                corrections[1] = self.state_primary_throat.h - h_mix + 1e-6

            # Update variables
            p_mix += corrections[0] * newton_relaxation_factor
            h_mix += corrections[1] * newton_relaxation_factor


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