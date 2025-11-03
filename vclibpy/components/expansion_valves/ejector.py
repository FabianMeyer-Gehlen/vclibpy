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
    Ejector model according to Zhu 2017:
    'Theoretical model of a transcritical CO2 ejector
    with non-equilibrium phase change correlation'

    Assumptions:

    - flow inside ejector is steady and one-dimensional
    - ejector walls are adiabatic
    - inlet/outlet-flow velocity is neglected in the energy-conservation equation
    - polytropic equations are used for the flow except in the mixing process
    - mixing occurs completely in the mixing chamber and under constant pressure
    - the two-phase flow in the suction chamber, mixing chamber
      and diffuser is homogeneous and is in thermodynamic equilibrium
    - primary nozzle flow becomes sonic in nozzle throat
    - pressure of primary flow at mixing-chamber inlet is assumed equal to pressure at secondary ejector-inlet

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
        self.c_m = c_m
        self.phi_n = phi_n
        self.phi_d = phi_d
        self.v_step_min = v_step_min
        self.v_step_max = v_step_max
        self.state_throat: ThermodynamicState = None
        self.state_primary_mixing: ThermodynamicState = None  # Thermodynamic state of primary flow at mixing chamber
        self.state_mixing: ThermodynamicState = None  # Thermodynamic state of mixed flow at mixing chamber

    def calc_m_flow(self,
                    p_3,
                    p_throat_start=None,
                    v_secondary_mixing_start=None,
                    correlation=False,
                    QNE = False):

        if self.med_prop.fluid_name != "CarbonDioxide":
            raise NotImplementedError("Current ejector model only valid for CarbonDioxide")

        # Starting with calculations inside the primary nozzle
        # 1: Iterate Pressure inside nozzle throat
        self.state_throat, c_throat = self.iterate_throat_pressure(p_throat_start, correlation=correlation, QNE=QNE)
        self.c_throat = c_throat
        # 2: Now that we know p_throat, we can continue with calculating the mass flow through the primary nozzle
        self.m_flow_primary = math.pi*(self.d_throat*10**-3)**2/4*self.state_throat.d*c_throat
        #print(f"m_flow_primary = π * ({self.d_throat} * 10^-3)^2 / 4 * {self.state_throat.d} * {c_throat}")
        #print(f"Calculated m_flow_primary: {self.m_flow_primary*3600}kg/h")
        #print(self.state_throat)

        # Now we can do calculations for the inlet of the mixing-chamber
        # 3: Calculate enthalpy of primary flow at inlet of mixing-chamber
        h_primary_mixing_isentropic = self.med_prop.calc_state("PS", self.state_secondary.p, self.state_throat.s).h
        h_primary_mixing = self.state_throat.h - self.phi_n*(self.state_throat.h-h_primary_mixing_isentropic)
        self.state_primary_mixing = self.med_prop.calc_state("PH", self.state_secondary.p, h_primary_mixing)

        # 4: Calculate velocity of primary flow at mixing-chamber inlet using energy conservation
        v_primary_mixing = math.sqrt(c_throat**2 + 2*(self.state_throat.h-self.state_primary_mixing.h +
                                                      self.state_throat.p/self.state_throat.d -
                                                      self.state_primary_mixing.p/self.state_primary_mixing.d))
        #print(f"v_primary_mixing = {v_primary_mixing}")
        # 5: Calculate flow area occupied by primary nozzle flow at mixing-chamber inlet
        A_primary_mixing = self.m_flow_primary/(self.state_primary_mixing.d*v_primary_mixing)
        #print(f"A_mixing = {(self.d_mixing*0.001)**2/4*math.pi}")
        #print(f"A_primary_mixing = {A_primary_mixing}")
        # 6: Calculate flow area occupied by secondary nozzle flow at mixing chamber inlet
        A_secondary_mixing = math.pi/4*(self.d_mixing*0.001)**2 - A_primary_mixing
        #print(f"A_secondary_mixing = {A_secondary_mixing}")
        # Now we can calculate the mixing process and the flow in the diffuser
        # For this we have a nonlinear equation system we need to solve numerically
        # 7: Calculate secondary mass flow, thermodynamic state in mixing chamber and at outlet
        #    and velocities in mixing chamber
        self.iterate_v_secondary_mixing(A_secondary_mixing, v_primary_mixing, p_3, v_secondary_mixing_start)
        #print(self.m_flow_primary*3600, self.m_flow_secondary*3600)
        self.m_flow_outlet = self.m_flow_primary + self.m_flow_secondary
        return self.m_flow_outlet

    def iterate_v_secondary_mixing(self,
                                   A_secondary_mixing,
                                   v_primary_mixing,
                                   p_3,
                                   v_secondary_mixing_start = None):  # Starting value for v_secondary_mixing
        # We start the solving process by guessing the speed of the secondary flow at the mixing chamber inlet
        if v_secondary_mixing_start is None:
            v_secondary_mixing_start = 40.00 #ToDO: find out how to guess v_sm (about 40 m/s in graphs from Zhu 2017)

        # Setup for iteration
        error_h_outlet_history = []
        num_iterations = 0
        iteration_multiplier = 0
        v_secondary_mixing = v_secondary_mixing_start
        if self.use_quick_solver:
            step_v_sm = self.v_step_max
        else:
            step_v_sm = self.v_step_min

        # Now we can calculate the missing values and update v_secondary until it converges
        while True:
            num_iterations += 1
            if isinstance(self.max_num_iterations, (int, float)):
                if num_iterations > self.max_num_iterations:
                    logger.warning("Maximum number of iterations %s exceeded, while iterating v_sm. Stopping.",
                                   self.max_num_iterations)
                    return

                if (num_iterations + 1) % (0.1 * self.max_num_iterations) == 0:
                    logger.info("Info: %s percent of max_num_iterations %s for iterating v_sm used",
                                100 * (num_iterations + 1) / self.max_num_iterations, self.max_num_iterations)
            # Check if v_secondary_mixing is negative, if so, stop and raise error
            if v_secondary_mixing < 0:
                logger.error("v_secondary_mixing is negative - Configuration is infeasible. Stopping.") #  ToDO: Check what happens if v_secondary_mixing is close to zero
                self.m_flow_secondary = None
                return

            self.m_flow_secondary = A_secondary_mixing*self.state_secondary.d*v_secondary_mixing
            phi_m = self.c_m - 0.34*math.exp(-3.125*self.m_flow_secondary/self.m_flow_primary)
            v_mixing = (phi_m*(self.m_flow_primary*v_primary_mixing + self.m_flow_secondary*v_secondary_mixing)/
                        (self.m_flow_primary+self.m_flow_secondary))
            h_mixing = ((self.m_flow_primary*(self.state_primary_mixing.h+0.5*v_primary_mixing**2) +
                        self.m_flow_secondary*(self.state_secondary.h+0.5*v_secondary_mixing**2)) /
                        (self.m_flow_primary + self.m_flow_secondary) - 0.5*v_mixing**2)
            self.state_mixing = self.med_prop.calc_state("PH", self.state_secondary.p, h_mixing)
            h_outlet_isentropic = self.med_prop.calc_state("PS", p_3, self.state_mixing.s).h
            h_outlet = h_mixing - self.phi_d*(h_mixing-h_outlet_isentropic)  # ToDO: Check with Zhu 2017 if this is correct
            h_outlet_2 = ((self.m_flow_primary * self.state_primary.h +
                          self.m_flow_secondary * self.state_secondary.h) /
                          (self.m_flow_primary + self.m_flow_secondary))
            error_h_outlet = (h_outlet_2/h_outlet-1)*100
            error_h_outlet_history.append(error_h_outlet)
            #print(f"Iteration {num_iterations}: v_secondary_mixing = {v_secondary_mixing}; v_mixing = {v_mixing}; error_h_outlet = {error_h_outlet}")
            #return error_h_outlet

            if abs(error_h_outlet) > self.max_err or step_v_sm > self.v_step_min:
                # In the first two iteration steps we need to find out in which direction to adjust v_secondary_mixing
                # if num_iterations == 1:
                #     # Assume, that we have to increase v_secondary_mixing to decrease the value of the error
                #     if error_h_outlet > 0:
                #         v_secondary_mixing += step_v_sm
                #         continue
                #     else:
                #         v_secondary_mixing -= step_v_sm
                #         continue
                # elif num_iterations == 2:
                #     factor_error = error_h_outlet_history[-1] / error_h_outlet_history[-2]
                #     # Test whether assumption was right, otherwise change direction of p_throat-adjustments
                #     if factor_error > 1:
                #         iteration_multiplier = -1
                #     else:
                #         iteration_multiplier = 1
                #
                #     if error_h_outlet > 0:
                #         v_secondary_mixing += iteration_multiplier * step_v_sm
                #         continue
                #     else:
                #         v_secondary_mixing -= iteration_multiplier * step_v_sm
                #         continue
                # else:
                if num_iterations > 2 and numpy.sign(error_h_outlet_history[-1]) != numpy.sign(error_h_outlet_history[-2]):
                    # If there was a sign change in the error, decrease the step size
                    step_v_sm /= 10

                if error_h_outlet > 0:
                    v_secondary_mixing += step_v_sm
                    continue
                else:
                    v_secondary_mixing -= step_v_sm
                    continue
            else:
                # The error and step_v_sm are smaller than max_error and v_step_min. We can break.
                self.state_outlet = self.med_prop.calc_state("PH", p_3, h_outlet)
                #print(f"v_secondary_mixing = {v_secondary_mixing}; converged after {num_iterations} iterations with error: {error_h_outlet}")
                return

    def iterate_throat_pressure(self, p_throat_start=None, correlation=False, QNE=False):
        use_correlation = False  # Flag to indicate whether to use correlation without iterating p_throat
        if p_throat_start is None:
            # Use correlation from paper to guess first value of p_throat
            p_throat_start = 1e6/(0.53 - 0.121*(self.state_primary.p/1e6)**0.5 + 6e-11*self.state_primary.d**3)
            if (7.6e6 < self.state_primary.p < 10.3e6
                    and 32+273.15 < self.state_primary.T < 43.4+273.15
                    and 300 < self.state_primary.d < 750 and correlation):
                use_correlation = True  # Within the boundary in the if-statement the correlation is valid
        self.p_throat = p_throat_start
        if correlation and not (7.6e6 < self.state_primary.p < 10.3e6 and 2+273.15 < self.state_primary.T < 43.4+273.15 and 300 < self.state_primary.d < 750):
            logger.warning("p_throat correlation not valid. Using iteration instead.")
        #print(f"p_throat_start: {p_throat_start}")
        # Setup for iteration
        error_h_throat_history = []
        num_iterations = 0
        iteration_multiplier = 0
        if self.use_quick_solver:
            step_p_throat = self.step_max
        else:
            step_p_throat = self.min_iteration_step

        while True:
            num_iterations += 1
            # print(f"Iteration: {num_iterations}; Throat Pressure: {self.p_throat}")
            if isinstance(self.max_num_iterations, (int, float)):
                if num_iterations > self.max_num_iterations:
                    logger.warning("Maximum number of iterations %s exceeded, while iterating p_throat. Stopping.",
                                   self.max_num_iterations)
                    return

                if (num_iterations + 1) % (0.1 * self.max_num_iterations) == 0:
                    logger.info("Info: %s percent of max_num_iterations %s for iterating p_throat used",
                                100 * (num_iterations + 1) / self.max_num_iterations, self.max_num_iterations)

            # 1a: Calculate specific enthalpy inside Nozzle throat from isentropic efficiency
            h_throat_isentropic = self.med_prop.calc_state("PS", self.p_throat, self.state_primary.s).h
            h_throat = self.state_primary.h - self.phi_n*(self.state_primary.h - h_throat_isentropic) #ToDO: Wirkungsgrade auf eta ändern

            # 1b: Calculate speed of sound for two phase flow inside throat
            # State calculation inside throat
            state_throat = self.med_prop.calc_state("PH", self.p_throat, h_throat)

            if not 0 < state_throat.q < 1:
                logger.warning("Nozzle throat state not inside two-phase region. Stopping")  # ToDO: Error handling
                #return -1

            state_throat_vapor = self.med_prop.calc_state("PQ", self.p_throat, 1)
            state_throat_liquid = self.med_prop.calc_state("PQ", self.p_throat, 0)
            # Volume fractions inside throat
            phi_throat_vapor = state_throat.q*state_throat.d/state_throat_vapor.d
            phi_throat_liquid = 1 - phi_throat_vapor
            # Speed of sound for separate phases
            a_liquid = self.med_prop.get_saturated_speed_of_sound(self.p_throat, False)
            a_vapor = self.med_prop.get_saturated_speed_of_sound(self.p_throat, True)
            # Specific heat capacities
            c_p_liquid = self.med_prop.calc_transport_properties(state_throat_liquid).cp
            c_p_vapor = self.med_prop.calc_transport_properties(state_throat_vapor).cp
            # Extensive heat capacities
            C_p_liquid = state_throat_liquid.d * phi_throat_liquid * c_p_liquid
            C_p_vapor = state_throat_vapor.d * phi_throat_vapor * c_p_vapor  # ToDO: Checken ob bei Benutzen von correlation die Berechnung von c_throat mit v_throat_2 übereinstimmt
            # To calculate Zeta we need a partial differential from medProp.
            # CoolProp can calculate this directly,
            # but with RefProp we have to transform it to values we can get using Maxwell's equations:
            if isinstance(self.med_prop, vclibpy.media.cool_prop.CoolProp):
                zeta_vapor = self.med_prop.get_partial_derivative("T", "P", "S", state_throat_vapor)
                zeta_liquid = self.med_prop.get_partial_derivative("T", "P", "S", state_throat_liquid)
            else:
                # Thermal expansion coefficients
                beta_liquid = self.med_prop.calc_transport_properties(state_throat_liquid).beta
                beta_vapor = self.med_prop.calc_transport_properties(state_throat_vapor).beta
                zeta_liquid = state_throat.T*beta_liquid/(state_throat_liquid.d*c_p_liquid)
                zeta_vapor = state_throat.T*beta_vapor/(state_throat_vapor.d*c_p_vapor)
            x_1 = (phi_throat_vapor/(state_throat_vapor.d*a_vapor**2) +
                   phi_throat_liquid/(state_throat_liquid.d*a_liquid**2))
            x_2 = C_p_vapor*C_p_liquid*(zeta_liquid-zeta_vapor)**2 / (C_p_vapor+C_p_liquid)
            # Speed of sound for two phase flow
            c_throat = (state_throat.d*x_1 + (state_throat.d/state_throat.T)*x_2)**-0.5  #
            # 1c: Calculate specific enthalpy using energy balance - assume Ma_throat = 1
            Q_NE = 8629*math.e**(-10*(state_throat.q))  # Correlation for non-equilibrium phase change
            #print(f"liquid mass fraction: {1-state_throat.q}, Q_NE = {Q_NE}")
            if QNE:
                h_throat_2 = self.state_primary.h - c_throat**2/2 + Q_NE
            else:
                h_throat_2 = self.state_primary.h - c_throat**2/2

            # Calculate error between two calculations for h_throat
            error_h_throat = (h_throat_2/h_throat-1)*100
            error_h_throat_history.append(error_h_throat)
            #return error_h_throat
            #print(error_h_throat)
            if use_correlation:
                self.c_throat2 = c_throat
                v_throat_2 = math.sqrt(2 * (self.state_primary.h - h_throat + Q_NE))
                #print(c_throat, v_throat_2, v_throat_2/c_throat-1)
                return state_throat, v_throat_2
            if abs(error_h_throat) > self.max_err or step_p_throat > self.min_iteration_step:
                # In the first two iteration steps we need to find out in which direction to adjust p_throat
                if num_iterations == 1:
                    # Assume, that we have to increase p_throat to decrease the value of the error
                    if error_h_throat > 0:
                        self.p_throat += self.min_iteration_step
                        continue
                    else:
                        self.p_throat -= self.min_iteration_step
                        continue
                elif num_iterations == 2:
                    factor_error = error_h_throat_history[-1]/error_h_throat_history[-2]
                    # Test whether assumption was right, otherwise change direction of p_throat-adjustments
                    if factor_error > 1:
                        iteration_multiplier = -1
                    else:
                        iteration_multiplier = 1

                    if error_h_throat > 0:
                        self.p_throat += iteration_multiplier*step_p_throat
                        continue
                    else:
                        self.p_throat -= iteration_multiplier*step_p_throat
                        continue
                else:
                    if numpy.sign(error_h_throat_history[-1]) != numpy.sign(error_h_throat_history[-2]):
                        # If there was a sign change in the error, decrease the step size
                        step_p_throat /= 10

                    if error_h_throat > 0:
                        self.p_throat += iteration_multiplier*step_p_throat
                        continue
                    else:
                        self.p_throat -= iteration_multiplier*step_p_throat
                        continue
            else:
                # The error and step_p_throat are smaller than max_error and min_iteration_step. We can break.
                v_throat_2 = math.sqrt(2 * (self.state_primary.h - h_throat + Q_NE))
                #print(f"Converged after {num_iterations} iterations")
                #print(state_throat.q, Q_NE)
                return state_throat, c_throat
