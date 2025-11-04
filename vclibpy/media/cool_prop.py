"""
Module with cool prop wrapper.
"""
import logging

import CoolProp.CoolProp as CoolPropInternal

from vclibpy.media.media import MedProp
from vclibpy.media.states import ThermodynamicState, TransportProperties

logger = logging.getLogger(__name__)


class CoolProp(MedProp):
    """
    Class using the open-source CoolProp package
    to access the properties.

    Args:
        use_high_level_api (bool):
            True to use the high-level api, which is much slower,
            but you can use all modes in calc_state.
            Default is False.
    """

    _mode_map = {
        "PT": (CoolPropInternal.PT_INPUTS, True),
        "TQ": (CoolPropInternal.QT_INPUTS, False),
        "PS": (CoolPropInternal.PSmass_INPUTS, True),
        "PH": (CoolPropInternal.HmassP_INPUTS, False),
        "PQ": (CoolPropInternal.PQ_INPUTS, True)
    }

    _state_function_map = {
        "P": CoolPropInternal.iP,  # Pressure
        "T": CoolPropInternal.iT,  # Temperature
        "D": CoolPropInternal.iDmass,  # Density (mass-based)
        "H": CoolPropInternal.iHmass,  # Enthalpy (mass-based)
        "S": CoolPropInternal.iSmass,  # Entropy (mass-based)
        "U": CoolPropInternal.iUmass,  # Internal Energy (mass-based)
        "q": CoolPropInternal.iQ,      # Vapor Quality (mass-based)
    } # additional states might be added later (check CoolProp documentation for available properties)

    def __init__(self, fluid_name, use_high_level_api: bool = False):
        super().__init__(fluid_name=fluid_name)
        # Set molar mass and trigger a possible fluid-name error
        # if the fluid is not supported.
        self._helmholtz_equation_of_state = CoolPropInternal.AbstractState("HEOS", self.fluid_name)
        self.M = self._helmholtz_equation_of_state.molar_mass()
        self.use_high_level_api = use_high_level_api

    def calc_state(self, mode: str, var1: float, var2: float):
        super().calc_state(mode=mode, var1=var1, var2=var2)

        if self.use_high_level_api:
            _var1_str, _var2_str = mode[0], mode[1]
            # CoolProp returns Pa
            p = CoolPropInternal.PropsSI('P', _var1_str, var1, _var2_str, var2, self.fluid_name)
            # CoolProp returns kg/m^3
            d = CoolPropInternal.PropsSI('D', _var1_str, var1, _var2_str, var2, self.fluid_name)
            # CoolProp returns K
            T = CoolPropInternal.PropsSI('T', _var1_str, var1, _var2_str, var2, self.fluid_name)
            # CoolProp returns J/kg
            u = CoolPropInternal.PropsSI('U', _var1_str, var1, _var2_str, var2, self.fluid_name)
            # CoolProp returns J/kg
            h = CoolPropInternal.PropsSI('H', _var1_str, var1, _var2_str, var2, self.fluid_name)
            # CoolProp returns J/kg/K
            s = CoolPropInternal.PropsSI('S', _var1_str, var1, _var2_str, var2, self.fluid_name)
            # CoolProp returns mol/mol
            q = CoolPropInternal.PropsSI('Q', _var1_str, var1, _var2_str, var2, self.fluid_name)
            # Return new state
            return ThermodynamicState(p=p, T=T, u=u, h=h, s=s, d=d, q=q)

        self._update_coolprop_heos(mode=mode, var1=var1, var2=var2)
        # Return new state
        return ThermodynamicState(
            p=self._helmholtz_equation_of_state.p(),
            T=self._helmholtz_equation_of_state.T(),
            u=self._helmholtz_equation_of_state.umass(),
            h=self._helmholtz_equation_of_state.hmass(),
            s=self._helmholtz_equation_of_state.smass(),
            d=self._helmholtz_equation_of_state.rhomass(),
            q=self._helmholtz_equation_of_state.Q()
        )

    def calc_transport_properties(self, state: ThermodynamicState):
        if 0 <= state.q <= 1:
            mode = "PQ"
            var1, var2 = state.p, state.q
        else:
            # Get using p and T
            mode = "PT"
            var1, var2 = state.p, state.T

        if self.use_high_level_api:
            args = [mode[0], var1, mode[1], var2, self.fluid_name]
            # CoolProp returns -
            pr = CoolPropInternal.PropsSI('PRANDTL', *args)
            # CoolProp returns J/kg/K
            cp = CoolPropInternal.PropsSI('C', *args)
            # CoolProp returns J/kg/K
            cv = CoolPropInternal.PropsSI('CVMASS', *args)
            # CoolProp returns W/m/K
            lam = CoolPropInternal.PropsSI('CONDUCTIVITY', *args)
            # CoolProp returns Pa*s
            dyn_vis = CoolPropInternal.PropsSI('VISCOSITY', *args)
            # Internal calculation as kinematic vis is ration of dyn_vis to density
            # In m^2/s
            kin_vis = dyn_vis / state.d

            # Create transport properties instance
            return TransportProperties(lam=lam, dyn_vis=dyn_vis, kin_vis=kin_vis,
                                       pr=pr, cp=cp, cv=cv, state=state)
        # Low-level API
        self._update_coolprop_heos(mode=mode, var1=var1, var2=var2)
        # Create transport properties instance
        return TransportProperties(
            lam=self._helmholtz_equation_of_state.conductivity(),
            dyn_vis=self._helmholtz_equation_of_state.viscosity(),
            kin_vis=self._helmholtz_equation_of_state.viscosity() / state.d,
            pr=self._helmholtz_equation_of_state.Prandtl(),
            cp=self._helmholtz_equation_of_state.cpmass(),
            cv=self._helmholtz_equation_of_state.cvmass(),
            state=state
        )

    def _update_coolprop_heos(self, mode, var1, var2):
        if mode not in self._mode_map:
            raise KeyError(
                f"Given mode '{mode}' is currently not supported with the "
                f"faster low-level API to cool-prop. "
                f"Either use the high-level API or raise an issue. "
                f"Supported modes: {', '.join(self._mode_map.keys())}"
            )
        i_input, not_reverse_variables = self._mode_map[mode]
        if not_reverse_variables:
            self._helmholtz_equation_of_state.update(i_input, var1, var2)
        else:
            self._helmholtz_equation_of_state.update(i_input, var2, var1)

    def get_critical_point(self):
        Tc = CoolPropInternal.PropsSI("TCRIT", self.fluid_name)
        pc = CoolPropInternal.PropsSI("PCRIT", self.fluid_name)
        dc = CoolPropInternal.PropsSI("RHOCRIT", self.fluid_name)
        return Tc, pc, dc

    def get_molar_mass(self):
        return self.M

    def get_speed_of_sound(self, state: ThermodynamicState) -> float:
        """
        Calculate the speed of sound for given thermodynamic state.

        Parameters:
            state (ThermodynamicState): Thermodynamic state

        Returns:
            a (float): Speed of sound in m/s
        """
        if 0 <= state.q <= 1:
            raise ValueError("Speed of sound in two-phase region is not defined. Use get_saturated_speed_of_sound or get_two_phase_speed_of_sound methods.")
        else:
            if self.use_high_level_api:
                a = CoolPropInternal.PropsSI('A', 'P', state.p, 'T', state.T, self.fluid_name)
            else:
                self._update_coolprop_heos("PT", state.p, state.T)
                a = self._helmholtz_equation_of_state.speed_sound()
        return a

    def get_saturated_speed_of_sound(self, p, vapor: bool):
        """
        Calculate the speed of sound for saturated points based on pressure and quality.

        Parameters:
            p (float): Pressure in Pa
            vapor (boolean): Boolean to calculate the speed of sound for either saturated vapour or liquid

        Returns:
            a (float): Speed of sound in m/s
        """

        if vapor:
            q = 1
        else:
            q = 0
        a = CoolPropInternal.PropsSI('A', 'P', p, 'Q', q, self.fluid_name)
        return a

    def get_two_phase_speed_of_sound(self, p: float, q: float) -> float:
        state_two_phase = self.calc_state("PQ", p, q)
        state_vapor = self.calc_state("PQ", p, 1)
        state_liquid = self.calc_state("PQ", p, 0)

        # compute partial derivatives for enthalpy and specific volume
        dh_dpq_v = self.get_partial_derivative("H", "P", "q", state_vapor)
        dh_dpq_l = self.get_partial_derivative("H", "P", "q", state_liquid)
        del_h = q * dh_dpq_v + (1 - q) * dh_dpq_l

        dD_dpq_v = self.get_partial_derivative("D", "P", "q", state_vapor)
        dD_dpq_l = self.get_partial_derivative("D", "P", "q", state_liquid)
        term_v = q * (-1.0) / state_vapor.d ** 2 * dD_dpq_v
        term_l = (1 - q) * (-1.0) / state_liquid.d ** 2 * dD_dpq_l
        del_v = term_v + term_l

        numerator = state_two_phase.v ** 2 * (state_vapor.h - state_liquid.h)
        denominator = ((state_vapor.v - state_liquid.v) * (del_h - state_two_phase.v) - del_v * (
                    state_vapor.h - state_liquid.h))
        c_attou = (numerator / denominator) ** 0.5
        return c_attou

    def get_partial_derivative(self, numerator: str, denominator: str, constant: str, state: ThermodynamicState, ):
        if {numerator, denominator, constant} - self._state_function_map.keys():
            raise ValueError("Invalid property name! Use one of: " + ", ".join(self._state_function_map.keys()))

        self._update_coolprop_heos("PH", state.p, state.h)
        numerator_code = self._state_function_map[numerator]
        denominator_code = self._state_function_map[denominator]
        constant_code = self._state_function_map[constant]

        if constant_code is CoolPropInternal.iQ:
            return self._helmholtz_equation_of_state.first_saturation_deriv(numerator_code, denominator_code)
        else:
            if 0<=state.q<=1:
                try:
                    return self._helmholtz_equation_of_state.first_two_phase_deriv(numerator_code, denominator_code, constant_code)
                except ValueError as err:
                    logger.info("Could not use low level Interface for calculation. Trying high level interface: %s",err)
                    return CoolPropInternal.PropsSI(f'd({numerator})/d({denominator})|{constant}', 'P', state.p, 'Q', state.q, self.fluid_name)

            else:
                return self._helmholtz_equation_of_state.first_partial_deriv(numerator_code, denominator_code, constant_code)

if __name__ == '__main__':
    CoolProp("Propan")
