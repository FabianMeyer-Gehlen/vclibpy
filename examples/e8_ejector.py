# # Example for standard ejector cycle
import logging

import vclibpy.components.heat_exchangers
from vclibpy.media import MedProp
import numpy
from vclibpy.components.expansion_valves.ejector import Ejector
from vclibpy.media import CoolProp, RefProp
from vclibpy.utils.plotting import plot_cycle
from vclibpy.flowsheets.standard_ejector_cycle import StandardEjectorCycle
from vclibpy.media import ThermodynamicState
from typing import List
import pathlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import plotly.graph_objects as go
import math
import pandas as pd
import openpyxl
import multiprocessing

try:
    plt.style.use('D:/kbr-fme/ebc.paper.mplstyle')
except OSError:
    logging.warning("Could not load the custom matplotlib style file. Using default style.")
    print("Could not load the custom matplotlib style file. Using default style.")

mpl.rcParams['svg.fonttype'] = 'none'  # make pyplot save text as text and not paths, so it is editable in Inkscape

med_prop = CoolProp(fluid_name="CarbonDioxide")
ejector = Ejector(0.8, 2, use_quick_solver=True)  # Ejector 1
ejector_2 = Ejector(1, 2.6, use_quick_solver=True)  # Ejector 2
ejector.med_prop = med_prop
ejector_2.med_prop = med_prop


def main(use_condenser_inlet: bool = True):
    # m_flow_calculation('J:/Massenstromberechnungen/Daten_Ejektor_1.xlsx')
    # error_calculation_p_throat()
    # p_throat_iteration()
    # calc_single_ejector_state()
    # test_standard_ejector_cycle()
    # error_calculation_v_secondary_mixing()
    # qne_plot()
    sound_speed_calculation()

def test_standard_ejector_cycle():
    from vclibpy.components.heat_exchangers import moving_boundary_ntu
    from vclibpy.components.heat_exchangers import heat_transfer
    from vclibpy.components.expansion_valves import Bernoulli
    from vclibpy.components.compressors import RotaryCompressor
    from vclibpy.components.phase_separator import PhaseSeparator
    from vclibpy.datamodels import Inputs, FlowsheetState, RelativeCompressorSpeedControl, HeatExchangerInputs
    from vclibpy.algorithms.iteration import Iteration

    condenser = moving_boundary_ntu.MovingBoundaryNTUCondenser(
        A=30,
        secondary_medium="air",
        flow_type="cross",
        ratio_outer_to_inner_area=10,
        two_phase_heat_transfer=heat_transfer.constant.ConstantTwoPhaseHeatTransfer(alpha=1000),
        gas_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=1000),
        wall_heat_transfer=heat_transfer.wall.WallTransfer(lambda_=236, thickness=2e-3),
        liquid_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=5000),
        secondary_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=25)
    )
    evaporator = moving_boundary_ntu.MovingBoundaryNTUEvaporator(
        A=30,
        secondary_medium="air",
        flow_type="cross",
        ratio_outer_to_inner_area=10,
        two_phase_heat_transfer=heat_transfer.constant.ConstantTwoPhaseHeatTransfer(alpha=1000),
        gas_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=1000),
        wall_heat_transfer=heat_transfer.wall.WallTransfer(lambda_=236, thickness=2e-3),
        liquid_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=5000),
        secondary_heat_transfer=heat_transfer.constant.ConstantHeatTransfer(alpha=25)
    )
    expansion_valve = Bernoulli(A=0.01)
    compressor = RotaryCompressor(
        N_max=125,
        V_h=19e-6
    )
    phase_seperator = PhaseSeparator()

    flowsheet = StandardEjectorCycle(
        evaporator=evaporator,
        condenser=condenser,
        fluid="CarbonDioxide",
        compressor=compressor,
        ejector=ejector_2,
        metering_valve=expansion_valve,
        phase_seperator=phase_seperator
    )
    save_path = pathlib.Path(r"D:\00_temp\standard_ejector_cycle")
    algorithm = Iteration(raise_errors=True, save_path_plots=save_path)
    speed_control = RelativeCompressorSpeedControl(0.2, 0.0, 5.0)
    eva_inputs = HeatExchangerInputs(T_in=-5 + 273.15, m_flow=1.00842)
    con_inputs = HeatExchangerInputs(T_in=15 + 273.15, m_flow=1.00842)
    # print(med_prop.calc_state("TQ", 279.65, 1))
    inputs = Inputs(control=speed_control, evaporator=eva_inputs, condenser=con_inputs)
    p_1_start, p_2_start, p_max, fs_state = algorithm.initial_setup(flowsheet, fluid=None, inputs=inputs)
    #flowsheet.calc_states(1e6, 6e6, inputs, fs_state)
    for i in range(10):
        speed_control = RelativeCompressorSpeedControl(0.2, 0.0, dT_con_subcooling=i)
        inputs = Inputs(control=speed_control, evaporator=eva_inputs, condenser=con_inputs)
        fs_state = algorithm.calc_steady_state(flowsheet, inputs, "CarbonDioxide", )
    # plot_cycle(flowsheet.med_prop, flowsheet.get_states_in_order_for_plotting(), show=True)
        print(f"Subcooling: {i} K")
        print(f"Compressor:")
        print(f"m_flow = {flowsheet.compressor.m_flow * 3600} kg/h")
        print(f"state_inlet = {flowsheet.compressor.state_inlet}")
        print(f"state_outlet = {flowsheet.compressor.state_outlet}")
        print(f"Ejector:")
        print(f"m_flow_outlet = {flowsheet.ejector.m_flow_outlet * 3600} kg/h, m_flow_primary = {flowsheet.ejector.m_flow_primary * 3600} kg/h, m_flow_secondary = {flowsheet.ejector.m_flow_secondary * 3600} kg/h")
        print(f"state_outlet = {flowsheet.ejector.state_outlet}")
        print(f"state_primary = {flowsheet.ejector.state_primary}")
        print(f"Condenser:")
        print(f"m_flow = {flowsheet.condenser.m_flow * 3600} kg/h")
        print(f"state_inlet = {flowsheet.condenser.state_inlet}")
        print(f"state_outlet = {flowsheet.condenser.state_outlet}")
        print(f"Evaporator:")
        print(f"m_flow = {flowsheet.evaporator.m_flow * 3600} kg/h")
        print(f"state_inlet = {flowsheet.evaporator.state_inlet}")
        print(f"state_outlet = {flowsheet.evaporator.state_outlet}")

        # Get all possible values:
        variables_to_excel = [{
            **fs_state.convert_to_str_value_format(with_unit_and_description=False),
        }]

        # Save to excel
        save_path_csv = save_path.joinpath(f"{flowsheet.flowsheet_name}_{flowsheet.fluid}_{i}.csv")
        pd.DataFrame(variables_to_excel).to_csv(
            save_path_csv, sep=","
        )



def calc_single_ejector_state():
    ejector_2.state_primary = med_prop.calc_state("PT", 9e6, 35 + 273.15)
    ejector_2.state_secondary = med_prop.calc_state("PQ", 3e6, 1)

    p3_list = numpy.arange(3.45, 3.8, 0.01)
    m_flow_p_list = []
    m_flow_s_list = []
    m_flow_outlet_list = []
    m_flow_outlet_vapor_list = []
    m_flow_outlet_liquid_list = []

    for p_3 in p3_list:
        ejector_2.calc_m_flow(p_3 * 1e6, correlation=True)
        m_flow_p_list.append(ejector_2.m_flow_primary * 3600)
        m_flow_s_list.append(ejector_2.m_flow_secondary * 3600)
        m_flow_outlet_list.append(ejector_2.m_flow_outlet * 3600)
        m_flow_outlet_vapor_list.append(ejector_2.m_flow_outlet * ejector_2.state_outlet.q * 3600)
        m_flow_outlet_liquid_list.append(ejector_2.m_flow_outlet * (1-ejector_2.state_outlet.q) * 3600)
        print(f"p_3 = {p_3} MPa")
        print(f"mdot_p = {ejector_2.m_flow_primary * 3600} kg/h, mdot_s = {ejector_2.m_flow_secondary * 3600} kg/h, mdot_outlet = {ejector_2.m_flow_outlet * 3600} kg/h")
        print(f"mdot_outlet_vapor = {ejector_2.m_flow_outlet * ejector_2.state_outlet.q * 3600} kg/h, mdot_outlet_liquid = {ejector_2.m_flow_outlet * (1-ejector_2.state_outlet.q) * 3600} kg/h")

    plt.figure(figsize=(6.125, 6.125/4*2))
    plt.subplot(1, 1, 1)
    colorcycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.plot(p3_list, m_flow_p_list, label='mdot_p', zorder=10, color=colorcycle[0])
    plt.plot(p3_list, m_flow_s_list, label='mdot_s', zorder=10, color=colorcycle[13])
    plt.plot(p3_list, m_flow_outlet_vapor_list, label='mdot_outlet_vapor', zorder=10, color=colorcycle[0], linestyle='--')
    plt.plot(p3_list, m_flow_outlet_liquid_list, label='mdot_outlet_liquid', zorder=10, color=colorcycle[13], linestyle='--')
    plt.xlabel('p_3 in MPa')
    plt.ylabel('Mass flow in kg/h')
    plt.grid(visible=True, which='both', zorder=0)
    plt.legend(ncol=2)
    plt.show()


    # print(ejector.calc_m_flow(p_3=2.5e6, correlation=False) * 3600)
    #
    # states = [ejector.state_primary,
    #           ejector.state_throat,
    #           ejector.state_primary_mixing,
    #           ejector.state_mixing,
    #           ejector.state_outlet,
    #           ejector.state_mixing,
    #           ejector.state_secondary]
    # print(states)
    # for state in states:
    #     print(state)
    # plot_cycle(med_prop, states, show=True)


def m_flow_calculation(excel_path: str):
    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Drop rows with NaN values
    df = df.dropna(subset=['P_p', 'T_p', 'P_s', 'T_s', 'P_b', 'T_b', 'm_s', 'm_p'])

    pp_list = df['P_p'].tolist()
    Tp_list = df['T_p'].tolist()
    ps_list = df['P_s'].tolist()
    Ts_list = df['T_s'].tolist()
    pb_list = df['P_b'].tolist()
    Tb_list = df['T_b'].tolist()
    m_s_list = df['m_s'].tolist()
    m_p_list = df['m_p'].tolist()
    m_flow_list = [[0]*2 for i in range(len(pp_list))]


    for i in range(len(pp_list)):
        print(f"Calculating state for pressure: {pp_list[i]} MPa")
        ejector.state_primary = med_prop.calc_state("PT", pp_list[i] * 1e6, Tp_list[i] + 273.15)
        ejector.state_secondary = med_prop.calc_state("PT", ps_list[i] * 1e6, Ts_list[i] + 273.15)
        ejector.calc_m_flow(pb_list[i] * 1e6, correlation=True)
        m_flow_list[i][0] = ejector.m_flow_primary * 3600
        m_flow_list[i][1] = ejector.m_flow_secondary * 3600

    # Add the m_flow_list values to a new column in the DataFrame
    df['m_p_iteration_QNE_2500'] = [m[0] for m in m_flow_list]
    df['m_s_iteration_QNE_2500'] = [m[1] for m in m_flow_list]

    # with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
    #     df.to_excel(writer, index=False, sheet_name='Tabelle1')


    # Erstellen von Grafik mit berechnetem Treibmassenstrom im Vergleich zu Messwerten
    # plt.figure(figsize=(6.125, 6.125/4*2))
    #
    # # Erste Subfigure: Treibmassenstrom
    # plt.subplot(1, 2, 1)
    # plt.scatter(pp_list, [m[0] for m in m_flow_list], label='berechnet', zorder=10m s=)
    # plt.scatter(pp_list, m_p_list, label='gemessen', zorder=10)
    # plt.xlabel('Primärdruck in MPa')
    # plt.ylabel('Treibmassenstrom in kg/h')
    # plt.grid(visible=True, which='both', zorder=0)
    # plt.legend()
    #
    # # Zweite Subfigure: Relativer Fehler
    # plt.subplot(1, 2, 2)
    # relative_error = [(calculated - measured) / measured * 100 for calculated, measured in zip([m[0] for m in m_flow_list], m_p_list)]
    # plt.scatter(pp_list, relative_error, label='Relativer Fehler', zorder=10)
    # plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Null-Linie
    # plt.grid(visible=True, which='both', zorder=0)
    # plt.xlabel('Primärdruck in MPa')
    # plt.ylabel('Relativer Fehler in %')
    # plt.legend()

    # Erstellen von Grafik mit berechnetem Sekundärmassenstrom im Vergleich zu Messwerten
    plt.figure(figsize=(6.125, 6.125/4*2.5))

    # Erste Subfigure: Sekundärmassenstrom
    plt.subplot(1, 2, 1)
    plt.scatter(pp_list[0:10], [m[1] for m in m_flow_list[0:10]], label='p_d=4 MPa', zorder=10)
    plt.scatter(pp_list[11:17], [m[1] for m in m_flow_list[11:17]], label='p_d=4,1 MPa', zorder=10)
    plt.scatter(pp_list[18:28], [m[1] for m in m_flow_list[18:28]], label='p_d=4,2 MPa', zorder=10)
    plt.scatter(pp_list[29:42], [m[1] for m in m_flow_list[29:42]], label='p_d=4,3 MPa', zorder=10)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.scatter(pp_list[0:10], m_s_list[0:10], label='gemessen', zorder=9, c=color_cycle[0], marker='v', s=50)
    plt.scatter(pp_list[11:17], m_s_list[11:17], label='gemessen', zorder=9, c=color_cycle[1], marker='v', s=50)
    plt.scatter(pp_list[18:28], m_s_list[18:28], label='gemessen', zorder=9, c=color_cycle[2], marker='v', s=50)
    plt.scatter(pp_list[29:42], m_s_list[29:42], label='gemessen', zorder=9, c=color_cycle[3], marker='v', s=50)
    plt.xlabel('Primärdruck in MPa')
    plt.ylabel('Saugmassenstrom in kg/h')
    plt.grid(visible=True, which='both', zorder=0)
    # plt.legend()

    # Legende über den Subplots
    # plt.legend(loc='upper center', ncol=4)  # , bbox_to_anchor=(1, 1.20)

    # Zweite Subfigure: Relativer Fehler
    plt.subplot(1, 2, 2)
    relative_error = [(calculated - measured) / measured * 100 for calculated, measured in zip([m[1] for m in m_flow_list], m_s_list)]
    plt.scatter(pp_list[0:10], relative_error[0:10], label='p_d=4 MPa', zorder=10)
    plt.scatter(pp_list[11:17], relative_error[11:17], label='p_d=4,1 MPa', zorder=10)
    plt.scatter(pp_list[18:28], relative_error[18:28], label='p_d=4,2 MPa', zorder=10)
    plt.scatter(pp_list[29:42], relative_error[29:42], label='p_d=4,3 MPa', zorder=10)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Null-Linie
    plt.grid(visible=True, which='both', zorder=0)
    plt.xlabel('Primärdruck in MPa')
    plt.ylabel('Relativer Fehler in %')
    # plt.legend()



    print(f"Maximaler relativer Fehler: {max(relative_error)} %")

    # plt.tight_layout()
    plt.show()


def qne_plot():
    x = numpy.arange(0.2, 0.8, 0.01)
    Q_NE = 8629 * math.e ** (-10 * (1 - x))
    plt.plot(x, Q_NE)
    plt.show()


def test_partial_derivation():

    def calculate_derivative_T_p_s(fluid, pressure, quality):
        """Calculate (dT/dp)_s using CoolProp."""

        # Use CoolProp to find temperature at given pressure and entropy
        temperature = CP.PropsSI('T', 'P', pressure, 'T', quality, fluid_name)

        # Calculate the derivative (dT/dp)_s using CoolProp's built-in function
        dT_dp_s_value = CP.PropsSI('d(T)/d(P)|S', 'P', pressure, 'Q', quality, fluid)

        return dT_dp_s_value

    # Example usage:
    fluid_name = "CarbonDioxide"  # Example fluid
    pressure_example = 5000000.0  # Pa (example pressure)
    entropy_example = 1417.3  # J/kg.K (example entropy)

    result_derivative_T_p_s = calculate_derivative_T_p_s(fluid_name, pressure_example, 1)

    print(f"The derivative (dT/dp)_s is: {result_derivative_T_p_s}")


def sound_speed_calculation():

    p_throat = 4500000 # Pa
    q_list = numpy.linspace(0.0001, 0.9999, 100)
    c = []
    c_2 = []
    c_3 = []
    c_attou_list = []
    term2 = []
    term1 = []
    phi = []

    state_throat_vapor = med_prop.calc_state("PQ", p_throat, 1)
    state_throat_liquid = med_prop.calc_state("PQ", p_throat, 0)
    for q in q_list:
        state_throat = med_prop.calc_state("PQ", p_throat, q)

        # Volume fractions inside throat
        phi_throat_vapor = ((state_throat.q / state_throat_vapor.d) /
                            ((state_throat.q / state_throat_vapor.d) +
                             ((1-state_throat.q) / state_throat_liquid.d)))
        phi.append(phi_throat_vapor)
        phi_throat_liquid = 1 - phi_throat_vapor
        # Speed of sound for separate phases
        a_liquid = med_prop.get_saturated_speed_of_sound(p_throat, False)
        a_vapor = med_prop.get_saturated_speed_of_sound(p_throat, True)
        # Specific heat capacities
        c_p_liquid = med_prop.calc_transport_properties(state_throat_liquid).cp
        c_p_vapor = med_prop.calc_transport_properties(state_throat_vapor).cp
        # Extensive heat capacities
        C_p_liquid = state_throat_liquid.d * phi_throat_liquid * c_p_liquid
        C_p_vapor = state_throat_vapor.d * phi_throat_vapor * c_p_vapor
        # Thermal expansion coefficients
        if isinstance(med_prop, RefProp):
            beta_liquid = med_prop.calc_transport_properties(state_throat_liquid).beta
            beta_vapor = med_prop.calc_transport_properties(state_throat_vapor).beta
            zeta_liquid = state_throat.T * beta_liquid / (state_throat_liquid.d * c_p_liquid)
            zeta_vapor = state_throat.T * beta_vapor / (state_throat_vapor.d * c_p_vapor)
        elif isinstance(med_prop, CoolProp):
            zeta_vapor = med_prop.get_partial_derivative("T", "P", "S", state_throat_vapor)
            zeta_liquid = med_prop.get_partial_derivative("T", "P", "S", state_throat_liquid)
        x_1 = (phi_throat_vapor / (state_throat_vapor.d * a_vapor ** 2) +
               phi_throat_liquid / (state_throat_liquid.d * a_liquid ** 2))
        x_2 = C_p_vapor * C_p_liquid * (zeta_liquid - zeta_vapor) ** 2 / (C_p_vapor + C_p_liquid)
        # Speed of sound for two phase flow
        c_throat = (state_throat.d * x_1 + (state_throat.d / state_throat.T) * x_2) ** -0.5  #
        c_throat_2 = (state_throat.d * x_1) ** -0.5
        c.append(c_throat)
        c_2.append(c_throat_2)

        temperature = CP.PropsSI('T', 'P', p_throat, 'T', q, "CarbonDioxide")
        dT_dp_s_value_l = CP.PropsSI('d(T)/d(P)|S', 'P', p_throat, 'Q', 0, "CarbonDioxide")
        dT_dp_s_value_g = CP.PropsSI('d(T)/d(P)|S', 'P', p_throat, 'Q', 1, "CarbonDioxide")
        c_throat_3 = (state_throat.d * x_1 + (state_throat.d / state_throat.T) * C_p_vapor*C_p_liquid*(dT_dp_s_value_l-dT_dp_s_value_g)**2/(C_p_vapor+C_p_liquid)) ** -0.5
        # print(f"q={q}: c_throat_3 = ({state_throat.d} * {x_1} + ({state_throat.d} / {state_throat.T}) * {C_p_vapor} * {C_p_liquid} * ({dT_dp_s_value_l} - {dT_dp_s_value_g}) ** 2 / ({C_p_vapor} + {C_p_liquid})) ** -0.5")
        c_3.append(c_throat_3)
        term2.append(state_throat.d/state_throat.T*x_2)
        term1.append(state_throat.d * x_1)

        # calculate speed of sound for two phase flow with correlation from Attou and Seynhaeve (1999)
        if isinstance(med_prop, CoolProp):
            c_attou = med_prop.get_two_phase_speed_of_sound(p_throat, q)
            c_attou_list.append(c_attou)

    print(med_prop.get_saturated_speed_of_sound(p_throat, False))
    print(med_prop.get_saturated_speed_of_sound(p_throat, True))
    print(med_prop.calc_transport_properties(state_throat_liquid).cp)
    print(med_prop.calc_transport_properties(state_throat_vapor).cp)
    print(state_throat_liquid.d)
    print(state_throat_vapor.d)

    plt.figure(figsize=(10, 8))

    # First subplot
    plt.subplot(2, 1, 1)
    plt.plot(phi, c, label="long")
    plt.plot(phi, c_2, label="short")
    plt.plot(phi, c_3, label="CoolProp", linestyle='--')
    plt.plot(phi, c_attou_list, label="attou")
    plt.legend()
    plt.grid()
    plt.xlabel('Quality (Volume fraction)')
    plt.ylabel('Speed of sound')
    plt.title('Speed of sound comparison')

    # Second subplot
    plt.subplot(2, 1, 2)
    plt.plot(phi, term1, label="term 1")
    plt.plot(phi, term2, label="term 2")
    plt.plot(phi, numpy.divide(term2, term1), label="term 2 / term 1")
    plt.legend()
    plt.xlabel('Quality (Volume fraction)')
    plt.ylabel('Speed of sound difference')
    plt.title('Speed of sound difference')

    plt.tight_layout()
    plt.show()


def sound_speed_calculation_3d():
    p_values = numpy.linspace(1e5, 10e6, 50)  # Pressure values from 0.1 MPa to 10 MPa
    q_values = numpy.linspace(0.01, 0.99, 50)  # Quality values from 0.01 to 0.99

    P, Q = numpy.meshgrid(p_values, q_values)
    Z = numpy.full_like(P, numpy.nan, dtype=float)

    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            p = float(P[i, j])
            q = float(Q[i, j])
            try:
                # med_prop.get_two_phase_speed_of_sound erwartet (p, q) in SI (Pa, -)
                Z[i, j] = med_prop.get_two_phase_speed_of_sound(p, q)
            except Exception:
                Z[i, j] = numpy.nan

    # 3D-Plot (Druck in MPa für Achsenbeschriftung)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Q, P / 1e6, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.8, aspect=10)
    ax.set_ylabel('p in MPa')
    ax.set_xlabel('Quality q')
    ax.set_zlabel('Speed of sound (m/s)')
    plt.title('Two-phase speed of sound over p and q')
    plt.tight_layout()
    plt.show()


def _compute_error(args):
    p_p, p_t = args
    # Ejector-Zustand setzen
    ejector.state_primary = med_prop.calc_state("PT", p_p, 35.2 + 273.15)
    return p_p, p_t, ejector.iterate_throat_pressure(p_t)


def error_calculation_p_throat():
    """When running: change in ejector.py: 1. check nozzle quality to return -1; 2. return error h_throat after calculation"""

    p_throat = numpy.arange(10e5, 80e5, 1e4)
    p_primary = numpy.arange(60e5, 120e5, 1e4)
    # Liste aller (p_primary, p_throat)-Paare
    combos = [(p_p, p_t) for p_p in p_primary for p_t in p_throat]

    ejector.use_quick_solver = True

    with multiprocessing.Pool(processes=12) as pool:
        results = pool.map(_compute_error, combos)

    # Ergebnisse in passender Form anordnen
    error_list = [[0]*len(p_throat) for _ in range(len(p_primary))]
    for p_p, p_t, err in results:
        i = int((p_p - 6e6)/1e4)
        j = int((p_t - 1e6)/1e4)
        error_list[i][j] = err

    data = numpy.array(error_list).T

    # # Define the range of throat pressures and primary pressures
    # p_throat = numpy.arange(10e5, 80e5, 10000)
    # p_primary = numpy.arange(30e5, 120e5, 10000)
    #
    # combos = [(p_p, p_t) for p_p in p_primary for p_t in p_throat]
    # ejector.use_quick_solver = True
    #
    # results = []
    # total = len(combos)
    # with multiprocessing.Pool(processes=12) as pool:
    #     for idx, r in enumerate(pool.imap(_compute_error, combos), start=1):
    #         results.append(r)
    #         if idx % 10000 == 0:  # alle 10.000 Durchläufe
    #             print(f"Verarbeitet: {idx}/{total} Kombinationen.")
    #
    # error_list = [[0] * len(p_throat) for _ in range(len(p_primary))]
    # for p_p, p_t, err in results:
    #     i = int((p_p - 3e6) / 1e4)
    #     j = int((p_t - 1e6) / 1e4)
    #     error_list[i][j] = err
    #
    # data = numpy.array(error_list).T


    # # Initialize the error list to store errors for each combination of throat and primary pressures
    # error_list = [[0] * len(p_throat) for _ in range(len(p_primary))]
    # print(error_list.__len__())
    # print(error_list[29].__len__())
    #
    # ejector.use_quick_solver = True
    #
    # # Iterate over each primary pressure
    # i, j = 0, 0
    # for p_p in p_primary:
    #     # Set the primary state for the ejector
    #     ejector.state_primary = med_prop.calc_state("PT", p_p, 35.2 + 273.15)
    #
    #     # Iterate over each throat pressure
    #     for p_t in p_throat:
    #         # Calculate the error for the current throat pressure and store it in the error list
    #         error_list[i][j] = ejector.iterate_throat_pressure(p_t)
    #         j += 1
    #     i += 1
    #     j = 0
    #     print(f"Progress: {i}/{len(p_primary)}")
    #
    # # Convert the error list to a numpy array for plotting
    # data = numpy.array(error_list).T

    # Create a meshgrid for plotting
    x = p_primary / 1e5  # Convert to bar
    y = p_throat / 1e5  # Convert to bar
    x, y = numpy.meshgrid(x, y)

    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    # Create a custom colormap
    # Definiere die Farben und ihre Positionen
    colors = ['#00549F', '#57AB27', '#FFED00']
    positions = [0.0, 0.5, 1.0]  # Positionen für die Übergänge (von 0 bis 1)

    # Erstelle die benutzerdefinierte Colormap mit Positionsangaben
    cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    newcolors = cmap(numpy.linspace(0, 1, 256))
    red = numpy.array([204 / 256, 7 / 256, 30 / 256, 1])
    # newcolors[125, :] = red
    newcolors[:1, :] = red
    newcmp = ListedColormap(newcolors)

    # Plot the error surface
    fig = plt.figure(figsize=(6.125, 6.125/4*3))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, data, cmap=newcmp, zorder=1)

    # Add a colorbar
    fig.colorbar(surface, ax=ax, shrink=1, aspect=10)

    # Mark positions where z is about 0
    z_threshold = 0.01  # Define a threshold for z values close to 0
    mask = numpy.abs(data) < z_threshold
    ax.scatter(x[mask], y[mask], data[mask], color='red', s=10, label='z ≈ 0', zorder=10)

    ax.set_xlabel(r'$p_\mathrm{p} in bar')
    ax.set_ylabel(r'$p_\mathrm{t} in bar')
    ax.set_zlabel(r'Relativer Fehler in \%')
    plt.show()

    # # Create a 3D surface plot
    # fig = go.Figure(data=[go.Surface(z=data, x=x, y=y)])
    #
    # # Add labels
    # fig.update_layout(scene=dict(
    #     xaxis_title='Primary pressure',
    #     yaxis_title='Throat pressure',
    #     zaxis_title='Error'
    # ))
    #
    # # Mark positions where z is about 0
    # z_threshold = 0.01  # Define a threshold for z values close to 0
    # mask = numpy.abs(data) < z_threshold
    # fig.add_trace(go.Scatter3d(
    #     x=x[mask],
    #     y=y[mask],
    #     z=data[mask],
    #     mode='markers',
    #     marker=dict(color='green', size=5),
    #     name='z ≈ 0'
    # ))
    #
    # # Save the plot as an HTML file
    # fig.write_html('J:/error_surface.html')
    #
    # # To view the plot, open the HTML file in a web browser


def _compute_v_secondary_error(args):
    p_3, v_sm = args
    # Primär- und Sekundärzustand lokal setzen
    ejector.state_primary = med_prop.calc_state("PT", 9e6, 20 + 273.15)
    ejector.state_secondary = med_prop.calc_state("PT", 3e6, -5 + 273.15)
    # Fehler berechnen und zurückgeben
    return p_3, v_sm, ejector.calc_m_flow(p_3, v_secondary_mixing_start=v_sm, correlation=False)


def error_calculation_v_secondary_mixing():
    """When running: change in ejector.py: 1. return error_h_outlet in iterate_v_secondary_mixing; 2. return error_h_outlet in calc_m_flow"""

    v_secondary_mixing = numpy.arange(0, 150, 1)
    p_outlet = numpy.arange(3e6, 4.5e6, 1e4)

    # Liste der (p_3, v_s)-Kombinationen erstellen
    combos = [(p_3, v_sm) for p_3 in p_outlet for v_sm in v_secondary_mixing]
    error_list = [[0] * len(v_secondary_mixing) for _ in range(len(p_outlet))]

    ejector.use_quick_solver = True

    with multiprocessing.Pool(processes=12) as pool:
        results = pool.map(_compute_v_secondary_error, combos)

    # Ergebnisse in passender Form anordnen
    for p_3, v_sm, err in results:
        i = int((p_3 - 3e6) / 1e4)
        j = int(v_sm)
        error_list[i][j] = err

    data = numpy.array(error_list).T

    # # Define the range of throat pressures and primary pressures
    # v_secondary_mixing = numpy.arange(0, 150, 1)
    # p_outlet = numpy.arange(3e6, 4.5e6, 10000)
    #
    # # Initialize the error list to store errors for each combination of throat and primary pressures
    # error_list = [[0] * len(v_secondary_mixing) for _ in range(len(p_outlet))]
    #
    # ejector.use_quick_solver = True
    #
    # # Set the primary state for the ejector
    # ejector.state_primary = med_prop.calc_state("PT", 9e6, 20 + 273.15)
    # ejector.state_secondary = med_prop.calc_state("PT", 3e6, -5 + 273.15)
    #
    # # Iterate over each primary pressure
    # i, j = 0, 0
    # for p_3 in p_outlet:
    #
    #     # Iterate over each throat pressure
    #     for v_s in v_secondary_mixing:
    #         # Calculate the error for the current throat pressure and store it in the error list
    #         error_list[i][j] = ejector.calc_m_flow(p_3, v_secondary_mixing_start=v_s, correlation=True)
    #         j += 1
    #     i += 1
    #     j = 0
    #     print(f"Progress: {i}/{len(p_outlet)}")
    #
    # # Convert the error list to a numpy array for plotting
    # data = numpy.array(error_list).T

    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    # Create a custom colormap
    # Definiere die Farben und ihre Positionen
    colors = ['#00549F', '#57AB27', '#FFED00']
    positions = [0.0, 0.5, 1.0]  # Positionen für die Übergänge (von 0 bis 1)

    # Erstelle die benutzerdefinierte Colormap mit Positionsangaben
    cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)))

    newcolors = cmap(numpy.linspace(0, 1, 256))
    red = numpy.array([204 / 256, 7 / 256, 30 / 256, 1])
    newcmp = ListedColormap(newcolors)

    # Create a meshgrid for plotting
    x = p_outlet/1e5  # Convert to bar
    y = v_secondary_mixing
    x, y = numpy.meshgrid(x, y)

    # Plot the error surface
    fig = plt.figure(figsize=(6.125, 6.125/4*3))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(x, y, data, cmap=newcmp)

    # Mark positions where z is about 0
    z_threshold = 0.01  # Define a threshold for z values close to 0
    mask = numpy.abs(data) < z_threshold
    ax.scatter(x[mask], y[mask], data[mask], color=red, s=10, label='z ≈ 0', zorder=10)

    # Add a colorbar
    fig.colorbar(surface, ax=ax, shrink=.8, aspect=10)

    ax.set_xlabel('$p_3$ in bar')
    ax.set_ylabel(r'$v_\mathrm{sm}$ in m/s')
    ax.set_zlabel('Relativer Fehler in %')
    plt.show()

    # # Create a 3D surface plot
    # fig = go.Figure(data=[go.Surface(z=data, x=x, y=y)])
    #
    # # Add labels
    # fig.update_layout(scene=dict(
    #     xaxis_title='Outlet pressure',
    #     yaxis_title='Secondary mixing velocity',
    #     zaxis_title='Error'
    # ))
    #
    # # Mark positions where z is about 0
    # z_threshold = 0.01  # Define a threshold for z values close to 0
    # mask = numpy.abs(data) < z_threshold
    # fig.add_trace(go.Scatter3d(
    #     x=x[mask],
    #     y=y[mask],
    #     z=data[mask],
    #     mode='markers',
    #     marker=dict(color='green', size=5),
    #     name='z ≈ 0'
    # ))
    #
    # # Save the plot as an HTML file
    # fig.write_html('J:/error_surface.html')
    #
    # # To view the plot, open the HTML file in a web browser


def p_throat_iteration():

    pp_list = [8.1, 8.15, 8.39, 8.43, 8.44, 8.66, 8.71, 8.72, 9.04, 9.05, 9.19, 9.22, 9.26, 9.3, 9.33, 9.34, 9.45, 9.57,
               9.67, 9.71]
    pp_list_2 = list(numpy.arange(7.9, 10.1, 0.1))
    m_flow_list = []
    m_flow_list_2 = []
    m_flow_list_correlation = []
    m_flow_list_qne = []
    q_list = []
    pt_list = []
    pt_list_correlation = []
    pt_list_qne = []
    rel_fehler_qne = []
    rel_fehler_no_qne = []

    ejector.state_secondary = med_prop.calc_state("PT", 3.6 * 1e6, 22.5 + 273.15)
    ejector_2.state_secondary = med_prop.calc_state("PT", 3.6 * 1e6, 22.5 + 273.15)

    for i in range(len(pp_list_2)):
        print(f"Calculating state for pressure: {pp_list_2[i]} MPa")
        ejector.state_primary = med_prop.calc_state("PT", pp_list_2[i] * 1e6, 35.2 + 273.15)
        ejector.calc_m_flow(4.2 * 1e6, correlation=False)
        m_flow_list.append(ejector.m_flow_primary * 3600)
        q_list.append(ejector.state_throat.q)
        pt_list.append(ejector.state_throat.p / 1e6)

        ejector.state_primary = med_prop.calc_state("PT", pp_list_2[i] * 1e6, 35.2 + 273.15)
        ejector.calc_m_flow(4.2 * 1e6, correlation=True)
        m_flow_list_correlation.append(ejector.m_flow_primary * 3600)
        pt_list_correlation.append(ejector.state_throat.p / 1e6)

        ejector.state_primary = med_prop.calc_state("PT", pp_list_2[i] * 1e6, 35.2 + 273.15)
        ejector.calc_m_flow(4.2 * 1e6, correlation=False, QNE=True)
        m_flow_list_qne.append(ejector.m_flow_primary * 3600)
        pt_list_qne.append(ejector.state_throat.p / 1e6)

        rel_fehler_qne.append((pt_list_qne[i] - pt_list_correlation[i])/pt_list_correlation[i]*100)
        rel_fehler_no_qne.append((pt_list[i] - pt_list_correlation[i])/pt_list_correlation[i]*100)

    print(max(rel_fehler_qne))

    # plt.subplot(3, 1, 1)
    # plt.plot(pp_list, m_flow_list)
    # plt.xlabel('Primary pressure [MPa]')
    # plt.ylabel('Primary mass flow [kg/h]')
    # plt.title('p_throat iteration - Q_NE factor = 2500')
    # plt.subplot(3, 1, 2)
    # plt.plot(pp_list, q_list)
    # plt.xlabel('Primary pressure [MPa]')
    # plt.ylabel('Quality [-]')
    # plt.subplot(3, 1, 3)
    plt.figure(figsize=(6.125, 6.125/4*2))
    plt.suptitle('Bestimmung des Drucks im Düsenhals')
    plt.subplot(1, 2, 1)
    plt.plot(pp_list_2, pt_list_qne, label="Iteration mit Q_NE")
    plt.plot(pp_list_2, pt_list, label="Iteration ohne Q_NE")
    plt.plot(pp_list_2, pt_list_correlation, label="Korrelation nach Zhu")
    plt.xlabel('Primärdruck in MPa')
    plt.ylabel('Druck im Düsenhals in MPa')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(pp_list_2, rel_fehler_qne, label="Iteration mit Q_NE")
    plt.plot(pp_list_2, rel_fehler_no_qne, label="Iteration ohne Q_NE")
    plt.xlabel('Primärdruck in MPa')
    plt.ylabel('Relativer Fehler in %')
    #plt.legend()

    plt.show()


if __name__ == "__main__":
    main(use_condenser_inlet=False)

