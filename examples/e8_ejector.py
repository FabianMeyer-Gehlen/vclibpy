# # Example for standard ejector cycle
import logging

from vclibpy.media import CoolProp
from vclibpy.components.expansion_valves import EjectorLiu
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

med_prop = CoolProp(fluid_name="CarbonDioxide")

def main(rcParams_path: str = None):

    try:
        plt.style.use(rcParams_path)
    except OSError:
        logging.warning("Could not load the custom matplotlib style file. Using default style.")
        print("Could not load the custom matplotlib style file. Using default style.")

    mpl.rcParams['svg.fonttype'] = 'none'  # make pyplot save text as text and not paths, so it is editable in Inkscape

    test_sound_speed(130e5, 47.71 + 273.15, 0.5)
    # test_ejector()
    # plot_two_phase_sound_speed(from_derivative = False)

def test_sound_speed(p_primary: float, T_primary: float, eta_is:float):
    state_primary = med_prop.calc_state("PT", p_primary, T_primary)
    pressures = np.linspace(10e5, state_primary.p, 500)
    soundspeed = np.zeros(len(pressures))
    velocity = np.zeros(len(pressures))
    for i, p in enumerate(pressures):
        # print(f"calculating for pressure: {p}")
        try:
            state_throat_is = med_prop.calc_state("PS", p, state_primary.s)
            state_throat = med_prop.calc_state("PH", p, state_primary.h + (state_throat_is.h - state_primary.h)*eta_is)
        except ValueError:
            continue
        if 0 <= state_throat.q <= 1:
            c = med_prop.get_two_phase_speed_of_sound(p, state_throat.q)
        else:
            try:
                c = med_prop.get_speed_of_sound(state_throat)
            except ValueError:
                c = np.nan
        velocity[i] = np.sqrt(2*(state_primary.h - state_throat.h))
        soundspeed[i] = c
    plt.figure()
    plt.scatter(pressures / 1e5, soundspeed, label='Speed of sound', s=2)
    plt.scatter(pressures / 1e5, velocity, label='Velocity', s=2, color='orange')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('m/s')
    plt.title('Speed of Sound and Velocity in Motive Nozzle')
    plt.legend()
    plt.show()

def _compute_soundspeed(args):
    p, h, from_derivative = args
    state = med_prop.calc_state("PH", p, h)
    if 0 <= state.q <= 1:
        if from_derivative:
            c = med_prop.get_partial_derivative("P", "D", "S", state) ** 0.5
        else:
            c = med_prop.get_two_phase_speed_of_sound(p, state.q)
    else:
        try:
            c = med_prop.get_speed_of_sound(state)
        except ValueError:
            c = np.nan
    return p, h, c

def plot_two_phase_sound_speed(from_derivative: bool = False, processes: int = multiprocessing.cpu_count()):
    pressures = np.linspace(50e5, 100e5, 500)
    enthalpies = np.linspace(150e3, 500e3, 500)
    combos = [(p,h,from_derivative) for p in pressures for h in enthalpies]

    with multiprocessing.Pool(processes) as pool:
        results = pool.map(_compute_soundspeed, combos)

    for p, h, c in results:
        i = np.where(pressures == p)[0][0]
        j = np.where(enthalpies == h)[0][0]
        if 'soundspeed' not in locals():
            soundspeed = np.zeros((len(pressures), len(enthalpies)))
        try:
            soundspeed[i, j] = c
        except TypeError:
            soundspeed[i, j] = np.nan

    P, H = np.meshgrid(enthalpies / 1e3, pressures / 1e5)
    plt.figure()
    cp = plt.contourf(P, H, soundspeed, levels=50, cmap='viridis')
    plt.colorbar(cp, label='Speed of sound (m/s)')
    plt.xlabel('Enthalpy (kJ/kg)')
    plt.ylabel('Pressure (bar)')
    plt.title('Speed of Sound in CO2')
    plt.show()


def test_ejector():
    # 1
    ejector = EjectorLiu(d_throat=2.7, d_suction=17.2, show_iteration=True)
    ejector.med_prop = CoolProp(fluid_name="CarbonDioxide")
    entrainment_ratio = 0.3
    state_motive = ejector.med_prop.calc_state("PT", 96.74e5, 42.89 + 273.15)
    state_suction = ejector.med_prop.calc_state("PT", 42.19e5, 23.86 + 273.15)

    # 8
    # ejector = EjectorLiu(d_throat=1.8, d_suction= 5.5 ,show_iteration=True)
    # ejector.med_prop = CoolProp(fluid_name="CarbonDioxide")
    # entrainment_ratio = 0.5
    # state_motive = ejector.med_prop.calc_state("PT", 129.45e5, 47.71+273.15)
    # state_suction = ejector.med_prop.calc_state("PT", 31.11e5, 24.89+273.15)

    # 10
    # ejector = EjectorLiu(d_throat=2.6, d_suction=17.2, show_iteration=True)
    # ejector.med_prop = CoolProp(fluid_name="CarbonDioxide")
    # entrainment_ratio = 0.06/0.23
    # state_motive = ejector.med_prop.calc_state("PT", 108.98e5, 49.08 + 273.15)
    # state_suction = ejector.med_prop.calc_state("PT", 46.7e5, 23 + 273.15)



    ejector.calculate_motive_nozzle(state_motive.p, state_motive.h, state_suction.p)
    ejector.calculate_suction_nozzle(entrainment_ratio, state_suction.p, state_suction.h)
    ejector.calculate_mixing_chamber(entrainment_ratio)
    ejector.calculate_diffusor()
    print(f"Ejector data:"
          f"\n Mass flow motive: {ejector.m_flow_primary:.3f} kg/s"
          f"\n Mass flow suction: {ejector.m_flow_secondary:.3f} kg/s"
          f"\n Mass flow mixed: {ejector.m_flow_outlet:.3f} kg/s"
          f"\n Pressures: {ejector.state_primary.p}, {ejector.state_primary_throat.p}, {ejector.state_secondary.p}, {ejector.state_secondary_mixing.p}, {ejector.state_mixing.p}, {ejector.state_outlet.p}")
if __name__ == "__main__":
    main('D:/kbr-fme/ebc.paper.mplstyle')

