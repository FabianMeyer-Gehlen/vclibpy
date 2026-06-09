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

    # test_sound_speed(129.45e5, 47.71 + 273.15, 0.5)
    # test_ejector()
    # plot_two_phase_sound_speed_3d(from_derivative = True)
    plot_two_phase_sound_speed(50e5, "q")
    # test_triple_point_calculation(med_prop.get_triple_point)

def test_triple_point_calculation(func, runs: int = 100):
    import time

    times = []

    for _ in range(runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    print(f"Average execution time over {runs} runs: {sum(times) / runs:.6f} seconds")
    print(f"Min: {min(times):.6f} s, Max: {max(times):.6f} s")

def plot_two_phase_sound_speed(p:float, mode = "h"):
    if mode == "h":
        enthalpies = np.linspace(150e3, 500e3, 500)
        x = enthalpies
        m = "PH"
        factor = 1e3
        label = 'Enthalpy (kJ/kg)'
    elif mode == "q":
        qualities = np.linspace(0, 1, 500)
        x = qualities
        m = "PQ"
        factor = 1
        label = 'Quality (-)'
    elif mode == "alpha":
        alphas = np.linspace(0, 1, 500)
        state_vapor = med_prop.calc_state("PQ", p, 1)
        state_liquid = med_prop.calc_state("PQ", p, 0)
        qualities = np.array([state_vapor.d*alpha/(state_vapor.d*alpha+state_liquid.d*(1-alpha)) for alpha in alphas])
        x = qualities
        m = "PQ"
        factor = 1
        label = 'void fraction (-)'
    else:
        raise ValueError("Invalid mode. Use 'h' for enthalpy or 'q' for quality or 'alpha' for void fraction.")
    soundspeeds_Attou_HEM = np.zeros_like(x)
    soundspeeds_derivatives = np.zeros_like(x)
    soundspeeds_lund = np.zeros_like(x)
    for i,j in enumerate(x):
        try:
            state = med_prop.calc_state(m, p, j)
            if 0 <= state.q <= 1:
                c = med_prop.get_two_phase_speed_of_sound(p, state.q, "Attou_HEM")
                c_d = med_prop.get_partial_derivative("P", "D", "S", state) ** 0.5
                c_lund = med_prop.get_two_phase_speed_of_sound(p, state.q, "Lund")
            else:
                c, c_d, c_lund = med_prop.get_speed_of_sound(state), np.nan, np.nan
        except ValueError as err:
            print(f"calculation failed for {j}, because of Error: {err}")
            c, c_d, c_lund = np.nan, np.nan, np.nan
        soundspeeds_Attou_HEM[i] = c
        soundspeeds_lund[i] = c_lund
        try:
            soundspeeds_derivatives[i] = c_d
        except TypeError:
            soundspeeds_derivatives[i] = np.nan
    if mode == "alpha":
        x = alphas
    plt.figure()
    plt.scatter(x / factor, soundspeeds_Attou_HEM, label='Speed of sound after Attou_HEM', s=2)
    plt.scatter(x / factor, soundspeeds_derivatives, label='Speed of sound from derivative', s=2, color='orange')
    plt.scatter(x / factor, soundspeeds_lund, label='Speed of sound after Lund', s=2, color='green')
    plt.xlabel(label)
    plt.ylabel('m/s')
    plt.title(f'Speed of Sound in CO2 at {p/1e5:.1f} bar')
    plt.legend()
    plt.show()


def test_sound_speed(p_primary: float, T_primary: float, eta_is:float):
    state_primary = med_prop.calc_state("PT", p_primary, T_primary)
    pressures = np.linspace(10e5, state_primary.p, 500)
    soundspeed = np.zeros(len(pressures))
    soundspeed_derivative = np.zeros(len(pressures))
    velocity = np.zeros(len(pressures))
    for i, p in enumerate(pressures):
        # print(f"calculating for pressure: {p}")
        try:
            state_throat_is = med_prop.calc_state("PS", p, state_primary.s)
            state_throat = med_prop.calc_state("PH", p, state_primary.h + (state_throat_is.h - state_primary.h)*eta_is)
        except ValueError:
            continue
        if 0 <= state_throat.q <= 1:
            c = med_prop.get_two_phase_speed_of_sound(p, state_throat.q, "Attou_HEM")
            c_2 = med_prop.get_partial_derivative("P", "D", "S", state_throat) ** 0.5
        else:
            try:
                c = med_prop.get_speed_of_sound(state_throat)
                c_2 = med_prop.get_speed_of_sound(state_throat)
            except ValueError:
                c = np.nan
                c_2 = np.nan
        velocity[i] = np.sqrt(2*(state_primary.h - state_throat.h))
        soundspeed[i] = c
        soundspeed_derivative[i] = c_2
    plt.figure()
    plt.scatter(pressures / 1e5, soundspeed, label='Speed of sound', s=2)
    plt.scatter(pressures / 1e5, velocity, label='Velocity', s=2, color='orange')
    plt.scatter(pressures / 1e5, soundspeed_derivative, label='Speed of sound from derivative', s=2, color='green')
    plt.xlabel('Pressure (bar)')
    plt.ylabel('m/s')
    plt.title('Speed of Sound and Velocity in Motive Nozzle')
    plt.legend()
    plt.show()

def _compute_soundspeed(args):
    p, h, from_derivative = args
    try:
        state = med_prop.calc_state("PH", p, h)
    except ValueError:
        return p, h, np.nan
    if 0 <= state.q <= 1:
        if from_derivative:
            c = med_prop.get_partial_derivative("P", "D", "S", state) ** 0.5
        else:
            c = med_prop.get_two_phase_speed_of_sound(p, state.q, "Attou_HEM")
    else:
        try:
            c = med_prop.get_speed_of_sound(state)
        except ValueError:
            c = np.nan
    return p, h, c

def plot_two_phase_sound_speed_3d(from_derivative: bool = False, processes: int = multiprocessing.cpu_count()):
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

