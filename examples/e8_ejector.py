# # Example for standard ejector cycle
import logging

from vclibpy.media import CoolProp
from vclibpy.components.expansion_valves import EjectorLiu
import matplotlib as mpl
import matplotlib.pyplot as plt

def main(rcParams_path: str = None):
    med_prop = CoolProp(fluid_name="CarbonDioxide")

    try:
        plt.style.use(rcParams_path)
    except OSError:
        logging.warning("Could not load the custom matplotlib style file. Using default style.")
        print("Could not load the custom matplotlib style file. Using default style.")

    mpl.rcParams['svg.fonttype'] = 'none'  # make pyplot save text as text and not paths, so it is editable in Inkscape

    test_ejector()

def test_ejector():
    ejector = EjectorLiu(d_throat=2.7, show_iteration=True)
    ejector.med_prop = CoolProp(fluid_name="CarbonDioxide")
    state_motive = ejector.med_prop.calc_state("PT", 97e5, 43+273.15)
    ejector.calculate_motive_nozzle(state_motive.p, 43e5, state_motive.h)
    state_suction = ejector.med_prop.calc_state("PT", 42e5, 24+273.15)
    ejector.calculate_suction_nozzle(0.3, state_suction.p, state_suction.h)
    print(f"Ejector data:"
          f"\n Mass flow motive: {ejector.m_flow_primary:.3f} kg/s"
          f"\n Mass flow suction: {ejector.m_flow_secondary:.3f} kg/s"
          f"\n Pressures: {ejector.state_primary.p}, {ejector.state_primary_throat.p}, {ejector.state_secondary.p}, {ejector.state_secondary_mixing.p}")
    ejector.calculate_mixing_chamber(0.3)
if __name__ == "__main__":
    main('D:/kbr-fme/ebc.paper.mplstyle')

