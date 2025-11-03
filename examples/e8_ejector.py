# # Example for standard ejector cycle
import logging

from vclibpy.media import CoolProp
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


if __name__ == "__main__":
    main('D:/kbr-fme/ebc.paper.mplstyle')

