# import scripts.Henrik.process_norkyst
# import scripts.Henrik.verify2norkyst2
# import scripts.Henrik.compare_rho
# import scripts.Henrik.era_variance
# from scripts.Henrik.climatology_wind import compute_climatology
from scripts.Henrik.era_coastline_davos import go_combo,go_combo_data

# from scripts.Henrik.test_map import go
# from scripts.Henrik.norkyst_variance import run as nv_run
# from scripts.Henrik.norkyst2BWloc import run as n2b_run
# from scripts.Henrik.hardanger_qq import run as qq_run
# from scripts.Henrik.norkyst_800 import assign_bw_coords, stack_hardanger
from scripts.Henrik.hardanger_skill_norkyst_davos import interpolate_hincast_to_observations, maps, calc_crps
# from scripts.Henrik.timeseries import observations_closeup,observations

from scripts.Henrik import fig1 as hr1
from scripts.Henrik import fig2 as hr2
from scripts.Henrik import fig3 as hr3

from scripts.Henrik import new_fig1 as f1
#from scripts.Henrik import new_fig2 as f2
from scripts.Henrik import new_fig2_abs_values as f2
from scripts.Henrik import new_fig3 as f3
from scripts.Henrik import new_fig4 as f4
from scripts.Henrik import new_fig4_persistence as f4p
from scripts.Henrik import new_fig4_combo as f4c
from scripts.Henrik import new_fig4_timeseries as f4t
from scripts.Henrik import new_fig4_mae as f4m
from scripts.Henrik import new_fig6 as f6
from scripts.Henrik import new_fig7 as f7
from scripts.Henrik import new_fig7_whole_coast as f7_wc
from scripts.Henrik import new_fig7_whole_coast_monthly as f7_wcm
from scripts.Henrik import distance_from_coast as dfc

from scripts.Henrik.process_data import interpolate_norkyst_to_barentswatch
from scripts.Henrik.process_data import fit_models

if __name__=='__main__':

    interpolate_norkyst_to_barentswatch.main()
    fit_models.main()
