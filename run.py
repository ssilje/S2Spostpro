# import scripts.Henrik.process_norkyst
# import scripts.Henrik.verify2norkyst2
# import scripts.Henrik.compare_rho
# import scripts.Henrik.era_variance
from scripts.Henrik.norkyst_variance import run as nv_run
# from scripts.Henrik.norkyst2BWloc import run as n2b_run
from scripts.Henrik.hardanger_qq import run as qq_run
from scripts.Henrik.norkyst_800 import download, download_new, to_bw
from scripts.Henrik.timeseries import observations_closeup,observations

if __name__=='__main__':

    # observations('Hisdalen')
    observations_closeup('Hisdalen')
    # download()
    # to_bw()
    # n2b_run()
    # qq_run()
    # nv_run()
# import scripts.Henrik.wind_against_ERA
# import scripts.Henrik.skill_on_norkyst_CY47R2
# import scripts.Henrik.correlation_norkyst_bw
# import scripts.Henrik.skill_on_era2


# import scripts.Henrik.hele_kysten
# import scripts.Henrik.hisdalen_obs

# import scripts.Henrik.process_norkyst
# import scripts.Henrik.norkyst_restack
# from scripts.Henrik.norkyst_restack import run
# if __name__=='__main__':
#     run()
# import scripts.Henrik.hardanger_hist
# import scripts.Henrik.hardanger_memory_test
# import scripts.Henrik.hardanger_season
# import scripts.Henrik.hardanger_qq
