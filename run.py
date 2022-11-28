from scripts.Henrik.process_data import interpolate_norkyst_to_barentswatch
from scripts.Henrik.process_data import fit_models
from scripts.Henrik.process_data import compare_timeseries
from scripts.Henrik import temp3m_analysis
from scripts.Henrik import temp3m_prep_hindcast
from scripts.Henrik import temp3m_finding_nano

if __name__=='__main__':

    # temp3m_finding_nano.main()

    temp3m_prep_hindcast.main()
    temp3m_analysis.main()

    # interpolate_norkyst_to_barentswatch.main()
    # fit_models.main()
    # compare_timeseries.main()
