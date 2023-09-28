#%%
import torch as th
from data_preparation import CiP_DMGD
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

device = th.device("cuda:1" if th.cuda.is_available() else "cpu")
th.manual_seed(42)

# Directories
raw_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/cip_dmgd_n5/baselines/"  # Raw data
save_dir = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/"  # Processed data, change when number of features changes

# Create graph dataset
graph_data = CiP_DMGD(
    data_Name="test_2",
    data_name_state="_graph_5",
    raw_dir=raw_dir,
    save_dir=save_dir,
    force_reload=True,
    verbose=False,
    oversampling=False,
    n_aggregated_features=5,  # number of aggregated features (number of features per process)
)