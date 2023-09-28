# %%
import pandas as pd
import warnings

warnings.simplefilter(action="ignore")
import pandas as pd
import numpy as np
from tsfresh import select_features, feature_extraction
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table

path_data = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/"

labels_df = pd.read_csv(path_data + "qc_data_labels_train_fe.csv", sep=";")
indexes_to_drop = [9, 50, 54, 73, 132, 134]
labels_used_df = labels_df.drop(indexes_to_drop).reset_index()
response = labels_used_df.iloc[:, 2]  # surface roughness
# response = labels_used_df.iloc[:, 3]  # parallelism

# Number of features to select for each subprocess
# n_features = 5
# n_features = 10
n_features = 15

# Total: 14 process (1 sawing and 13 milling)

processes = {
    "sawing": "sawing_df",
    "side_1_planfraesen": "side_1_planfraesen_df",
    "side_1_aussenkontur_schruppen_schlichten": "side_1_aussenkontur_schruppen_schlichten_df",
    "side_1_nut_seitlich": "side_1_nut_seitlich_df",
    "side_1_stufenbohrung": "side_1_stufenbohrung_df",
    "side_1_endgraten_aussenkontur_bohrungen": "side_1_endgraten_aussenkontur_bohrungen_df",
    "side_1_bohren_seitlich": "side_1_bohren_seitlich_df",
    "side_1_bohren_senken": "side_1_bohren_senken_df",
    "side_1_bohren": "side_1_bohren_df",
    "side_1_gewinde_fraesen": "side_1_gewinde_fraesen_df",
    "side_2_planfraesen": "side_2_planfraesen_df",
    "side_2_kreistasche_fraesen": "side_2_kreistasche_fraesen_df",
    "side_2_bauteil_entgraten": "side_2_bauteil_entgraten_df",
    "side_2_ringnut": "side_2_ringnut_df",
}


def extract_kind_to_fc_parameter(df_name, file_name, n_features):
    df = pd.read_csv(path_data + file_name + ".csv", sep=";")
    df.fillna(0, inplace=True)

    relevance_table = calculate_relevance_table(df, response)
    # relevance_table = relevance_table[relevance_table.relevant]
    relevance_table.sort_values("p_value", inplace=True)
    top_n_features = relevance_table.feature[:n_features]

    kind_to_fc_parameter = feature_extraction.settings.from_columns(top_n_features)

    return kind_to_fc_parameter


kind_to_fc_parameters = {}

for df_name, file_name in processes.items():
    kind_to_fc_parameter = extract_kind_to_fc_parameter(df_name, file_name, n_features)
    kind_to_fc_parameters[df_name] = kind_to_fc_parameter

# %%
