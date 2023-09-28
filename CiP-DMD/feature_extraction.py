import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from process.sawing import SawingProcessData
from process.milling import MillingProcessData

path_data_labels = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/"
path_data_sawing = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/"
path_data_milling = "/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/milling_raw_data/"

# For each feature calculate the tsfresh comprehensive features
labels_df = pd.read_csv(path_data_labels + "qc_data_labels_train_fe.csv", sep=";")
part_IDs = labels_df.iloc[:, 0].tolist()
# # response = labels_df.iloc[:, 1].tolist()
# # Quality control labels
# surface_roughness_label = labels_df.iloc[:, 1].tolist()
# parallelism_label = labels_df.iloc[:, 2].tolist()
# groove_depth_label = labels_df.iloc[:, 3].tolist()
# groove_diameter_label = labels_df.iloc[:, 4].tolist()
# qc_label = labels_df.iloc[:, 5].tolist()
# # Anomaly labels
# anomaly_all_label = labels_df.iloc[:, 6].tolist()

data_frames = {
    "sawing": pd.DataFrame(),
    "side_1_planfraesen": pd.DataFrame(),
    "side_1_aussenkontur_schruppen_schlichten": pd.DataFrame(),
    "side_1_nut_seitlich": pd.DataFrame(),
    "side_1_stufenbohrung": pd.DataFrame(),
    "side_1_endgraten_aussenkontur_bohrungen": pd.DataFrame(),
    "side_1_bohren_seitlich": pd.DataFrame(),
    "side_1_bohren_senken": pd.DataFrame(),
    "side_1_bohren": pd.DataFrame(),
    "side_1_gewinde_fraesen": pd.DataFrame(),
    "side_2_planfraesen": pd.DataFrame(),
    "side_2_kreistasche_fraesen": pd.DataFrame(),
    "side_2_bauteil_entgraten": pd.DataFrame(),
    "side_2_ringnut": pd.DataFrame(),
}

for id in range(len(part_IDs)):
    part_ID = part_IDs[id]
    print(id)

    try:
        # Get sawing process features
        reader = SawingProcessData(path_data_sawing)
        saw_features_raw = reader.get_process_QH_id(str(part_ID))

        # Get milling process features
        reader = MillingProcessData(path_data_milling)
        acc_features, bfc_features = reader.get_process_QH_id(str(part_ID))

        saw_features_raw.fillna(0, inplace=True)
        acc_features.fillna(0, inplace=True)
        bfc_features.fillna(0, inplace=True)

        milling_features = pd.concat([acc_features, bfc_features], axis=0)

        data_frames["sawing"] = data_frames["sawing"].append(saw_features_raw)
        data_frames["side_1_planfraesen"] = data_frames["side_1_planfraesen"].append(
            milling_features.iloc[0]
        )
        data_frames["side_1_aussenkontur_schruppen_schlichten"] = data_frames[
            "side_1_aussenkontur_schruppen_schlichten"
        ].append(milling_features.iloc[1])
        data_frames["side_1_nut_seitlich"] = data_frames["side_1_nut_seitlich"].append(
            milling_features.iloc[2]
        )
        data_frames["side_1_stufenbohrung"] = data_frames[
            "side_1_stufenbohrung"
        ].append(milling_features.iloc[3])
        data_frames["side_1_endgraten_aussenkontur_bohrungen"] = data_frames[
            "side_1_endgraten_aussenkontur_bohrungen"
        ].append(milling_features.iloc[4])
        data_frames["side_1_bohren_seitlich"] = data_frames[
            "side_1_bohren_seitlich"
        ].append(milling_features.iloc[5])
        data_frames["side_1_bohren_senken"] = data_frames[
            "side_1_bohren_senken"
        ].append(milling_features.iloc[6])
        data_frames["side_1_bohren"] = data_frames["side_1_bohren"].append(
            milling_features.iloc[7]
        )
        data_frames["side_1_gewinde_fraesen"] = data_frames[
            "side_1_gewinde_fraesen"
        ].append(milling_features.iloc[8])
        data_frames["side_2_planfraesen"] = data_frames["side_2_planfraesen"].append(
            milling_features.iloc[9]
        )
        data_frames["side_2_kreistasche_fraesen"] = data_frames[
            "side_2_kreistasche_fraesen"
        ].append(milling_features.iloc[10])
        data_frames["side_2_bauteil_entgraten"] = data_frames[
            "side_2_bauteil_entgraten"
        ].append(milling_features.iloc[11])
        data_frames["side_2_ringnut"] = data_frames["side_2_ringnut"].append(
            milling_features.iloc[12]
        )

    except:
        print("Part ID", id, "has nan-values.")

# Save dataframes to separate CSV files
for key, df in data_frames.items():
    file_path = f"/home/b.cassoli@PTW.Maschinenbau.TU-Darmstadt.de/projects/phd/phd/cip_dmd/notebooks/{key}_df.csv"
    df.to_csv(file_path, sep=";", index=False)
