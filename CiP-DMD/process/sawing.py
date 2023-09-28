import json
import h5py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tsfresh.feature_extraction import (
    extract_features,
    feature_calculators,
    MinimalFCParameters,
    EfficientFCParameters,
)

from config import Features_Sawing_5, Features_Sawing_10

selection_5 = {
    "CutCounter": {
        "cwt_coefficients": [
            {"coeff": 6, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 9, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 8, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 7, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 10, "w": 20, "widths": (2, 5, 10, 20)},
        ]
    }
}

selection_10 = {
    "CutCounter": {
        "cwt_coefficients": [
            {"coeff": 6, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 9, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 8, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 7, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 10, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 11, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 5, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 12, "w": 20, "widths": (2, 5, 10, 20)},
        ]
    },
    "Vib01.Peak": {"fft_coefficient": [{"attr": "abs", "coeff": 20}]},
    "Vib01.RMS": {"fft_coefficient": [{"attr": "real", "coeff": 48}]},
}

selection_15 = {
    "CutCounter": {
        "cwt_coefficients": [
            {"coeff": 6, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 9, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 8, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 7, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 10, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 11, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 5, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 12, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 13, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 14, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 2, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 4, "w": 20, "widths": (2, 5, 10, 20)},
            {"coeff": 3, "w": 20, "widths": (2, 5, 10, 20)},
        ]
    },
    "Vib01.Peak": {"fft_coefficient": [{"attr": "abs", "coeff": 20}]},
    "Vib01.RMS": {"fft_coefficient": [{"attr": "real", "coeff": 48}]},
}


class SawingProcessData:
    def __init__(self, path_data):
        self.tmp_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "../tmp_files")
        )
        self._path = path_data
        self.process_name = "cutting"
        self.selected_features = Features_Sawing_5
        self.all_features = [
            "CPU_Kuehler_Temp",
            "CPU_Temp",
            "CutCounter",
            "CutTime",
            "FFT_Anforderung",
            "FlatstreamCutCounter",
            "FlatstreamDone",
            "FsMode_1Raw_2FftRaw_3FttHK",
            "HebenAktiv",
            "MotorAn",
            "PData.CosPhi",
            "PData.CutEnergy",
            "PData.PEff",
            "P_Vorschub",
            "Position",
            "Position_Band",
            "TData.T1",
            "TData.T2",
            "TData.T3",
            "TData.T4",
            "TData.T_IR",
            "Vib01.CREST",
            "Vib01.Kurtosis",
            "Vib01.Peak",
            "Vib01.RMS",
            "Vib01.Skewness",
            "Vib01.VDI3832",
            "Vib02.CREST",
            "Vib02.Kurtosis",
            "Vib02.Peak",
            "Vib02.RMS",
            "Vib02.Skewness",
            "Vib02.VDI3832",
            "Vib03.CREST",
            "Vib03.Kurtosis",
            "Vib03.Peak",
            "Vib03.RMS",
            "Vib03.Skewness",
            "Vib03.VDI3832",
            "ZaehneProBand",
            "bCutActive",
            "fLichtschranke",
            "obereMaterialkante",
            "vVorschub",
        ]

    def extract_features(self, data):
        features = extract_features(
            data,
            column_sort="time",
            column_id="id",
            kind_to_fc_parameters=selection_5,
            # default_fc_parameters=EfficientFCParameters(),
            # default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
        )
        return features

    def read_raw_from_id(self, id):
        path = os.path.join(self._path, "sawing_process_data.h5")
        try:
            hf = h5py.File(path, "r")
        except:
            print("Failed to open file: " + str(path))
            exit()
        try:
            data_arr = np.array(hf[id])
        except:
            print("Failed to find dataset: " + str(id) + " in file " + str(path))
            exit()

        data_arr = np.array(data_arr)
        dataframes = []
        for i in range(len(data_arr)):
            if self.all_features[i] in self.selected_features:
                data_df = pd.DataFrame(
                    data=np.array([data_arr[i][0], data_arr[i][1]]).transpose(),
                    columns=[self.all_features[i], "time"],
                    index=[k for k in range(len(data_arr[i][0]))],
                )
                data_df.insert(0, "id", [id for _ in range(len(data_arr[i][0]))])
                dataframes.append(data_df)
        return dataframes

    def get_process_QH_id(self, id):
        dataframes = self.read_raw_from_id(id)
        features_dataframe = self.extract_features(dataframes.pop(0))
        for dataframe in dataframes:
            features = self.extract_features(dataframe)
            features_dataframe = pd.concat([features_dataframe, features], axis=1)
        return features_dataframe


if __name__ == "__main__":
    pass
