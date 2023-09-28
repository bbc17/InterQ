import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import torch.nn.functional as F

train_test_size = 0.8

# Selected tsfresh features
# n_features = 5
kind_to_fc_parameters_5 = {
    "sawing": {
        "CutCounter": {
            "cwt_coefficients": [
                {"coeff": 6, "w": 20, "widths": (2, 5, 10, 20)},
                {"coeff": 9, "w": 20, "widths": (2, 5, 10, 20)},
                {"coeff": 8, "w": 20, "widths": (2, 5, 10, 20)},
                {"coeff": 7, "w": 20, "widths": (2, 5, 10, 20)},
                {"coeff": 10, "w": 20, "widths": (2, 5, 10, 20)},
            ]
        }
    },
    "side_1_planfraesen": {
        "acc_x": {"ar_coefficient": [{"coeff": 8, "k": 10}]},
        "acc_y": {
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
            "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 7}],
            "friedrich_coefficients": [{"coeff": 0, "m": 3, "r": 30}],
        },
        "acc_z": {"energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 7}]},
    },
    "side_1_aussenkontur_schruppen_schlichten": {
        "acc_y": {
            "fourier_entropy": [{"bins": 100}],
            "benford_correlation": None,
            "median": None,
            "ratio_beyond_r_sigma": [{"r": 7}],
        },
        "acc_z": {"number_peaks": [{"n": 5}]},
    },
    "side_1_nut_seitlich": {
        "acc_y": {
            "change_quantiles": [
                {"f_agg": "mean", "isabs": False, "qh": 0.6, "ql": 0.0},
                {"f_agg": "mean", "isabs": False, "qh": 0.4, "ql": 0.0},
                {"f_agg": "mean", "isabs": False, "qh": 0.6, "ql": 0.2},
            ]
        },
        "acc_z": {
            "time_reversal_asymmetry_statistic": [{"lag": 1}],
            "ratio_beyond_r_sigma": [{"r": 0.5}],
        },
    },
    "side_1_stufenbohrung": {
        "acc_y": {
            "fourier_entropy": [{"bins": 5}, {"bins": 3}, {"bins": 2}, {"bins": 10}]
        },
        "acc_x": {"permutation_entropy": [{"dimension": 7, "tau": 1}]},
    },
    "side_1_endgraten_aussenkontur_bohrungen": {
        "acc_y": {"ar_coefficient": [{"coeff": 10, "k": 10}]},
        "acc_z": {
            "agg_autocorrelation": [{"f_agg": "median", "maxlag": 40}],
            "ar_coefficient": [{"coeff": 9, "k": 10}, {"coeff": 7, "k": 10}],
            "partial_autocorrelation": [{"lag": 2}],
        },
    },
    "side_1_bohren_seitlich": {
        "acc_x": {
            "permutation_entropy": [
                {"dimension": 7, "tau": 1},
                {"dimension": 6, "tau": 1},
            ]
        },
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}],
            "autocorrelation": [{"lag": 5}],
        },
        "acc_y": {"ar_coefficient": [{"coeff": 4, "k": 10}]},
    },
    "side_1_bohren_senken": {
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}],
            "autocorrelation": [{"lag": 5}],
        },
        "acc_x": {
            "ar_coefficient": [{"coeff": 3, "k": 10}],
            "fourier_entropy": [{"bins": 100}],
            "permutation_entropy": [{"dimension": 7, "tau": 1}],
        },
    },
    "side_1_bohren": {
        "acc_x": {
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
            "ar_coefficient": [{"coeff": 4, "k": 10}],
            "change_quantiles": [{"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.6}],
        },
        "acc_z": {
            "agg_autocorrelation": [{"f_agg": "var", "maxlag": 40}],
            "quantile": [{"q": 0.7}],
        },
    },
    "side_1_gewinde_fraesen": {
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}, {"lag": 4}],
            "ar_coefficient": [{"coeff": 3, "k": 10}],
        },
        "acc_x": {
            "ar_coefficient": [{"coeff": 1, "k": 10}],
            "partial_autocorrelation": [{"lag": 3}],
        },
    },
    "side_2_planfraesen": {
        "acc_x": {
            "ratio_beyond_r_sigma": [{"r": 0.5}],
            "spkt_welch_density": [{"coeff": 8}],
        },
        "acc_z": {"spkt_welch_density": [{"coeff": 5}]},
        "acc_y": {
            "augmented_dickey_fuller": [
                {"attr": "pvalue", "autolag": "AIC"},
                {"attr": "teststat", "autolag": "AIC"},
            ]
        },
    },
    "side_2_kreistasche_fraesen": {
        "acc_x": {
            "maximum": None,
            "mean_n_absolute_max": [{"number_of_maxima": 7}],
            "change_quantiles": [
                {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.0}
            ],
            "cid_ce": [{"normalize": False}],
        },
        "acc_z": {"number_crossing_m": [{"m": 1}]},
    },
    "side_2_bauteil_entgraten": {
        "acc_z": {
            "ar_coefficient": [
                {"coeff": 2, "k": 10},
                {"coeff": 9, "k": 10},
                {"coeff": 1, "k": 10},
                {"coeff": 4, "k": 10},
            ]
        },
        "acc_y": {
            "change_quantiles": [{"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.6}]
        },
    },
    "side_2_ringnut": {
        "acc_x": {
            "autocorrelation": [{"lag": 5}],
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.6, "ql": 0.2},
            ],
            "quantile": [{"q": 0.2}],
        }
    },
}

# n_features = 10
kind_to_fc_parameters_10 = {
    "sawing": {
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
    },
    "side_1_planfraesen": {
        "acc_x": {
            "ar_coefficient": [{"coeff": 8, "k": 10}],
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
            "autocorrelation": [{"lag": 9}],
        },
        "acc_y": {
            "permutation_entropy": [
                {"dimension": 3, "tau": 1},
                {"dimension": 4, "tau": 1},
            ],
            "energy_ratio_by_chunks": [
                {"num_segments": 10, "segment_focus": 7},
                {"num_segments": 10, "segment_focus": 6},
            ],
            "friedrich_coefficients": [
                {"coeff": 0, "m": 3, "r": 30},
                {"coeff": 2, "m": 3, "r": 30},
            ],
        },
        "acc_z": {"energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 7}]},
    },
    "side_1_aussenkontur_schruppen_schlichten": {
        "acc_y": {
            "fourier_entropy": [{"bins": 100}],
            "benford_correlation": None,
            "median": None,
            "ratio_beyond_r_sigma": [{"r": 7}],
            "change_quantiles": [
                {"f_agg": "mean", "isabs": False, "qh": 0.4, "ql": 0.2}
            ],
            "agg_autocorrelation": [{"f_agg": "var", "maxlag": 40}],
        },
        "acc_z": {
            "number_peaks": [{"n": 5}],
            "fft_aggregated": [{"aggtype": "skew"}],
            "permutation_entropy": [
                {"dimension": 7, "tau": 1},
                {"dimension": 6, "tau": 1},
            ],
        },
    },
    "side_1_nut_seitlich": {
        "acc_y": {
            "change_quantiles": [
                {"f_agg": "mean", "isabs": False, "qh": 0.6, "ql": 0.0},
                {"f_agg": "mean", "isabs": False, "qh": 0.4, "ql": 0.0},
                {"f_agg": "mean", "isabs": False, "qh": 0.6, "ql": 0.2},
                {"f_agg": "mean", "isabs": False, "qh": 1.0, "ql": 0.2},
            ]
        },
        "acc_z": {
            "time_reversal_asymmetry_statistic": [{"lag": 1}],
            "ratio_beyond_r_sigma": [{"r": 0.5}],
            "number_cwt_peaks": [{"n": 5}],
        },
        "acc_x": {
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.0},
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.2},
                {"f_agg": "var", "isabs": False, "qh": 0.4, "ql": 0.0},
            ]
        },
    },
    "side_1_stufenbohrung": {
        "acc_y": {
            "fourier_entropy": [
                {"bins": 5},
                {"bins": 3},
                {"bins": 2},
                {"bins": 10},
                {"bins": 100},
            ]
        },
        "acc_x": {
            "permutation_entropy": [
                {"dimension": 7, "tau": 1},
                {"dimension": 5, "tau": 1},
                {"dimension": 6, "tau": 1},
            ]
        },
        "acc_z": {
            "permutation_entropy": [
                {"dimension": 3, "tau": 1},
                {"dimension": 4, "tau": 1},
            ]
        },
    },
    "side_1_endgraten_aussenkontur_bohrungen": {
        "acc_y": {"ar_coefficient": [{"coeff": 10, "k": 10}]},
        "acc_z": {
            "agg_autocorrelation": [{"f_agg": "median", "maxlag": 40}],
            "ar_coefficient": [
                {"coeff": 9, "k": 10},
                {"coeff": 7, "k": 10},
                {"coeff": 2, "k": 10},
                {"coeff": 1, "k": 10},
                {"coeff": 10, "k": 10},
            ],
            "partial_autocorrelation": [{"lag": 2}],
            "fft_aggregated": [{"aggtype": "kurtosis"}],
        },
        "acc_x": {"fft_aggregated": [{"aggtype": "kurtosis"}]},
    },
    "side_1_bohren_seitlich": {
        "acc_x": {
            "permutation_entropy": [
                {"dimension": 7, "tau": 1},
                {"dimension": 6, "tau": 1},
                {"dimension": 5, "tau": 1},
            ],
            "ar_coefficient": [{"coeff": 4, "k": 10}],
        },
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}],
            "autocorrelation": [{"lag": 5}],
            "ar_coefficient": [{"coeff": 2, "k": 10}],
            "number_cwt_peaks": [{"n": 5}],
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
        },
        "acc_y": {"ar_coefficient": [{"coeff": 4, "k": 10}]},
    },
    "side_1_bohren_senken": {
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}],
            "autocorrelation": [{"lag": 5}],
            "ar_coefficient": [{"coeff": 2, "k": 10}],
            "number_cwt_peaks": [{"n": 5}],
        },
        "acc_x": {
            "ar_coefficient": [{"coeff": 3, "k": 10}],
            "fourier_entropy": [{"bins": 100}],
            "permutation_entropy": [{"dimension": 7, "tau": 1}],
        },
        "acc_y": {
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 0.6, "ql": 0.0},
                {"f_agg": "mean", "isabs": True, "qh": 0.4, "ql": 0.0},
            ],
            "median": None,
        },
    },
    "side_1_bohren": {
        "acc_x": {
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
            "ar_coefficient": [{"coeff": 4, "k": 10}],
            "change_quantiles": [{"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.6}],
        },
        "acc_z": {
            "agg_autocorrelation": [{"f_agg": "var", "maxlag": 40}],
            "quantile": [{"q": 0.7}, {"q": 0.8}],
            "change_quantiles": [{"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.2}],
            "ratio_value_number_to_time_series_length": None,
            "agg_linear_trend": [
                {"attr": "slope", "chunk_len": 10, "f_agg": "min"},
                {"attr": "slope", "chunk_len": 5, "f_agg": "min"},
            ],
        },
    },
    "side_1_gewinde_fraesen": {
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}, {"lag": 4}],
            "ar_coefficient": [{"coeff": 3, "k": 10}, {"coeff": 8, "k": 10}],
            "autocorrelation": [{"lag": 2}],
        },
        "acc_x": {
            "ar_coefficient": [{"coeff": 1, "k": 10}, {"coeff": 3, "k": 10}],
            "partial_autocorrelation": [{"lag": 3}],
            "fft_aggregated": [{"aggtype": "skew"}],
        },
        "acc_y": {"ar_coefficient": [{"coeff": 2, "k": 10}]},
    },
    "side_2_planfraesen": {
        "acc_x": {
            "ratio_beyond_r_sigma": [{"r": 0.5}],
            "spkt_welch_density": [{"coeff": 8}],
            "ar_coefficient": [{"coeff": 8, "k": 10}],
        },
        "acc_z": {"spkt_welch_density": [{"coeff": 5}, {"coeff": 8}]},
        "acc_y": {
            "augmented_dickey_fuller": [
                {"attr": "pvalue", "autolag": "AIC"},
                {"attr": "teststat", "autolag": "AIC"},
            ],
            "spkt_welch_density": [{"coeff": 5}, {"coeff": 8}],
            "number_peaks": [{"n": 5}],
        },
    },
    "side_2_kreistasche_fraesen": {
        "acc_x": {
            "maximum": None,
            "mean_n_absolute_max": [{"number_of_maxima": 7}],
            "change_quantiles": [
                {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.0},
                {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.0},
            ],
            "cid_ce": [{"normalize": False}],
            "absolute_maximum": None,
            "agg_autocorrelation": [{"f_agg": "mean", "maxlag": 40}],
            "agg_linear_trend": [{"attr": "stderr", "chunk_len": 50, "f_agg": "min"}],
        },
        "acc_z": {
            "number_crossing_m": [{"m": 1}],
            "ratio_beyond_r_sigma": [{"r": 2.5}],
        },
    },
    "side_2_bauteil_entgraten": {
        "acc_z": {
            "ar_coefficient": [
                {"coeff": 2, "k": 10},
                {"coeff": 9, "k": 10},
                {"coeff": 1, "k": 10},
                {"coeff": 4, "k": 10},
                {"coeff": 10, "k": 10},
            ],
            "agg_autocorrelation": [{"f_agg": "median", "maxlag": 40}],
            "max_langevin_fixed_point": [{"m": 3, "r": 30}],
        },
        "acc_y": {
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.6},
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.0},
            ],
            "ar_coefficient": [{"coeff": 10, "k": 10}],
        },
    },
    "side_2_ringnut": {
        "acc_x": {
            "autocorrelation": [{"lag": 5}],
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.6, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.4},
            ],
            "quantile": [{"q": 0.2}, {"q": 0.7}, {"q": 0.3}, {"q": 0.8}, {"q": 0.6}],
        }
    },
}

kind_to_fc_parameters_15 = {
    "sawing": {
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
    },
    "side_1_planfraesen": {
        "acc_x": {
            "ar_coefficient": [{"coeff": 8, "k": 10}],
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
            "autocorrelation": [{"lag": 9}],
            "change_quantiles": [{"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.2}],
        },
        "acc_y": {
            "permutation_entropy": [
                {"dimension": 3, "tau": 1},
                {"dimension": 4, "tau": 1},
            ],
            "energy_ratio_by_chunks": [
                {"num_segments": 10, "segment_focus": 7},
                {"num_segments": 10, "segment_focus": 6},
            ],
            "friedrich_coefficients": [
                {"coeff": 0, "m": 3, "r": 30},
                {"coeff": 2, "m": 3, "r": 30},
            ],
            "benford_correlation": None,
        },
        "acc_z": {
            "energy_ratio_by_chunks": [
                {"num_segments": 10, "segment_focus": 7},
                {"num_segments": 10, "segment_focus": 4},
            ],
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
            "ratio_beyond_r_sigma": [{"r": 10}],
        },
    },
    "side_1_aussenkontur_schruppen_schlichten": {
        "acc_y": {
            "fourier_entropy": [{"bins": 100}],
            "benford_correlation": None,
            "median": None,
            "ratio_beyond_r_sigma": [{"r": 7}, {"r": 2}, {"r": 2.5}],
            "change_quantiles": [
                {"f_agg": "mean", "isabs": False, "qh": 0.4, "ql": 0.2}
            ],
            "agg_autocorrelation": [{"f_agg": "var", "maxlag": 40}],
            "autocorrelation": [{"lag": 4}],
        },
        "acc_z": {
            "number_peaks": [{"n": 5}],
            "fft_aggregated": [{"aggtype": "skew"}],
            "permutation_entropy": [
                {"dimension": 7, "tau": 1},
                {"dimension": 6, "tau": 1},
                {"dimension": 3, "tau": 1},
            ],
            "friedrich_coefficients": [{"coeff": 2, "m": 3, "r": 30}],
        },
    },
    "side_1_nut_seitlich": {
        "acc_y": {
            "change_quantiles": [
                {"f_agg": "mean", "isabs": False, "qh": 0.6, "ql": 0.0},
                {"f_agg": "mean", "isabs": False, "qh": 0.4, "ql": 0.0},
                {"f_agg": "mean", "isabs": False, "qh": 0.6, "ql": 0.2},
                {"f_agg": "mean", "isabs": False, "qh": 1.0, "ql": 0.2},
                {"f_agg": "mean", "isabs": False, "qh": 0.4, "ql": 0.2},
            ]
        },
        "acc_z": {
            "time_reversal_asymmetry_statistic": [{"lag": 1}],
            "ratio_beyond_r_sigma": [{"r": 0.5}],
            "number_cwt_peaks": [{"n": 5}],
        },
        "acc_x": {
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.0},
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.2},
                {"f_agg": "var", "isabs": False, "qh": 0.4, "ql": 0.0},
                {"f_agg": "var", "isabs": True, "qh": 0.6, "ql": 0.0},
                {"f_agg": "var", "isabs": False, "qh": 0.6, "ql": 0.0},
            ],
            "quantile": [{"q": 0.1}],
            "agg_linear_trend": [
                {"attr": "intercept", "chunk_len": 10, "f_agg": "min"}
            ],
        },
    },
    "side_1_stufenbohrung": {
        "acc_y": {
            "fourier_entropy": [
                {"bins": 5},
                {"bins": 3},
                {"bins": 2},
                {"bins": 10},
                {"bins": 100},
            ]
        },
        "acc_x": {
            "permutation_entropy": [
                {"dimension": 7, "tau": 1},
                {"dimension": 5, "tau": 1},
                {"dimension": 6, "tau": 1},
                {"dimension": 4, "tau": 1},
            ],
            "autocorrelation": [{"lag": 5}],
        },
        "acc_z": {
            "permutation_entropy": [
                {"dimension": 3, "tau": 1},
                {"dimension": 4, "tau": 1},
            ],
            "friedrich_coefficients": [{"coeff": 2, "m": 3, "r": 30}],
            "index_mass_quantile": [{"q": 0.4}, {"q": 0.3}],
        },
    },
    "side_1_endgraten_aussenkontur_bohrungen": {
        "acc_y": {"ar_coefficient": [{"coeff": 10, "k": 10}]},
        "acc_z": {
            "agg_autocorrelation": [{"f_agg": "median", "maxlag": 40}],
            "ar_coefficient": [
                {"coeff": 9, "k": 10},
                {"coeff": 7, "k": 10},
                {"coeff": 2, "k": 10},
                {"coeff": 1, "k": 10},
                {"coeff": 10, "k": 10},
            ],
            "partial_autocorrelation": [{"lag": 2}],
            "fft_aggregated": [{"aggtype": "kurtosis"}, {"aggtype": "variance"}],
            "energy_ratio_by_chunks": [
                {"num_segments": 10, "segment_focus": 8},
                {"num_segments": 10, "segment_focus": 1},
            ],
            "agg_linear_trend": [
                {"attr": "stderr", "chunk_len": 50, "f_agg": "min"},
                {"attr": "stderr", "chunk_len": 50, "f_agg": "max"},
            ],
        },
        "acc_x": {"fft_aggregated": [{"aggtype": "kurtosis"}]},
    },
    "side_1_bohren_seitlich": {
        "acc_x": {
            "permutation_entropy": [
                {"dimension": 7, "tau": 1},
                {"dimension": 6, "tau": 1},
                {"dimension": 5, "tau": 1},
            ],
            "ar_coefficient": [{"coeff": 4, "k": 10}, {"coeff": 5, "k": 10}],
        },
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}],
            "autocorrelation": [{"lag": 5}],
            "ar_coefficient": [{"coeff": 2, "k": 10}],
            "number_cwt_peaks": [{"n": 5}],
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
        },
        "acc_y": {
            "ar_coefficient": [{"coeff": 4, "k": 10}],
            "c3": [{"lag": 3}],
            "agg_linear_trend": [
                {"attr": "intercept", "chunk_len": 5, "f_agg": "min"},
                {"attr": "intercept", "chunk_len": 10, "f_agg": "min"},
            ],
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 0.6, "ql": 0.4}
            ],
        },
    },
    "side_1_bohren_senken": {
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}],
            "autocorrelation": [{"lag": 5}],
            "ar_coefficient": [{"coeff": 2, "k": 10}],
            "number_cwt_peaks": [{"n": 5}],
        },
        "acc_x": {
            "ar_coefficient": [
                {"coeff": 3, "k": 10},
                {"coeff": 5, "k": 10},
                {"coeff": 4, "k": 10},
            ],
            "fourier_entropy": [{"bins": 100}],
            "permutation_entropy": [{"dimension": 7, "tau": 1}],
        },
        "acc_y": {
            "change_quantiles": [
                {"f_agg": "mean", "isabs": True, "qh": 0.6, "ql": 0.0},
                {"f_agg": "mean", "isabs": True, "qh": 0.4, "ql": 0.0},
                {"f_agg": "var", "isabs": False, "qh": 0.4, "ql": 0.0},
                {"f_agg": "var", "isabs": False, "qh": 0.6, "ql": 0.0},
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.0},
            ],
            "median": None,
        },
    },
    "side_1_bohren": {
        "acc_x": {
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
            "ar_coefficient": [{"coeff": 4, "k": 10}],
            "change_quantiles": [{"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.6}],
        },
        "acc_z": {
            "agg_autocorrelation": [{"f_agg": "var", "maxlag": 40}],
            "quantile": [{"q": 0.7}, {"q": 0.8}],
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.2},
                {"f_agg": "var", "isabs": False, "qh": 0.8, "ql": 0.2},
            ],
            "ratio_value_number_to_time_series_length": None,
            "agg_linear_trend": [
                {"attr": "slope", "chunk_len": 10, "f_agg": "min"},
                {"attr": "slope", "chunk_len": 5, "f_agg": "min"},
            ],
            "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
            "percentage_of_reoccurring_values_to_all_values": None,
            "autocorrelation": [{"lag": 3}],
        },
        "acc_y": {"spkt_welch_density": [{"coeff": 8}]},
    },
    "side_1_gewinde_fraesen": {
        "acc_z": {
            "partial_autocorrelation": [{"lag": 2}, {"lag": 4}],
            "ar_coefficient": [
                {"coeff": 3, "k": 10},
                {"coeff": 8, "k": 10},
                {"coeff": 1, "k": 10},
            ],
            "autocorrelation": [{"lag": 2}],
        },
        "acc_x": {
            "ar_coefficient": [{"coeff": 1, "k": 10}, {"coeff": 3, "k": 10}],
            "partial_autocorrelation": [{"lag": 3}],
            "fft_aggregated": [{"aggtype": "skew"}],
            "spkt_welch_density": [{"coeff": 5}],
            "permutation_entropy": [{"dimension": 3, "tau": 1}],
        },
        "acc_y": {
            "ar_coefficient": [{"coeff": 2, "k": 10}, {"coeff": 0, "k": 10}],
            "fft_aggregated": [{"aggtype": "kurtosis"}],
        },
    },
    "side_2_planfraesen": {
        "acc_x": {
            "ratio_beyond_r_sigma": [{"r": 0.5}],
            "spkt_welch_density": [{"coeff": 8}, {"coeff": 5}],
            "ar_coefficient": [{"coeff": 8, "k": 10}],
            "index_mass_quantile": [{"q": 0.6}],
        },
        "acc_z": {
            "spkt_welch_density": [{"coeff": 5}, {"coeff": 8}],
            "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 5}],
            "partial_autocorrelation": [{"lag": 3}],
        },
        "acc_y": {
            "augmented_dickey_fuller": [
                {"attr": "pvalue", "autolag": "AIC"},
                {"attr": "teststat", "autolag": "AIC"},
            ],
            "spkt_welch_density": [{"coeff": 5}, {"coeff": 8}],
            "number_peaks": [{"n": 5}],
            "partial_autocorrelation": [{"lag": 7}],
        },
    },
    "side_2_kreistasche_fraesen": {
        "acc_x": {
            "maximum": None,
            "mean_n_absolute_max": [{"number_of_maxima": 7}],
            "change_quantiles": [
                {"f_agg": "var", "isabs": False, "qh": 1.0, "ql": 0.0},
                {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.0},
                {"f_agg": "mean", "isabs": True, "qh": 1.0, "ql": 0.0},
            ],
            "cid_ce": [{"normalize": False}],
            "absolute_maximum": None,
            "agg_autocorrelation": [{"f_agg": "mean", "maxlag": 40}],
            "agg_linear_trend": [
                {"attr": "stderr", "chunk_len": 50, "f_agg": "min"},
                {"attr": "stderr", "chunk_len": 10, "f_agg": "min"},
                {"attr": "intercept", "chunk_len": 10, "f_agg": "var"},
            ],
            "mean_abs_change": None,
        },
        "acc_z": {
            "number_crossing_m": [{"m": 1}],
            "ratio_beyond_r_sigma": [{"r": 2.5}],
            "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 0}],
        },
    },
    "side_2_bauteil_entgraten": {
        "acc_z": {
            "ar_coefficient": [
                {"coeff": 2, "k": 10},
                {"coeff": 9, "k": 10},
                {"coeff": 1, "k": 10},
                {"coeff": 4, "k": 10},
                {"coeff": 10, "k": 10},
                {"coeff": 7, "k": 10},
            ],
            "agg_autocorrelation": [{"f_agg": "median", "maxlag": 40}],
            "max_langevin_fixed_point": [{"m": 3, "r": 30}],
            "partial_autocorrelation": [{"lag": 8}, {"lag": 4}],
        },
        "acc_y": {
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 1.0, "ql": 0.6},
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.0},
            ],
            "ar_coefficient": [{"coeff": 10, "k": 10}],
            "friedrich_coefficients": [{"coeff": 0, "m": 3, "r": 30}],
        },
        "acc_x": {"skewness": None},
    },
    "side_2_ringnut": {
        "acc_x": {
            "autocorrelation": [{"lag": 5}, {"lag": 3}],
            "change_quantiles": [
                {"f_agg": "var", "isabs": True, "qh": 0.4, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.6, "ql": 0.2},
                {"f_agg": "var", "isabs": True, "qh": 0.8, "ql": 0.4},
                {"f_agg": "var", "isabs": False, "qh": 0.4, "ql": 0.2},
                {"f_agg": "var", "isabs": False, "qh": 0.8, "ql": 0.2},
            ],
            "quantile": [{"q": 0.2}, {"q": 0.7}, {"q": 0.3}, {"q": 0.8}, {"q": 0.6}],
            "fourier_entropy": [{"bins": 10}],
            "agg_autocorrelation": [{"f_agg": "var", "maxlag": 40}],
        }
    },
}

Processes = [
    "saegen",
    "side_1_planfraesen",
    "side_1_aussenkontur_schruppen_schlichten",
    "side_1_nut_seitlich",
    "side_1_stufenbohrung",
    "side_1_endgraten_aussenkontur_bohrungen",
    "side_1_bohren_seitlich",
    "side_1_bohren_senken",
    "side_1_bohren",
    "side_1_gewinde_fraesen",
    "side_2_planfraesen",
    "side_2_kreistasche_fraesen",
    "side_2_bauteil_entgraten",
    "side_2_ringnut",
]

Features_Sawing_10 = ["CutCounter", "Vib01.Peak", "Vib01.RMS"]

Features_Sawing_5 = ["CutCounter"]

Features_Milling = ["acc_x", "acc_y", "acc_z"]

# GNN Training
HYPERPARAMETERS = {
    "random_seed": [8],
    "n_processes": [14],  # total number of (sub-)processes
    "n_aggregated_features": [1],  # number of features for each aggregated feature
    "batch_size": [8],  # [4, 8, 16, 24, 48],
    "learning_rate": [0.001],  # [0.001, 0.0001, 0.00001],
    "weight_decay": [0.0001],  # [0.0001, 0.00001, 0.001],
    "sgd_momentum": [0.9],
    "scheduler_gamma": [1],  # [1, 0.995, 0.9],
    "num_layers": [1],  # [1, 2, 3, 4],
    "hid_feats": [48],  # [24, 48, 96, 192],
    "out_feats": [1],
    "aggregator": ["lstm"],  # ["mean", "gcn", "pool", "lstm"],
    "feat_drop": [0.1],  # [0.1, 0.2, 0.3, 0.4],
    "activation": [F.relu],  # [F.relu, F.leaky_relu, F.elu],
    "conv_type": ["gin"],  # ["sage", "gin", "conv", "pna"],
}

# MLFlow model signature
input_schema = Schema(
    [
        TensorSpec(np.dtype(np.float32), (-1, 1), name="Line"),
        TensorSpec(np.dtype(np.float32), (-1, 1), name="Station"),
        TensorSpec(np.dtype(np.float32), (-1, 1), name="Feature"),
    ]
)

output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)
