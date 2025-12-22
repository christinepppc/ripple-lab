# packages/ripple_core/ripple_core/signal/__init__.py

from .pairs import make_bipolar_pairs_from_grid
from .reref import reref_trial
from .process_bipolar import (
    process_bipolar_referencing,
    load_bipolar_dataset,
    get_bipolar_lfp_matrix,
    save_bipolar_analysis,
    create_bipolar_processing_script,
    BipolarChannel,
    BipolarDataset
)
from .car import (
    process_car_referencing,
    load_car_dataset,
    get_car_lfp_matrix,
    save_car_analysis,
    create_car_processing_script,
    CARChannel,
    CARDataset
)

__all__ = [
    "make_bipolar_pairs_from_grid",
    "reref_trial",
    "process_bipolar_referencing",
    "load_bipolar_dataset",
    "get_bipolar_lfp_matrix",
    "save_bipolar_analysis",
    "create_bipolar_processing_script",
    "BipolarChannel",
    "BipolarDataset",
    "process_car_referencing",
    "load_car_dataset",
    "get_car_lfp_matrix",
    "save_car_analysis",
    "create_car_processing_script",
    "CARChannel",
    "CARDataset",
]