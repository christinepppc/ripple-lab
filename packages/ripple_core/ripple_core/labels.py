"""
Channel labeling and region categorization utilities.

This module provides functions for mapping electrode channels to anatomical labels
and categorizing bipolar pairs into brain regions.
"""

from typing import Dict, Tuple
import pandas as pd
from pathlib import Path

# Anatomical label mapping for all 220 channels
CHANNEL_LABELS: Dict[int, str] = {
    1: "r_medial_orbital_gyrus",
    2: "r_frontal_white_matter",
    3: "r_lateral_orbital_gyrus",
    4: "r_lateral_orbital_gyrus",
    5: "r_medial_orbital_gyrus",
    6: "r_genu_of_the_corpus_callosum",
    7: "r_lateral_orbital_gyrus",
    8: "r_medial_orbital_gyrus",
    9: "r_lateral_orbital_gyrus",
    10: "r_frontal_white_matter",
    11: "r_putamen",
    12: "r_putamen",
    13: "r_middle_frontal_gyrus",
    14: "r_frontal_white_matter",
    15: "r_caudate_nucleus",
    16: "r_frontal_white_matter",
    17: "r_putamen",
    18: "r_caudate_nucleus",
    19: "r_middle_frontal_gyrus",
    20: "r_caudate_nucleus",
    21: "r_putamen",
    22: "r_putamen",
    23: "r_caudate_nucleus",
    24: "r_anterior_cingulate_gyrus",
    25: "r_frontal_white_matter",
    26: "r_frontal_white_matter",
    27: "r_caudate_nucleus",
    28: "r_frontal_white_matter",
    29: "r_putamen",
    30: "r_putamen",
    31: "r_anterior_amygdalar_area",
    32: "r_caudate_nucleus",
    33: "r_caudate_nucleus",
    34: "r_anterior_cingulate_gyrus",
    35: "r_caudate_nucleus",
    36: "r_frontal_white_matter",
    37: "r_internal_capsule",
    38: "r_caudate_nucleus",
    39: "r_frontal_white_matter",
    40: "r_lateral_globus_pallidus",
    41: "r_precentral_gyrus",
    42: "r_frontal_white_matter",
    43: "r_central_amygdalar_nucleus",
    44: "r_internal_capsule",
    45: "r_precentral_gyrus",
    46: "r_frontal_white_matter",
    47: "r_anterior_cingulate_gyrus",
    48: "r_anterior_cingulate_gyrus",
    49: "r_precentral_gyrus",
    50: "r_precentral_gyrus",
    51: "r_precentral_gyrus",
    52: "r_precentral_gyrus",
    53: "r_cerebral_white_matter",
    54: "r_cerebral_white_matter",
    55: "r_cerebral_white_matter",
    56: "r_caudate_nucleus",
    57: "r_cerebral_white_matter",
    58: "r_caudate_nucleus",
    59: "r_cerebral_white_matter",
    60: "r_cerebral_white_matter",
    61: "r_putamen",
    62: "r_putamen",
    63: "r_cerebral_white_matter",
    64: "r_cerebral_white_matter",
    65: "r_posterior_cingulate_gyrus",
    66: "r_cerebral_white_matter",
    67: "r_cerebral_white_matter",
    68: "r_precentral_gyrus",
    69: "r_cerebral_white_matter",
    70: "r_cerebral_white_matter",
    71: "r_cerebral_white_matter",
    72: "r_cerebral_white_matter",
    73: "r_cerebral_white_matter",
    74: "r_cerebral_white_matter",
    75: "r_cerebral_white_matter",
    76: "r_cerebral_white_matter",
    77: "r_presubiculum",
    78: "r_cerebral_white_matter",
    79: "r_postcentral_gyrus",
    80: "r_cerebral_white_matter",
    81: "r_cerebral_white_matter",
    82: "r_posterior_cingulate_gyrus",
    83: "r_superior_parietal_lobule",
    84: "r_precuneus",
    85: "r_cerebral_white_matter",
    86: "r_supramarginal_gyrus",
    87: "r_cerebral_white_matter",
    88: "r_superior_parietal_lobule",
    89: "r_superior_parietal_lobule",
    90: "r_superior_parietal_lobule",
    91: "r_supramarginal_gyrus",
    92: "r_supramarginal_gyrus",
    93: "r_supramarginal_gyrus",
    94: "r_cerebral_white_matter",
    95: "r_cerebral_white_matter",
    96: "r_supramarginal_gyrus",
    97: "r_lateral_orbital_gyrus",
    98: "r_lateral_orbital_gyrus",
    99: "r_fronto-orbital_gyrus_(macaque)",
    100: "r_frontal_white_matter",
    101: "r_frontal_white_matter",
    102: "r_frontal_white_matter",
    103: "r_superior_frontal_gyrus",
    104: "r_anterior_cingulate_gyrus",
    105: "r_anterior_cingulate_gyrus",
    106: "r_frontal_white_matter",
    107: "r_nucleus_accumbens",
    108: "r_caudate_nucleus",
    109: "r_middle_frontal_gyrus",
    110: "r_caudate_nucleus",
    111: "r_frontal_white_matter",
    112: "r_anterior_cingulate_gyrus",
    113: "r_caudate_nucleus",
    114: "r_caudate_nucleus",
    115: "r_superior_frontal_gyrus",
    116: "r_putamen",
    117: "r_frontal_white_matter",
    118: "r_putamen",
    119: "r_caudate_nucleus",
    120: "r_caudate_nucleus",
    121: "r_internal_capsule",
    122: "r_caudate_nucleus",
    123: "r_lateral_globus_pallidus",
    124: "r_anterior_amygdalar_area",
    125: "r_frontal_white_matter",
    126: "r_lateral_globus_pallidus",
    127: "r_putamen",
    128: "r_frontal_white_matter",
    129: "r_anterior_cingulate_gyrus",
    130: "r_anterior_cingulate_gyrus",
    131: "r_frontal_white_matter",
    132: "r_precentral_gyrus",
    133: "r_anterior_cingulate_gyrus",
    134: "r_frontal_white_matter",
    135: "r_cerebral_white_matter",
    136: "r_cerebral_white_matter",
    137: "r_cerebral_white_matter",
    138: "r_cerebral_white_matter",
    139: "r_precentral_gyrus",
    140: "",  # Empty label
    141: "r_cerebral_white_matter",
    142: "r_cerebral_white_matter",
    143: "r_cerebral_white_matter",
    144: "r_cerebral_white_matter",
    145: "r_cerebral_white_matter",
    146: "r_cerebral_white_matter",
    147: "r_cerebral_white_matter",
    148: "r_cerebral_white_matter",
    149: "r_cerebral_white_matter",
    150: "r_cerebral_white_matter",
    151: "r_cerebral_white_matter",
    152: "r_cerebral_white_matter",
    153: "r_cerebral_white_matter",
    154: "",  # Empty label
    155: "r_thalamus",
    156: "r_cerebral_white_matter",
    157: "r_cerebral_white_matter",
    158: "r_cerebral_white_matter",
    159: "r_postcentral_gyrus",
    160: "r_optic_tract",
    161: "r_posterior_cingulate_gyrus",
    162: "r_precentral_gyrus",
    163: "r_cerebral_white_matter",
    164: "r_cerebral_white_matter",
    165: "r_cerebral_white_matter",
    166: "r_precentral_gyrus",
    167: "r_cerebral_white_matter",
    168: "r_cerebral_white_matter",
    169: "r_precentral_gyrus",
    170: "r_precentral_gyrus",
    171: "r_cerebral_white_matter",
    172: "r_posterior_cingulate_gyrus",
    173: "r_cerebral_white_matter",
    174: "r_cerebral_white_matter",
    175: "r_cerebral_white_matter",
    176: "r_cerebral_white_matter",
    177: "r_supramarginal_gyrus",
    178: "r_cerebral_white_matter",
    179: "r_posterior_cingulate_gyrus",
    180: "",  # Empty label
    181: "r_posterior_cingulate_gyrus",
    182: "r_cerebral_white_matter",
    183: "r_postcentral_gyrus",
    184: "",  # Empty label
    185: "r_posterior_cingulate_gyrus",
    186: "r_posterior_cingulate_gyrus",
    187: "r_postcentral_gyrus",
    188: "l_postcentral_gyrus",
    189: "r_precuneus",
    190: "r_postcentral_gyrus",
    191: "r_supramarginal_gyrus",
    192: "r_cerebral_white_matter",
    193: "r_superior_parietal_lobule",
    194: "l_postcentral_gyrus",
    195: "r_precuneus",
    196: "r_precuneus",
    197: "r_precuneus",
    198: "",  # Empty label
    199: "r_cerebral_white_matter",
    200: "r_cerebral_white_matter",
    201: "",  # Empty label
    202: "",  # Empty label
    203: "r_precuneus",
    204: "r_cerebral_white_matter",
    205: "r_superior_parietal_lobule",
    206: "r_superior_parietal_lobule",
    207: "",  # Empty label
    208: "r_cerebral_white_matter",
    209: "r_precuneus",
    210: "",  # Empty label
    211: "r_superior_parietal_lobule",
    212: "r_cerebral_white_matter",
    213: "r_supramarginal_gyrus",
    214: "r_supramarginal_gyrus",
    215: "",  # Empty label
    216: "",  # Empty label
    217: "r_superior_parietal_lobule",
    218: "r_cerebral_white_matter",
    219: "r_supramarginal_gyrus",
    220: "r_superior_parietal_lobule",
}


def get_region_category(label: str) -> str:
    """
    Categorize an anatomical label into a broader brain region.
    
    Parameters
    ----------
    label : str
        Anatomical label (e.g., "r_superior_parietal_lobule")
    
    Returns
    -------
    region : str
        Broad region category: "prefrontal", "motor", "parietal", 
        "basal_ganglia", "amygdala", "mtl", "thalamus", "white_matter", 
        "other", or "unknown"
    """
    label_lower = label.lower()
    
    # White matter (check first - highest priority)
    if "white_matter" in label_lower or "white matter" in label_lower:
        return "white_matter"
    
    # Parietal (including posterior cingulate!)
    if any(x in label_lower for x in ["parietal", "precuneus", "postcentral", "posterior_cingulate"]):
        return "parietal"
    
    # Motor
    if "precentral" in label_lower:
        return "motor"
    
    # Prefrontal (including anterior cingulate, orbital gyri)
    if any(x in label_lower for x in [
        "frontal", "anterior_cingulate", "orbital"
    ]):
        return "prefrontal"
    
    # Basal ganglia
    if any(x in label_lower for x in [
        "caudate", "putamen", "pallidus", "nucleus_accumbens", "accumbens"
    ]):
        return "basal_ganglia"
    
    # MTL (medial temporal lobe)
    if any(x in label_lower for x in ["subiculum", "hippocampus", "entorhinal"]):
        return "mtl"
    
    # Amygdala
    if "amygdala" in label_lower:
        return "amygdala"
    
    # Thalamus
    if "thalamus" in label_lower:
        return "thalamus"
    
    # Supramarginal gyrus → parietal
    if "supramarginal" in label_lower:
        return "parietal"
    
    # Empty or unknown
    if not label or label.strip() == "":
        return "unknown"
    
    # Everything else
    return "other"


def categorize_bipolar_pair(region_i: str, region_j: str) -> str:
    """
    Categorize a bipolar electrode pair based on the regions of its two channels.
    
    Logic:
    - If both channels are in the same valid region → "within_{region}"
    - If one is white matter/unknown and the other is valid → "within_{valid_region}"
    - If both are different valid regions → "cross_{region_i}_{region_j}"
    - Otherwise → "mixed_or_unknown"
    
    Parameters
    ----------
    region_i : str
        Region category of first channel
    region_j : str
        Region category of second channel
    
    Returns
    -------
    region_type : str
        Category of the bipolar pair
    """
    # Both in same valid region
    if region_i == region_j and region_i not in ["white_matter", "unknown"]:
        return f"within_{region_i}"
    
    # One is white matter/unknown, the other is valid
    if (region_i in ["white_matter", "unknown"]) and region_j not in ["white_matter", "unknown"]:
        return f"within_{region_j}"
    
    if (region_j in ["white_matter", "unknown"]) and region_i not in ["white_matter", "unknown"]:
        return f"within_{region_i}"
    
    # Both are different valid regions
    if region_i not in ["white_matter", "unknown"] and region_j not in ["white_matter", "unknown"]:
        return f"cross_{region_i}_{region_j}"
    
    # Both are white matter/unknown, or other edge cases
    return "mixed_or_unknown"


def create_bipolar_channel_labels(pairs_info: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mapping of bipolar channels to their anatomical labels and regions.
    
    Parameters
    ----------
    pairs_info : pd.DataFrame
        DataFrame with columns: bipolar_ch, pair_anchor, pair_ref
    
    Returns
    -------
    labels_df : pd.DataFrame
        DataFrame with columns: bipolar_channel, channel_i, channel_j, 
        label_i, label_j, region_i, region_j, region_type
    """
    bipolar_data = []
    
    for _, row in pairs_info.iterrows():
        bipolar_name = row['bipolar_ch']
        ch_i = int(row['pair_anchor'])
        ch_j = int(row['pair_ref'])
        
        # Get labels
        label_i = CHANNEL_LABELS.get(ch_i, "")
        label_j = CHANNEL_LABELS.get(ch_j, "")
        
        # Categorize regions
        region_i = get_region_category(label_i)
        region_j = get_region_category(label_j)
        
        # Categorize the bipolar pair
        region_type = categorize_bipolar_pair(region_i, region_j)
        
        bipolar_data.append({
            'bipolar_channel': bipolar_name,
            'channel_i': ch_i,
            'channel_j': ch_j,
            'label_i': label_i,
            'label_j': label_j,
            'region_i': region_i,
            'region_j': region_j,
            'region_type': region_type,
        })
    
    return pd.DataFrame(bipolar_data)
