# HOI4D sequence configuration
#
# Sequence naming convention:
#   ZY{date}_{H}_{C#}_{N#}_{S}_{s}_{T}
#   C# = category, N# = object instance ID
#
# Category → CAD model name mapping (from Hoi4dReader):
#   C1  → ToyCar
#   C2  → Mug
#   C5  → Bottle
#   C7  → Bowl
#   C12 → Kettle
#   C13 → Knife
#
# Object model key: {CategoryName}{N:03d}
# Model path: {dataset_root}/models/{obj_name}/textured_simple.obj

sequence_name_list = [
    # Bowl (C7)
    "ZY20210800002_H2_C7_N41_S57_s04_T1",
    "ZY20210800002_H2_C7_N42_S57_s05_T1",
    "ZY20210800002_H2_C7_N45_S58_s01_T1",
    "ZY20210800002_H2_C7_N46_S58_s02_T1",
    "ZY20210800002_H2_C7_N47_S58_s02_T1",
    "ZY20210800002_H2_C7_N48_S58_s03_T1",
    "ZY20210800002_H2_C7_N49_S58_s03_T1",
    "ZY20210800002_H2_C7_N50_S58_s04_T1",
    # ToyCar (C1) – subject H3
    "ZY20210800003_H3_C1_N44_S84_s01_T2",
    "ZY20210800003_H3_C1_N45_S84_s02_T1",
    # Kettle (C12) – subject H4
    "ZY20210800004_H4_C12_N44_S235_s05_T1",
    "ZY20210800004_H4_C12_N45_S236_s01_T1",
    "ZY20210800004_H4_C12_N46_S236_s01_T1",
    "ZY20210800004_H4_C12_N47_S236_s01_T1",
    "ZY20210800004_H4_C12_N48_S236_s01_T1",
    "ZY20210800004_H4_C12_N49_S236_s02_T1",
    "ZY20210800004_H4_C12_N50_S186_s02_T2",
    "ZY20210800004_H4_C12_N50_S236_s02_T1",
    # Knife (C13)
    "ZY20210800004_H4_C13_N44_S233_s05_T1",
    "ZY20210800004_H4_C13_N45_S233_s05_T1",
    "ZY20210800004_H4_C13_N46_S186_s01_T3",
    "ZY20210800004_H4_C13_N47_S234_s01_T1",
    "ZY20210800004_H4_C13_N48_S234_s01_T1",
    "ZY20210800004_H4_C13_N49_S234_s01_T1",
    "ZY20210800004_H4_C13_N50_S186_s02_T3",
    "ZY20210800004_H4_C13_N50_S234_s02_T1",
    # ToyCar (C1) – subject H4
    "ZY20210800004_H4_C1_N46_S240_s02_T2",
    "ZY20210800004_H4_C1_N47_S240_s02_T2",
    "ZY20210800004_H4_C1_N48_S240_s02_T2",
    "ZY20210800004_H4_C1_N49_S240_s03_T2",
    "ZY20210800004_H4_C1_N50_S240_s03_T2",
    "ZY20210800004_H4_C1_N50_S6_s02_T1",
    # Mug (C2)
    "ZY20210800004_H4_C2_N40_S10_s05_T1",
    "ZY20210800004_H4_C2_N43_S15_s01_T5",
    "ZY20210800004_H4_C2_N44_S364_s05_T6",
    "ZY20210800004_H4_C2_N45_S5_s01_T1",
    "ZY20210800004_H4_C2_N47_S16_s02_T1",
    "ZY20210800004_H4_C2_N48_S16_s02_T1",
    "ZY20210800004_H4_C2_N48_S16_s02_T5",
    "ZY20210800004_H4_C2_N49_S16_s03_T1",
    "ZY20210800004_H4_C2_N49_S16_s03_T5",
    # Bottle (C5)
    "ZY20210800004_H4_C5_N20_S109_s03_T5",
    "ZY20210800004_H4_C5_N23_S109_s04_T5",
    "ZY20210800004_H4_C5_N26_S113_s01_T1",
    "ZY20210800004_H4_C5_N27_S113_s01_T5",
    "ZY20210800004_H4_C5_N28_S113_s02_T5",
    "ZY20210800004_H4_C5_N29_S113_s02_T5",
    "ZY20210800004_H4_C5_N31_S56_s03_T1",
    "ZY20210800004_H4_C5_N32_S56_s03_T1",
]

sequences = {
    "default": {
        "geometry_poor_frames": [],
        "cond_idx": 0,
        "cond_select_strategy": "manual",  # manual or object_hand_ratio
        "frame_star": 0,
        "frame_end": 9999,
        "frame_interval": 1,
        "frame_number": 300,
        "obj_num": 6,
        "obj_1_cond_idx": 0,
        "obj_2_cond_idx": 0,
        "obj_3_cond_idx": 0,
        "obj_4_cond_idx": 0,
        "obj_name": None,  # CAD model key, e.g. "Bowl041"
    },

    ########## Bowl (C7) ##########
    "ZY20210800002_H2_C7_N41_S57_s04_T1": {"cond_idx": 0, "obj_name": "Bowl041"},
    "ZY20210800002_H2_C7_N42_S57_s05_T1": {"cond_idx": 0, "obj_name": "Bowl042"},
    "ZY20210800002_H2_C7_N45_S58_s01_T1": {"cond_idx": 0, "obj_name": "Bowl045"},
    "ZY20210800002_H2_C7_N46_S58_s02_T1": {"cond_idx": 0, "obj_name": "Bowl046"},
    "ZY20210800002_H2_C7_N47_S58_s02_T1": {"cond_idx": 0, "obj_name": "Bowl047"},
    "ZY20210800002_H2_C7_N48_S58_s03_T1": {"cond_idx": 0, "obj_name": "Bowl048"},
    "ZY20210800002_H2_C7_N49_S58_s03_T1": {"cond_idx": 0, "obj_name": "Bowl049"},
    "ZY20210800002_H2_C7_N50_S58_s04_T1": {"cond_idx": 0, "obj_name": "Bowl050"},

    ########## ToyCar (C1) – H3 ##########
    "ZY20210800003_H3_C1_N44_S84_s01_T2": {"cond_idx": 0, "obj_name": "ToyCar044"},
    "ZY20210800003_H3_C1_N45_S84_s02_T1": {"cond_idx": 0, "obj_name": "ToyCar045"},

    ########## Kettle (C12) ##########
    "ZY20210800004_H4_C12_N44_S235_s05_T1": {"cond_idx": 0, "obj_name": "Kettle044"},
    "ZY20210800004_H4_C12_N45_S236_s01_T1": {"cond_idx": 0, "obj_name": "Kettle045"},
    "ZY20210800004_H4_C12_N46_S236_s01_T1": {"cond_idx": 0, "obj_name": "Kettle046"},
    "ZY20210800004_H4_C12_N47_S236_s01_T1": {"cond_idx": 0, "obj_name": "Kettle047"},
    "ZY20210800004_H4_C12_N48_S236_s01_T1": {"cond_idx": 0, "obj_name": "Kettle048"},
    "ZY20210800004_H4_C12_N49_S236_s02_T1": {"cond_idx": 0, "obj_name": "Kettle049"},
    "ZY20210800004_H4_C12_N50_S186_s02_T2": {"cond_idx": 0, "obj_name": "Kettle050"},
    "ZY20210800004_H4_C12_N50_S236_s02_T1": {"cond_idx": 0, "obj_name": "Kettle050"},

    ########## Knife (C13) ##########
    "ZY20210800004_H4_C13_N44_S233_s05_T1": {"cond_idx": 0, "obj_name": "Knife044"},
    "ZY20210800004_H4_C13_N45_S233_s05_T1": {"cond_idx": 0, "obj_name": "Knife045"},
    "ZY20210800004_H4_C13_N46_S186_s01_T3": {"cond_idx": 0, "obj_name": "Knife046"},
    "ZY20210800004_H4_C13_N47_S234_s01_T1": {"cond_idx": 0, "obj_name": "Knife047"},
    "ZY20210800004_H4_C13_N48_S234_s01_T1": {"cond_idx": 0, "obj_name": "Knife048"},
    "ZY20210800004_H4_C13_N49_S234_s01_T1": {"cond_idx": 0, "obj_name": "Knife049"},
    "ZY20210800004_H4_C13_N50_S186_s02_T3": {"cond_idx": 0, "obj_name": "Knife050"},
    "ZY20210800004_H4_C13_N50_S234_s02_T1": {"cond_idx": 0, "obj_name": "Knife050"},

    ########## ToyCar (C1) – H4 ##########
    "ZY20210800004_H4_C1_N46_S240_s02_T2": {"cond_idx": 0, "obj_name": "ToyCar046"},
    "ZY20210800004_H4_C1_N47_S240_s02_T2": {"cond_idx": 0, "obj_name": "ToyCar047"},
    "ZY20210800004_H4_C1_N48_S240_s02_T2": {"cond_idx": 0, "obj_name": "ToyCar048"},
    "ZY20210800004_H4_C1_N49_S240_s03_T2": {"cond_idx": 0, "obj_name": "ToyCar049"},
    "ZY20210800004_H4_C1_N50_S240_s03_T2": {"cond_idx": 0, "obj_name": "ToyCar050"},
    "ZY20210800004_H4_C1_N50_S6_s02_T1":   {"cond_idx": 0, "obj_name": "ToyCar050"},

    ########## Mug (C2) ##########
    "ZY20210800004_H4_C2_N40_S10_s05_T1":   {"cond_idx": 0, "obj_name": "Mug040"},
    "ZY20210800004_H4_C2_N43_S15_s01_T5":   {"cond_idx": 0, "obj_name": "Mug043"},
    "ZY20210800004_H4_C2_N44_S364_s05_T6":  {"cond_idx": 0, "obj_name": "Mug044"},
    "ZY20210800004_H4_C2_N45_S5_s01_T1":    {"cond_idx": 0, "obj_name": "Mug045"},
    "ZY20210800004_H4_C2_N47_S16_s02_T1":   {"cond_idx": 0, "obj_name": "Mug047"},
    "ZY20210800004_H4_C2_N48_S16_s02_T1":   {"cond_idx": 0, "obj_name": "Mug048"},
    "ZY20210800004_H4_C2_N48_S16_s02_T5":   {"cond_idx": 0, "obj_name": "Mug048"},
    "ZY20210800004_H4_C2_N49_S16_s03_T1":   {"cond_idx": 0, "obj_name": "Mug049"},
    "ZY20210800004_H4_C2_N49_S16_s03_T5":   {"cond_idx": 0, "obj_name": "Mug049"},

    ########## Bottle (C5) ##########
    "ZY20210800004_H4_C5_N20_S109_s03_T5":  {"cond_idx": 0, "obj_name": "Bottle020"},
    "ZY20210800004_H4_C5_N23_S109_s04_T5":  {"cond_idx": 0, "obj_name": "Bottle023"},
    "ZY20210800004_H4_C5_N26_S113_s01_T1":  {"cond_idx": 0, "obj_name": "Bottle026"},
    "ZY20210800004_H4_C5_N27_S113_s01_T5":  {"cond_idx": 0, "obj_name": "Bottle027"},
    "ZY20210800004_H4_C5_N28_S113_s02_T5":  {"cond_idx": 0, "obj_name": "Bottle028"},
    "ZY20210800004_H4_C5_N29_S113_s02_T5":  {"cond_idx": 0, "obj_name": "Bottle029"},
    "ZY20210800004_H4_C5_N31_S56_s03_T1":   {"cond_idx": 0, "obj_name": "Bottle031"},
    "ZY20210800004_H4_C5_N32_S56_s03_T1":   {"cond_idx": 0, "obj_name": "Bottle032"},
}
