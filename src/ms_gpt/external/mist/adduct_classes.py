ADDUCT_CLASSES_TO_IDX = {
    # === 正离子基础 ===
    '[M+H]+': 0,
    '[M+Na]+': 1,
    '[M+K]+': 2,
    '[M+NH4]+': 4,
    '[M+H3N+H]+': 4,  # 别名 -> [M+NH4]+
    '[M]+': 5,
    
    # === 正离子源内丢失 ===
    '[M+H-H2O]+': 3,
    '[M-H2O+H]+': 3,  # 别名
    '[M+H-2H2O]+': 6,
    '[M-H4O2+H]+': 6,  # 别名
    
    # === 负离子基础 ===
    '[M-H]-': 7,
    '[M+Cl]-': 8,
    '[M+FA-H]-': 11,
    '[M+HCOO]-': 11,  # 别名 -> [M+FA-H]-
    '[M+Ac-H]-': 12,
    '[M+CH3COO]-': 12,  # 别名 -> [M+Ac-H]-
    
    # === 负离子源内丢失 ===
    '[M-H-H2O]-': 9,
    '[M-H2O-H]-': 9,  # 别名
    '[M-H-2H2O]-': 13,
    '[M-2H2O-H]-': 13,  # 别名
    '[M-H-CO2]-': 10,
    '[M-CO2-H]-': 10,  # 别名 -> [M-H-CO2]-
    '[M-H-NH3]-': 14,
    '[M-NH3-H]-': 14,  # 别名
    
    # === 负离子金属置换===
    '[M-2H+Na]-': 15,
    '[M-2H+K]-': 16,
    
    # === 正离子金属置换===
    '[M+2Na-H]+': 17,
    '[M+2K-H]+': 18,
}

def get_adduct_class_idx(adduct_class: str) -> int:
    return ADDUCT_CLASSES_TO_IDX[adduct_class]