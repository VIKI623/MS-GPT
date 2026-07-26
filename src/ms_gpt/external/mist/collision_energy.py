from typing import Optional
import numpy as np


UNKNOWN_CE_ARRAY = np.array([1, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333], dtype=np.float32)

def get_collision_energy_class_array(collision_energy_array, low_collision_energy_ref, mid_collision_energy_ref, high_collision_energy_ref):
    ce = np.asanyarray(collision_energy_array, dtype=np.float32)
    
    col_low = np.interp(ce, [low_collision_energy_ref, mid_collision_energy_ref], [1.0, 0.0], left=1.0, right=0.0)
    col_mid = np.interp(ce, [low_collision_energy_ref, mid_collision_energy_ref, high_collision_energy_ref], [0.0, 1.0, 0.0], left=0.0, right=0.0)
    col_high = np.interp(ce, [mid_collision_energy_ref, high_collision_energy_ref], [0.0, 1.0], left=0.0, right=1.0)
    col_zero = np.zeros_like(ce)
    
    result = np.stack([col_zero, col_low, col_mid, col_high], axis=1, dtype=np.float32)
    
    with np.errstate(invalid='ignore'):
        unknown_mask = (ce < 1) | np.isnan(ce)
    
    if np.any(unknown_mask):
        result[unknown_mask] = UNKNOWN_CE_ARRAY.copy()
        
    return result


def get_normalized_collision_energy(collision_energy: Optional[float], precursor_mz: float, precursor_charge: int = 1) -> float:
    if precursor_mz <= 0:
        raise ValueError("Precursor m/z must be greater than 0.")
    
    if collision_energy is None or collision_energy < 1:
        return 0.0

    # 根据电荷数选择经验性的电荷因子
    # 这些是常用的近似值
    if precursor_charge == 1:
        charge_factor = 1.0
    elif precursor_charge == 2:
        charge_factor = 0.9
    elif precursor_charge == 3:
        charge_factor = 0.85
    else:
        # 对于更高电荷的离子，使用一个通用的折减因子
        charge_factor = 0.8

    # 逆转公式: CE = NCE * (m/z / 500) * charge_factor
    # 推导出: NCE = (CE * 500) / (m/z * charge_factor)
    denominator = precursor_mz * charge_factor

    nce = (collision_energy * 500) / denominator
    
    return nce
