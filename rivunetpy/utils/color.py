import numpy as np

def RGB_from_hex(hex: str, norm=True):
    rgb = tuple(int(hex.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
    if norm:
        rgb = np.array(rgb)/255
    return rgb
