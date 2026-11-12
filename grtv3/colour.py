"""顏色軸：agrt_colour_lut.json 的弧長（dE00）→ hex。LUT 由 agrt_setup.export_lut() 離線產生（需要 colour-science）。"""
import json
import os

import numpy as np


class ColourLUT:
    def __init__(self, path=None):
        path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agrt_colour_lut.json")
        with open(path, encoding="utf-8") as f:
            lut = json.load(f)
        self.meta = {k: lut[k] for k in ("anchor_h", "lstar", "cstar", "arc_min", "step") if k in lut}
        self.hex = list(lut["hex"])
        self.arc = lut["arc_min"] + np.arange(len(self.hex)) * lut["step"]

    @property
    def arc_min(self):
        return float(self.arc[0])

    @property
    def arc_max(self):
        return float(self.arc[-1])

    @property
    def arc_lim(self):
        """對稱可用範圍 ±arc_lim（Psi 的 dim1range）。"""
        return float(min(-self.arc_min, self.arc_max))

    def hex_for(self, arc):
        """超出範圍就丟錯，不要悄悄截掉（GRTv3_ada_EN.py:488-493）。"""
        if not (self.arc_min <= arc <= self.arc_max):
            raise ValueError(f"colour arc length {arc} is outside the usable range [{self.arc_min:.2f}, {self.arc_max:.2f}] dE00")
        return self.hex[int(np.abs(self.arc - arc).argmin())]

    def clip(self, arc):
        return float(np.clip(arc, self.arc_min, self.arc_max))
