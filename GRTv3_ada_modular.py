#!/usr/bin/env python
"""GRTv3_ada，模組化版本（grtv3/）。與 GRTv3_ada_EN.py 相同的流程、參數與資料欄位，不再依賴 PsychoPy Builder。

    python GRTv3_ada_modular.py                      # 需要耳機；--audio-device NAME 指定裝置
    python GRTv3_ada_modular.py --pilot              # 視窗模式
    python GRTv3_ada_modular.py --list-audio         # 只列裝置

邏輯層（grtv3/plan.py、colour.py、masking.py、calibration.py）不需要 PsychoPy：pytest tests/test_grtv3.py
"""
import sys

from grtv3.session import run_session

if __name__ == "__main__":
    run_session(sys.argv, require_headphones="--no-headphone-check" not in sys.argv)
