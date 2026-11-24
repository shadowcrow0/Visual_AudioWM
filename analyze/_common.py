"""analyze/ 的共用工具：路徑、字型、讀檔。

為什麼有這個檔：每支分析程式都要 (1) 從 repo 根目錄找 data/ 與 stimuli/，不管使用者從哪裡下指令；
(2) 用同一套中文字型設定；(3) 用同一條規則把原始 csv 變成「有反應的試次」。寫在一處，數字才不會因為
各支程式的過濾條件不同而對不上。

這個檔不 import PyMC。adaptivesft 套件的 __init__ 會 import PyMC，所以 load_sic() 只載 sic.py 一個檔
（做法同 adaptive_vawm_EN.load_psi_module），沒有 PyMC 的機器也能跑存活函數檢定。
"""
import ast
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "analyze", "out")

IPA = {"theta": "θ", "eth": "ð", "sh": "ʃ", "zh": "ʒ"}
RT_WINDOW = (0.15, 3.0)          # 秒；校準與主實驗的正確 RT 都用這個窗（與 SFT_WM_nochange_EN.fit_calibration 相同）
RED, BLUE, GREY, ORANGE, GREEN = "#c0392b", "#2471a3", "#7f8c8d", "#e67e22", "#27ae60"


def ipa(c):
    return IPA.get(c, c)


def path(p):
    """使用者給的路徑：先照原樣（相對於目前目錄），找不到再當作相對於 repo 根。"""
    if os.path.isabs(p) or os.path.exists(p):
        return p
    return os.path.join(ROOT, p)


def out_dir(d=None):
    d = d or OUT
    os.makedirs(d, exist_ok=True)
    return d


def setup_plot():
    """matplotlib：Agg 後端、中文用 WenQuanYi Zen Hei、IPA 與缺字退到 DejaVu Sans、負號用 ASCII（該中文字型沒有 U+2212）。"""
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "figure.dpi": 200,
                         "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def load_perception(participant, snrs):
    """sound_perception_EN.py --snr 的輸出 data/<id>_sound_perception_dprime_snr<dB>.csv，多個 SNR 疊成一張表。
    said_diff：按 n（不一樣）= 1、按 y = 0、沒按 = NaN。無反應的試次留在表裡，由各分析自己決定怎麼算。"""
    import numpy as np
    import pandas as pd
    frames = []
    for s in snrs:
        f = os.path.join(DATA, f"{participant}_sound_perception_dprime_snr{s:g}.csv")
        d = pd.read_csv(f).assign(snr=float(s))
        d["said_diff"] = d.key.eq("n").astype(float)
        d.loc[d.key.isna(), "said_diff"] = np.nan
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_calib_trials(f):
    """*_simple_calib_trials.csv。回傳 (全部, 擬合用的「不同」試次, 「相同」試次)，後兩者都只取 other == 'T'：
    'T' 是探測落在目標位置的試次；'H' / 'L' 是插進去的確認試次，不進擬合（SFT_WM_nochange_EN.fit_calibration 同一條規則）。"""
    import pandas as pd
    d = pd.read_csv(path(f))
    t = d[d.other == "T"]
    return d, t[(t.is_match == 0) & t.rt.between(*RT_WINDOW)], t[t.is_match == 1]


def load_main(f):
    """*_simple_main.csv（每個探測一列）。回傳 (正式 block 的全部列, 有反應的列)；block 0 是練習，不算。"""
    import pandas as pd
    m = pd.read_csv(path(f))
    m = m[m.block >= 1].copy()
    m["rt"] = m["ResponseBox.rt"]
    return m, m[m["ResponseBox.keys"].notna()].copy()


def load_sic():
    """adaptiveSFT/adaptivesft/sic.py（numpy + scipy），不經過套件 __init__（那會 import PyMC）。"""
    import adaptive_vawm_EN as A
    f = os.path.join(A.adaptivesft_root(), "adaptivesft", "sic.py")
    spec = importlib.util.spec_from_file_location("adaptivesft_sic", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def experiment_fit_kw():
    """實驗用的取樣設定 FIT_KW 只定義在 SFT_WM_nochange_EN.py（ARCHITECTURE §3），但那個檔一 import 就要 PsychoPy。
    所以用 ast 把那一行賦值讀出來，不執行整個檔；這樣重算用的設定永遠跟實驗機上的一樣，不用抄一份。"""
    src = open(os.path.join(ROOT, "SFT_WM_nochange_EN.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "FIT_KW" for t in node.targets):
            return eval(compile(ast.Expression(node.value), "FIT_KW", "eval"), {"dict": dict})
    raise RuntimeError("FIT_KW not found in SFT_WM_nochange_EN.py")
