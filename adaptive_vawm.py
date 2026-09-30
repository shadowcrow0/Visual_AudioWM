"""
VAWM_nobox.py 的自適應校準：每位受試者各校一次「顏色差距」與「子音混淆度」，
再用校到的 H / L 產生 stimuli/block1..6.csv（欄位與 practice.csv 完全相同），主實驗照舊跑 VAWM_nobox.py。
不依賴 PsychoPy；畫面在 VAWM_calibrate.py。

    區塊 1  校顏色   探測色 = 目標色 ± x（Lab 距離 x 由 Psi 提），聲音 = 目標音        → ΔE_H、ΔE_L
    區塊 2  校聲音   探測音 = 目標子音的 foil（混淆度由 Psi 提、貼到可用子音），顏色 = 目標色  → count_H、count_L
    穿插 AA 試（探測 = 目標）估假警報率，當 yes/no 心理計量函數的下漸近線。

   受試者的 yes/no                      Psi 的 r
   ────────────────────────────────    ─────────────
   n（不同）                            1
   y（相同）                            0
   P(r=1 | x) = FA + (1 − FA − lapse)·Φ((x − α)/β)        x 越大越容易說「不同」

兩個軸：
   顏色  x = 探測色與目標色的 CIELAB 歐氏距離（colorpool.py 的 delta_e，與現有 block csv 的 *_deltaE 同單位）
   聲音  x = −log10(混淆次數 + 1)（generate_trials.py 的 CONFUSION_DATA，Miller & Nicely 1955）
         count 小 = 不易混淆 = 好分 = 高 salience（現有 csv：audio1_H_count = 1、audio1_L_count = 91）
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _adaptivesft_root():
    """adaptiveSFT repo 的根目錄：$ADAPTIVESFT_PATH，否則本目錄隔壁的 ../adaptiveSFT；找不到回 None。"""
    for c in (os.environ.get("ADAPTIVESFT_PATH"), os.path.join(os.path.dirname(_HERE), "adaptiveSFT")):
        if c and os.path.isfile(os.path.join(c, "adaptivesft", "psi.py")):
            return c
    return None


def _import_psi():
    """
    要的是 adaptiveSFT repo 的 adaptivesft/psi.py（只用 numpy + scipy）。兩個坑：
    (1) 本 repo 自己也有一個舊的 adaptivesft/（PyMC LNRM，沒有 psi），從這個目錄啟動時它會遮住裝好的那個；
    (2) adaptiveSFT 的 adaptivesft/__init__.py 會 import PyMC，實驗機器（PsychoPy）通常沒有 PyMC。
    所以：先試正常 import；不行就從 _adaptivesft_root() 直接按檔案載入 psi.py，不執行套件的 __init__.py。
    實驗端（校準、產生 block csv）只需要這個；分析端（analyze_vawm）才需要完整套件。
    """
    import importlib
    import importlib.util
    import sys
    try:
        m = importlib.import_module("adaptivesft.psi")
        return m.Psi, m.salience_levels
    except ImportError:
        pass
    root = _adaptivesft_root()
    if root is None:
        raise ImportError("找不到 adaptiveSFT 的 adaptivesft/psi.py：設 ADAPTIVESFT_PATH=<adaptiveSFT 路徑>，"
                          "或把 adaptiveSFT clone 在本目錄隔壁（本目錄的 adaptivesft/ 是舊的 LNRM 套件，會遮住它）")
    spec = importlib.util.spec_from_file_location("adaptivesft_psi", os.path.join(root, "adaptivesft", "psi.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules["adaptivesft_psi"] = m
    return m.Psi, m.salience_levels


def _import_experiment():
    """分析端：完整的 adaptivesft 套件（要 PyMC 等）。裝好的優先；否則清掉遮住的舊套件、從 repo 路徑載入。"""
    import importlib
    import sys
    try:
        return importlib.import_module("adaptivesft.experiment")
    except ImportError:
        pass
    root = _adaptivesft_root()
    if root is None:
        raise ImportError("找不到 adaptiveSFT：pip install -e <adaptiveSFT 路徑> 或設 ADAPTIVESFT_PATH")
    for k in [k for k in sys.modules if k == "adaptivesft" or k.startswith("adaptivesft.")]:
        del sys.modules[k]
    sys.path.insert(0, root)
    return importlib.import_module("adaptivesft.experiment")


Psi, salience_levels = _import_psi()

__all__ = ["ColourCalibrator", "AudioCalibrator", "load_confusion", "write_blocks", "read_calibration",
           "lab_to_hex", "delta_e", "random_lab", "find_colour_at", "CSV_COLUMNS", "vawm_to_dfp_rows", "analyze_vawm"]

# stimuli/practice.csv 的欄位順序（VAWM_nobox.py 用名字取欄，順序照舊最保險）
CSV_COLUMNS = ["trial",
               "color1_target", "color1_H", "color1_L", "color1_H_deltaE", "color1_L_deltaE", "color1_HL_deltaE",
               "color2_target", "color2_H", "color2_L", "color2_H_deltaE", "color2_L_deltaE", "color2_HL_deltaE",
               "audio1_target", "audio1_H", "audio1_L", "audio1_H_count", "audio1_L_count",
               "audio1_target_talker", "audio1_H_talker", "audio1_L_talker",
               "audio1_target_file", "audio1_H_file", "audio1_L_file",
               "audio2_target", "audio2_H", "audio2_L", "audio2_H_count", "audio2_L_count",
               "audio2_target_talker", "audio2_H_talker", "audio2_L_talker",
               "audio2_target_file", "audio2_H_file", "audio2_L_file"]
TALKERS = [f"T{i:02d}" for i in range(1, 13)]


# ============================================================================================
# 顏色：與 colorpool.py 相同的定義（CIELAB 歐氏距離、D65 sRGB、色域檢查），不 import 它（它一 import 就跑 155 試的產生迴圈）
# ============================================================================================

def _lab_to_rgb(lab):
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0
    eps, kap = 216.0 / 24389.0, 24389.0 / 27.0

    def finv(t):
        return t ** 3 if t ** 3 > eps else (116.0 * t - 16.0) / kap
    xn, yn, zn = 0.95047, 1.0, 1.08883
    X = xn * finv(fx)
    Y = yn * ((L + 16.0) / 116.0) ** 3 if L > kap * eps else yn * L / kap
    Z = zn * finv(fz)
    r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    bb = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    def gamma(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    return np.array([gamma(r), gamma(g), gamma(bb)])


def is_in_gamut(lab):
    rgb = _lab_to_rgb(lab)
    return bool(np.all(rgb >= 0.0) and np.all(rgb <= 1.0))


def lab_to_hex(lab):
    rgb = np.clip(_lab_to_rgb(lab), 0, 1)
    r, g, b = [int(x * 255) for x in rgb]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def delta_e(c1, c2):
    """colorpool.py:7-8：CIELAB 歐氏距離（ΔE76）。"""
    return float(np.sqrt(np.sum((np.asarray(c1, float) - np.asarray(c2, float)) ** 2)))


def random_lab(rng):
    """colorpool.py:18-22：L 40–60、C 40–60、任意色相。"""
    L = rng.uniform(40, 60)
    C = rng.uniform(40, 60)
    h = rng.uniform(0, 2 * np.pi)
    return np.array([L, C * np.cos(h), C * np.sin(h)])


def random_target(rng, avoid=(), min_sep=20.0, max_tries=500):
    for _ in range(max_tries):
        c = random_lab(rng)
        if is_in_gamut(c) and all(delta_e(c, o) >= min_sep for o in avoid):
            return c
    raise RuntimeError("找不到色域內的目標色")


def find_colour_at(target, de, rng, tol=0.5, max_tries=2000):
    """目標色往隨機方向走 de（±tol）的色域內顏色；colorpool.py find_color 的固定距離版。"""
    for _ in range(max_tries):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        cand = np.asarray(target, float) + direction * de
        if is_in_gamut(cand) and abs(delta_e(target, cand) - de) <= tol:
            return cand
    return None


# ============================================================================================
# 聲音：混淆表
# ============================================================================================

def load_confusion(path=None):
    """從 generate_trials.py 讀 CONFUSION_DATA（不 import 它，它 import pandas）。回傳 {sound: [(target, count), …]}。"""
    path = path or os.path.join(_HERE, "generate_trials.py")
    src = open(path, encoding="utf-8").read()
    m = re.search(r"CONFUSION_DATA\s*=\s*(\[.*?\n\])", src, re.S)
    table = {}
    for row in ast.literal_eval(m.group(1)):
        table.setdefault(row["sound"], []).append((row["target"], int(row["count"])))
    return table


def audio_x(count):
    """混淆次數 → 不相似度軸。count 1 → 0 越接近 −0.3；count 451 → −2.66。"""
    return -np.log10(float(count) + 1.0)


def count_from_x(x):
    return 10.0 ** (-float(x)) - 1.0


def sound_file(talker, cons):
    return f"stimuli/{talker}_{cons}3.wav"


# ============================================================================================
# 校準控制器
# ============================================================================================

@dataclass
class CalibResult:
    dim: str
    n_trials: int
    n_match: int
    false_alarm: float
    alpha: float
    beta: float
    high: float
    low: float
    p_high: float
    p_low: float
    x_range: tuple
    in_range: bool
    warnings: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class _WMCalibrator:
    """兩個維度共用的骨架：next_trial() 給一試的刺激；record(key, rt) 更新 Psi；finish() 給 H/L。"""
    dim = ""

    def __init__(self, psi, p_high, p_low, p_match, floor_guess, seed):
        self.psi = psi
        self.p_high, self.p_low, self.p_match = float(p_high), float(p_low), float(p_match)
        self.floor_guess = float(floor_guess)
        self.rng = np.random.default_rng(seed)
        self.log = []
        self._pending = None

    # 子類別實作
    def _make_trial(self, x, is_match):
        raise NotImplementedError

    def next_trial(self):
        is_match = self.rng.uniform() < self.p_match
        x = float(self.psi.next_intensity)
        trial = self._make_trial(x, is_match)
        trial.update(trial_no=len(self.log) + 1, is_match=is_match, x_psi=x)
        self._pending = trial
        return trial

    def record(self, key, rt):
        """key：'y' = 相同、'n' = 不同、None = 沒作答。回傳 (r, correct)。"""
        t = self._pending
        if t is None:
            raise RuntimeError("先呼叫 next_trial()")
        if key not in ("y", "n", None):
            raise ValueError("key 必須是 'y' / 'n' / None")
        r = None if key is None else int(key == "n")
        correct = None if r is None else int(r == (0 if t["is_match"] else 1))
        if r is not None and not t["is_match"]:                       # AA 試不進 Psi（x = 0 不在網格上）
            self.psi.update_at(t["x_index"], r)
        self.log.append(dict(**t, key=key, rt=rt, r=r, correct=correct))
        self._pending = None
        return r, correct

    def false_alarm_rate(self):
        aa = [t["r"] for t in self.log if t["is_match"] and t["r"] is not None]
        return float(np.mean(aa)) if len(aa) >= 5 else self.floor_guess

    def finish(self):
        a_hat, b_hat = self.psi.estimate()
        fa = self.false_alarm_rate()
        x_range = (float(self.psi.x.min()), float(self.psi.x.max()))
        (high, low), warns = salience_levels(a_hat, b_hat, self.psi.d, [self.p_high, self.p_low],
                                             x_range=x_range, lower=fa, upper=self.psi.upper)
        n_match = sum(t["is_match"] for t in self.log)
        return CalibResult(self.dim, len(self.log), n_match, fa, a_hat, b_hat, high, low, self.p_high, self.p_low,
                           x_range, not warns, warns, self._extra(high, low))

    def _extra(self, high, low):
        return {}


class ColourCalibrator(_WMCalibrator):
    """
    區塊 1：顏色差距。x = 探測色與目標色的 Lab 距離（2–50，步 1；現有 csv 的 L 在 20–30、H 在 25–50）。
    每試：兩個目標色（相距 ≥ 20）、兩個目標音（不同子音、各自隨機 talker）；探測 = 其中一個目標的顏色 ± x 與它自己的聲音。
    """
    dim = "colour"

    def __init__(self, p_high=0.90, p_low=0.75, p_match=1 / 3, floor_guess=0.10, lapse=0.02, seed=None,
                 x=(2.0, 50.0, 1.0), a=(0.0, 45.0, 1.0), b=(1.0, 25.0, 1.0), confusion=None):
        psi = Psi(x, a, b, d=lapse, lower=floor_guess, upper=lapse / 2)
        super().__init__(psi, p_high, p_low, p_match, floor_guess, seed)
        self.confusion = confusion or load_confusion()
        self.sounds = sorted(self.confusion)

    def _make_trial(self, x, is_match):
        t1 = random_target(self.rng)
        t2 = random_target(self.rng, avoid=[t1])
        which = int(self.rng.integers(2))
        target = (t1, t2)[which]
        s1, s2 = self.rng.choice(self.sounds, 2, replace=False)
        k1, k2 = self.rng.choice(TALKERS, 2)
        if is_match:
            probe = target
        else:
            probe = find_colour_at(target, x, self.rng)
            if probe is None:                                          # 這個目標往哪走都出色域：換目標
                target = random_target(self.rng)
                probe = find_colour_at(target, x, self.rng) or target
        de = delta_e(target, probe)
        return dict(color1_target=lab_to_hex(t1), color2_target=lab_to_hex(t2),
                    audio1_target_file=sound_file(k1, s1), audio2_target_file=sound_file(k2, s2),
                    probe_colour=lab_to_hex(probe), probe_sound=sound_file((k1, k2)[which], (s1, s2)[which]),
                    probe_of=which + 1, x=de, x_index=self.psi.nearest_index(de))


class AudioCalibrator(_WMCalibrator):
    """
    區塊 2：子音混淆度。x = −log10(count + 1)，Psi 提的 x 貼到目標子音可用 foil 裡最近的一個（update_at）。
    每試：兩個目標色、兩個目標音；探測 = 其中一個目標的顏色與它的 foil 子音（同 talker）。
    """
    dim = "audio"

    def __init__(self, p_high=0.90, p_low=0.75, p_match=1 / 3, floor_guess=0.10, lapse=0.02, seed=None,
                 x=(-2.7, 0.0, 0.05), a=(-2.7, 0.0, 0.05), b=(0.05, 2.0, 0.05), confusion=None):
        psi = Psi(x, a, b, d=lapse, lower=floor_guess, upper=lapse / 2)
        super().__init__(psi, p_high, p_low, p_match, floor_guess, seed)
        self.confusion = confusion or load_confusion()
        self.sounds = sorted(self.confusion)

    def foil_for(self, cons, x):
        """目標子音 cons 的 foil 裡，不相似度最接近 x 的那個。回傳 (foil, count)。"""
        cands = self.confusion[cons]
        j = int(np.argmin([abs(audio_x(c) - x) for _, c in cands]))
        return cands[j]

    def _make_trial(self, x, is_match):
        t1 = random_target(self.rng)
        t2 = random_target(self.rng, avoid=[t1])
        s1, s2 = self.rng.choice(self.sounds, 2, replace=False)
        k1, k2 = self.rng.choice(TALKERS, 2)
        which = int(self.rng.integers(2))
        target_s, talker = (s1, s2)[which], (k1, k2)[which]
        if is_match:
            foil, count, x_act = target_s, 0, 0.0
        else:
            foil, count = self.foil_for(target_s, x)
            x_act = audio_x(count)
        return dict(color1_target=lab_to_hex(t1), color2_target=lab_to_hex(t2),
                    audio1_target_file=sound_file(k1, s1), audio2_target_file=sound_file(k2, s2),
                    probe_colour=lab_to_hex((t1, t2)[which]), probe_sound=sound_file(talker, foil),
                    probe_of=which + 1, foil=foil, count=count, x=x_act, x_index=self.psi.nearest_index(x_act))

    def _extra(self, high, low):
        return dict(count_high=count_from_x(high), count_low=count_from_x(low))


# ============================================================================================
# 存 / 讀校準結果，產生 block csv
# ============================================================================================

def save_calibration(path, colour: CalibResult, audio: CalibResult, info=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(info=info or {}, colour=asdict(colour), audio=asdict(audio)), f, ensure_ascii=False, indent=2)


def read_calibration(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_blocks(calib, out_dir, n_blocks=6, trials_per_block=24, seed=None, min_target_sep=20.0,
                 min_hl_sep=0.0):
    """
    用校準結果產生 stimuli/block1..6.csv（欄位 = CSV_COLUMNS）。
    顏色：H / L foil 在 ΔE_H / ΔE_L（±0.5）處；聲音：H / L foil = 目標子音的 foil 裡 count 最接近 count_H / count_L 者。
    """
    rng = np.random.default_rng(seed)
    conf = load_confusion()
    sounds = sorted(conf)
    de_h, de_l = float(calib["colour"]["high"]), float(calib["colour"]["low"])
    c_h, c_l = float(calib["audio"]["extra"]["count_high"]), float(calib["audio"]["extra"]["count_low"])

    def foil_near(cons, count):
        cands = conf[cons]
        j = int(np.argmin([abs(np.log10(c + 1) - np.log10(count + 1)) for _, c in cands]))
        return cands[j]

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    prev = []
    for b in range(1, n_blocks + 1):
        rows = []
        for t in range(1, trials_per_block + 1):
            for _ in range(200):
                t1 = random_target(rng, avoid=prev, min_sep=30.0)
                t2 = random_target(rng, avoid=[t1] + prev, min_sep=min_target_sep)
                h1, l1 = find_colour_at(t1, de_h, rng), find_colour_at(t1, de_l, rng)
                h2, l2 = find_colour_at(t2, de_h, rng), find_colour_at(t2, de_l, rng)
                if any(v is None for v in (h1, l1, h2, l2)):
                    continue
                if delta_e(h1, l1) < min_hl_sep or delta_e(h2, l2) < min_hl_sep:
                    continue
                break
            else:
                raise RuntimeError("色域內找不到符合 ΔE_H / ΔE_L 的顏色；校到的 ΔE 太大或太小")
            prev = [t1, t2]
            s1, s2 = rng.choice(sounds, 2, replace=False)
            k1, k2 = rng.choice(TALKERS, 2)
            (h1s, h1c), (l1s, l1c) = foil_near(s1, c_h), foil_near(s1, c_l)
            (h2s, h2c), (l2s, l2c) = foil_near(s2, c_h), foil_near(s2, c_l)
            rows.append(dict(
                trial=t,
                color1_target=lab_to_hex(t1), color1_H=lab_to_hex(h1), color1_L=lab_to_hex(l1),
                color1_H_deltaE=round(delta_e(t1, h1), 2), color1_L_deltaE=round(delta_e(t1, l1), 2),
                color1_HL_deltaE=round(delta_e(h1, l1), 2),
                color2_target=lab_to_hex(t2), color2_H=lab_to_hex(h2), color2_L=lab_to_hex(l2),
                color2_H_deltaE=round(delta_e(t2, h2), 2), color2_L_deltaE=round(delta_e(t2, l2), 2),
                color2_HL_deltaE=round(delta_e(h2, l2), 2),
                audio1_target=s1, audio1_H=h1s, audio1_L=l1s, audio1_H_count=h1c, audio1_L_count=l1c,
                audio1_target_talker=k1, audio1_H_talker=k1, audio1_L_talker=k1,
                audio1_target_file=sound_file(k1, s1), audio1_H_file=sound_file(k1, h1s), audio1_L_file=sound_file(k1, l1s),
                audio2_target=s2, audio2_H=h2s, audio2_L=l2s, audio2_H_count=h2c, audio2_L_count=l2c,
                audio2_target_talker=k2, audio2_H_talker=k2, audio2_L_talker=k2,
                audio2_target_file=sound_file(k2, s2), audio2_H_file=sound_file(k2, h2s), audio2_L_file=sound_file(k2, l2s),
            ))
        path = os.path.join(out_dir, f"block{b}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        paths.append(path)
    return paths


# ============================================================================================
# VAWM_nobox.py 的輸出 → DFP 分析
# ============================================================================================

# VAWM_nobox.py 的 condition 編碼（probe 的 Begin Routine 註解）：
#   AA=0,1  HA=2,3  LA=4,5  AH=6,7  HH=8,9  LH=10,11  AL=12,13  HL=14,15  LL=16,17
# 第一個字母 = 顏色、第二個 = 聲音；A = 與目標相同。DFP 的 2×2 只用 HH / HL / LH / LL。
_COND_CELL = {8: (2, 2), 9: (2, 2), 14: (2, 1), 15: (2, 1), 10: (1, 2), 11: (1, 2), 16: (1, 1), 17: (1, 1)}
_COND_MATCH = {0, 1}


def vawm_to_dfp_rows(path, subject=None, rt_col="ResponseBox.rt", key_col="ResponseBox.keys", cond_col="condition"):
    """
    讀 VAWM_nobox.py 存的 csv（每個探測一列），轉成 adaptivesft.experiment.analyze_participant 要的列：
    channel1 = 顏色（2 = H、1 = L）、channel2 = 聲音、correct = 是否答對（AA 該答 y，其餘該答 n）、rt。
    只保留 HH / HL / LH / LL 的探測；AA / HA / LA / AH / AL 這些單通道與相同試次另外回傳（給正確率檢查）。
    """
    rows, others = [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                cond = int(float(r[cond_col]))
            except (KeyError, TypeError, ValueError):
                continue
            key = (r.get(key_col) or "").strip()
            try:
                rt = float(r[rt_col])
            except (KeyError, TypeError, ValueError):
                rt = None
            should_say_yes = cond in _COND_MATCH
            correct = None if key not in ("y", "n") else int((key == "y") == should_say_yes)
            rec = dict(subject=subject or r.get("participant", ""), condition="DFP", cond_code=cond, key=key,
                       correct=correct, rt=rt)
            if cond in _COND_CELL:
                rec["channel1"], rec["channel2"] = _COND_CELL[cond]
                rec["block"] = r.get("block1.thisN", r.get("block", ""))
                rec["trial"] = len(rows) + 1
                rows.append(rec)
            else:
                others.append(rec)
    return rows, others


def analyze_vawm(path, subject=None, **kw):
    """VAWM_nobox.py 的 csv → 四格答對 RT → SIC / MIC / dominance → 預測架構（用 adaptivesft.experiment）。"""
    exp = _import_experiment()
    analyze_participant, report = exp.analyze_participant, exp.report
    rows, others = vawm_to_dfp_rows(path, subject)
    for r in rows:
        r["correct"] = "" if r["correct"] is None else r["correct"]
        r["rt"] = "" if r["rt"] is None else r["rt"]
    res = analyze_participant(rows, **kw)
    acc_single = {}
    for r in others:
        if r["correct"] is not None:
            acc_single.setdefault(r["cond_code"], []).append(r["correct"])
    return dict(result=res, report=report(res), n_dfp=len(rows),
                accuracy_other={c: float(np.mean(v)) for c, v in sorted(acc_single.items())})
