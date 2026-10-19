#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT_WM_change.py —— SFT 實驗的「看得懂」版（change 版）：校準（原始 adaptiveSFT，lnrm）→ 練習 → 主實驗，一個檔案跑到尾。
【change 版】校準時另一個維度也每試重抽：校顏色時兩個聲音每試換、校聲音時兩個顏色每試換。
            姊妹版 SFT_WM_nochange.py：校顏色時固定一組聲音配對、校聲音時固定一組顏色配對。

跟正式版（VAWM_calibrate.py + make_adaptive_blocks.py + VAWM_nobox.py）做的事一樣、時間軸一樣、資料欄位相容，
只是寫法攤平：一個區塊一個區塊往下讀就是受試者會經歷的順序。數學（擬合、找顏色）照舊借用，其餘都寫在這裡。

    Coder 直接 Run。                          正式跑
    python SFT_WM_change.py --robot             機器人自己答，不開視窗（測流程）
    python SFT_WM_change.py --robot --quick     機器人 + 縮短擬合 + 一個 block（幾分鐘跑完）

產出（都在 data/）：
    <subj>_simple_calib_trials.csv   校準的每一試
    <subj>_simple_calib.json         校到的 H / L（make_adaptive_blocks.py 也讀得懂）
    <subj>_simple_main.csv           主實驗每個探測一列（adaptive_vawm.analyze_vawm 讀得懂）
"""
import csv
import json
import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):                 # Windows 主控台是 cp1252，印中文會炸
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.chdir(os.path.dirname(os.path.abspath(__file__)))   # 刺激檔都是相對路徑 stimuli/...

ROBOT = "--robot" in sys.argv       # True = 不開視窗，機器人作答
QUICK = "--quick" in sys.argv       # True = 擬合縮短、只跑 1 個 block

# ════════════════════════════════════════════════════════════════════════════════
# 0. 設定 —— 要改的數字都在這裡
# ════════════════════════════════════════════════════════════════════════════════
# 校準（原始 adaptiveSFT：固定幾個強度各跑幾試，收對錯 + 反應時間，區塊結束擬合 lnrm2）
LEVELS_COLOUR = [2.0, 6.0, 12.0, 20.0, 30.0, 45.0]   # 探測色與目標色的距離 ΔE（越大越好分）
LEVELS_AUDIO = [-2.6, -2.1, -1.6, -1.1, -0.7, -0.3]  # 子音不相似度 = −log10(混淆次數+1)（越大越好分）
N_PER_LEVEL = 8                                      # 每個強度幾試 → 6×8 = 48 試「不同」
P_MATCH = 1 / 3                                      # 再加 1/3 的「相同」試（探測 = 目標）→ 共 72 試
H_TARG, L_TARG = 2.0, 0.5                            # 要找的兩個難度：漂移差 2.0（容易 H）/ 0.5（困難 L）
FIT_KW = dict(tune=1000, draws=1000, chains=4, nuts_sampler="nutpie")   # 擬合設定；nutpie 約 70 s
RT_MIN, RT_MAX = 0.15, 3.0                           # 擬合只用這範圍內的反應時間
# 主實驗
N_BLOCKS = 6                 # 幾個 block
SETS_PER_BLOCK = 24          # 每個 block 幾組「學習 2 色 2 音」
PROBES_PER_SET = 9           # 每組學習後接幾個探測（固定：2 AA、1 HA、1 LA、1 AH、1 AL、3 個 HH/HL/LH/LL）
N_PRACTICE_SETS = 2          # 練習幾組（有回饋）
RESP_WINDOW = 3.0            # 探測後等作答幾秒
if QUICK:
    FIT_KW = dict(tune=300, draws=300, chains=2, nuts_sampler="nutpie")
    N_BLOCKS, SETS_PER_BLOCK, N_PRACTICE_SETS = 1, 3, 1

# ════════════════════════════════════════════════════════════════════════════════
# 1. 借來的工具（數學在這些函式裡；本檔只負責流程）
# ════════════════════════════════════════════════════════════════════════════════
from adaptive_vawm import (TALKERS, audio_x, count_from_x, delta_e, find_colour_at, foil_for,  # noqa: E402
                           from_model_units, lab_to_hex, load_confusion, load_full_package, random_target,
                           sound_file, to_model_units)
#   random_target(rng)            抽一個色域內的目標色（Lab）
#   find_colour_at(t, dE, rng)    找一個跟 t 距離 dE 的顏色（找不到回 None）
#   lab_to_hex / delta_e          Lab → '#RRGGBB' / 兩色距離
#   load_confusion()              {子音: [(容易搞混的子音, 混淆次數), ...]}
#   foil_for(conf, 子音, x)        依不相似度 x 挑一個干擾子音 → (foil, count)
#   audio_x(count) / count_from_x 混淆次數 ↔ 不相似度 x
#   sound_file(talker, 子音)       → 'stimuli/T01_b3.wav'
#   to/from_model_units           進出擬合前的尺度換算（顏色 ΔE/10；聲音 x+2.7）
CONFUSION = load_confusion()
CONSONANTS = sorted(CONFUSION)
rng = np.random.default_rng()

# ════════════════════════════════════════════════════════════════════════════════
# 2. 畫面與一試的流程（跟 VAWM_nobox.py 的時間軸一樣）
#    注視 .3 → 左色 1 s → 右色 1 s → 音 1（1 s）→ 音 2（1 s）→ 注視 .3 → 探測色+音 1 s → 等按 y / n
# ════════════════════════════════════════════════════════════════════════════════
if not ROBOT:
    from psychopy import core, event, gui, sound, visual
    from psychopy.hardware import keyboard

    # 偏好設定裡記的喇叭（例如 DELL 螢幕）不在時 sound.Sound() 會直接炸，先逐一試開、留下第一個能用的
    AUDIO_DEVICE = ""                                   # 指定輸出裝置名稱的一部分（例 "Headphones"）；空字串 = 自動挑
    AUDIO_PREFER = ["headphone", "耳機", "earphone"]     # 自動挑時優先的名字（小寫比對）

    def _pick_audio_device():
        """
        PsychoPy 的偏好設定會記住上次用的喇叭名字（userPrefs.cfg 的 hardware/audioDevice）。
        那台裝置拔掉或換機器之後，sound.Sound() 會丟 DeviceNotConnectedError；
        而且「列得出來」不等於「開得起來」（例如耳機孔沒插東西，PTB 會開串流失敗）。
        所以這裡逐一實際試開，留下第一個成功的；偏好設定裡那個仍可用就不動它。
        """
        from psychopy import prefs
        try:
            from psychopy.hardware.speaker import SpeakerDevice
        except Exception as e:                      # 舊版 PsychoPy 沒有 SpeakerDevice：維持原設定
            print(f"[audio] 無法列出裝置（{e}），沿用偏好設定")
            return
        avail = []
        for d in SpeakerDevice.getAvailableDevices():
            name = d.get("deviceName") if isinstance(d, dict) else getattr(d, "deviceName", None)
            if name and name not in avail:
                avail.append(name)
        if not avail:
            print("[audio] 找不到任何輸出裝置")
            return
        want = prefs.hardware.get("audioDevice") or []
        if isinstance(want, str):
            want = [want]
        # 排序：AUDIO_DEVICE 指名的 > 名字像耳機的 > 偏好設定裡的 > 其他。受試者要戴耳機，不能落到喇叭。
        named = [a for a in avail if AUDIO_DEVICE and AUDIO_DEVICE.lower() in a.lower()]
        headph = [a for a in avail if any(k in a.lower() for k in AUDIO_PREFER) and a not in named]
        rest = [a for a in avail if a not in named + headph]
        order = named + headph + [w for w in want if w in rest] + [a for a in rest if a not in want]
        for name in order:
            try:
                SpeakerDevice(name=name)      # 第一個位置參數是 index，一定要用關鍵字
            except Exception as e:
                print(f"[audio] {name} 開不起來（{type(e).__name__}），試下一個")
                continue
            if list(want) != [name]:
                prefs.hardware["audioDevice"] = [name]
                print(f"[audio] 改用：{name}")
            else:
                print(f"[audio] 使用偏好設定的裝置：{name}")
            if not any(k in name.lower() for k in AUDIO_PREFER):
                print(f"[audio] ⚠ 這不是耳機（{name}）。請插上耳機再跑，或在檔頭 AUDIO_DEVICE 指定裝置名稱")
            return
        print(f"[audio] 這些裝置都開不起來：{avail}；請插上喇叭或耳機再跑")
    _pick_audio_device()

    exp = {"participant": "", "session": "001"}
    if not gui.DlgFromDict(exp, title="VAWM simple").OK:
        core.quit()
    SUBJ = exp["participant"]
    win = visual.Window(fullscr=True, color="black", units="pix")
    fix = visual.TextStim(win, text="+", height=40, color="white")
    msg = visual.TextStim(win, text="", height=28, color="white", wrapWidth=1000,
                          alignText="left", anchorHoriz="center")
    patch_l = visual.Rect(win, width=100, height=100, pos=(-200, 0), colorSpace="hex", lineColor="#808080")
    patch_r = visual.Rect(win, width=100, height=100, pos=(200, 0), colorSpace="hex", lineColor="#808080")
    patch_c = visual.Rect(win, width=100, height=100, pos=(0, 0), colorSpace="hex", lineColor="#808080")
    snd = sound.Sound("A", secs=1, stereo=True, hamming=True)
    clock = core.Clock()
    kb = keyboard.Keyboard(clock=clock)   # RT 以這個 clock 為準；每題探測前 clock.reset()

    def show(text):                       # 顯示一段文字，等空白鍵；esc 離開
        msg.text = text
        msg.draw()
        win.flip()
        if "escape" in event.waitKeys(keyList=["space", "escape"]):
            core.quit()

    def present(draw, secs, play=None):   # 畫 draw() 幾秒；play 給 wav 路徑就同時播
        if play:
            snd.setSound(play, secs=1, hamming=True)
            snd.play()
        t0 = core.getTime()
        while core.getTime() - t0 < secs:
            if draw:
                draw()
            win.flip()
            if kb.getKeys(["escape"]):
                core.quit()
        if play:
            snd.stop()

    def one_trial(c1, c2, s1, s2, probe_col, probe_snd):
        """學習 2 色 2 音，然後探測一個 色+音，回傳 (按鍵 'y'/'n'/None, 反應時間)。"""
        present(fix.draw, 0.3)
        patch_l.fillColor = c1
        present(patch_l.draw, 1.0)
        patch_r.fillColor = c2
        present(patch_r.draw, 1.0)
        present(None, 1.0, play=s1)
        present(None, 1.0, play=s2)
        present(fix.draw, 0.3)
        patch_c.fillColor = probe_col
        kb.clearEvents()
        clock.reset()
        present(patch_c.draw, 1.0, play=probe_snd)
        keys = kb.getKeys(["y", "n"], waitRelease=False)
        while not keys and clock.getTime() < RESP_WINDOW:
            win.flip()
            keys = kb.getKeys(["y", "n"], waitRelease=False)
            if kb.getKeys(["escape"]):
                core.quit()
        return (keys[0].name, keys[0].rt) if keys else (None, None)

    def feedback(text):
        msg.text = text
        present(msg.draw, 0.8)

else:
    # 機器人：不開視窗。答「不同」的機率隨差異變大而上升；AA 試以 10% 誤按「不同」。
    SUBJ = "robot"
    from scipy.stats import norm

    def show(text):
        print("[畫面]", text.split("\n")[0])

    def feedback(text):
        pass

    load_full_package()
    from adaptivesft.race import lnrm_random     # 跟 tests/test_adaptive_vawm.py 的 lnrm_observer 一樣的模擬受試者

    def one_trial(c1, c2, s1, s2, probe_col, probe_snd, _info=None):
        dim, x, is_match = _info
        if is_match:                                            # AA 試：10% 誤按「不同」
            return ("n" if rng.uniform() < 0.10 else "y"), 0.8
        u = to_model_units(dim, x)                              # 難度 d(u) = 1.0·u − 0.12·u²（跟測試同一組真值）
        rt, correct = lnrm_random([1.0 * u - 0.12 * u * u], 1.0, 0.5, 0.2, rng)
        return ("n" if correct[0] == 1 else "y"), float(rt[0])

os.makedirs("data", exist_ok=True)


def write_csv(path, rows):               # 每試寫一次整個檔，中途當掉也留得住
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def two_targets():
    """抽兩個目標色（相距 ≥ 20 ΔE）、兩個不同子音、各自一個語者。"""
    t1 = random_target(rng)
    t2 = random_target(rng, avoid=[t1])
    s1, s2 = rng.choice(CONSONANTS, 2, replace=False)
    k1, k2 = rng.choice(TALKERS, 2)
    return t1, t2, str(s1), str(s2), str(k1), str(k2)

# ════════════════════════════════════════════════════════════════════════════════
# 3. 校準：一個維度跑 72 試 → 擬合 → 得到 H（容易）與 L（困難）。不給回饋（回饋只在練習）
# ════════════════════════════════════════════════════════════════════════════════
def calibrate(dim, log_rows):
    levels = LEVELS_COLOUR if dim == "colour" else LEVELS_AUDIO
    # 3a. 排試次：每個強度 N_PER_LEVEL 試，再加 AA 試湊到 1/3，打散
    plan = [x for x in levels for _ in range(N_PER_LEVEL)]
    plan += [None] * int(round(len(plan) * P_MATCH / (1 - P_MATCH)))
    rng.shuffle(plan)

    show(("Part 1 of 3: colour calibration (72 trials)" if dim == "colour"
          else "Part 2 of 3: sound calibration (72 trials)") +
         "\n\nOn each trial you will first see two coloured squares and hear two speech sounds. "
         "Try to remember them. Then one colour will appear together with one sound.\n\n"
         "If the colour AND the sound are both the same as what you just saw and heard, press Y.\n"
         "If anything is different, press N.\n\n"
         "There is no feedback in this part.\n\nPress the space bar to begin.")
    for i, x in enumerate(plan, 1):
        t1, t2, s1, s2, k1, k2 = two_targets()
        which = int(rng.integers(2))                 # 探測第 1 還是第 2 個目標
        target_col = [t1, t2][which]
        target_snd, talker = [s1, s2][which], [k1, k2][which]
        is_match = x is None
        if dim == "colour":                          # 只動顏色：探測音 = 目標音
            if is_match:
                probe_lab = target_col
            else:
                probe_lab = find_colour_at(target_col, x, rng)
                if probe_lab is None:                 # 這個目標往哪走都出色域：退回目標色（極少發生）
                    probe_lab = target_col
            x_actual = delta_e(target_col, probe_lab)
            probe_col, probe_snd = lab_to_hex(probe_lab), sound_file(talker, target_snd)
        else:                                        # 只動聲音：探測色 = 目標色
            if is_match:
                foil, count = target_snd, 0
            else:
                foil, count = foil_for(CONFUSION, target_snd, x)
            x_actual = audio_x(count)
            probe_col, probe_snd = lab_to_hex(target_col), sound_file(talker, foil)

        args = (lab_to_hex(t1), lab_to_hex(t2), sound_file(k1, s1), sound_file(k2, s2), probe_col, probe_snd)
        key, rt = one_trial(*args, _info=(dim, x_actual, is_match)) if ROBOT else one_trial(*args)
        correct = None if key is None else int(key == ("y" if is_match else "n"))
        log_rows.append(dict(dim=dim, trial=i, is_match=int(is_match), x_planned=x, x=round(x_actual, 3),
                             probe_col=probe_col, probe_snd=probe_snd, key=key, rt=rt, correct=correct))
        write_csv(f"data/{SUBJ}_simple_calib_trials.csv", log_rows)

    # 3b. 擬合 lnrm2（只用有作答、RT 在範圍內的「不同」試）
    if not ROBOT:
        msg.text = "Calculating... this takes about a minute. Please wait."
        msg.draw(); win.flip()
    else:
        print("  fitting...")
    load_full_package()
    from adaptivesft.models import fit_lnrm, make_data
    from adaptivesft.salience import find_salience
    use = [r for r in log_rows if r["dim"] == dim and not r["is_match"] and r["key"] is not None
           and RT_MIN <= r["rt"] <= RT_MAX]
    data = make_data([to_model_units(dim, r["x"]) for r in use], [r["rt"] for r in use], [r["correct"] for r in use])
    t0 = time.time()
    trace = fit_lnrm(data, link="quadratic", **FIT_KW)
    res = find_salience(trace, h_targ=H_TARG, l_targ=L_TARG, alpha2_rule="filter")
    high = from_model_units(dim, res["high"]["intensity"])     # 回到 ΔE 或 x
    low = from_model_units(dim, res["low"]["intensity"])
    lo_lim, hi_lim = min(levels), max(levels)                  # 校準時呈現過的範圍
    if not (np.isfinite(high) and np.isfinite(low)):           # 後驗裡沒有任何 draw 解得出來
        print(f"  ⚠ [{dim}] 擬合解不出 H/L（{res['warnings']}），暫用中間層級頂著；這位受試者的校準要重做")
        high, low = levels[-2], levels[1]
    in_range = bool(lo_lim <= low < high <= hi_lim)
    if not in_range:                                           # 超出範圍 = 模型外插，不能信；夾回去並明講
        print(f"  ⚠ [{dim}] 擬合給的 H={high:.2f} L={low:.2f} 超出可呈現範圍 [{lo_lim}, {hi_lim}]，已夾回；"
              "這位受試者的校準要重做或檢查資料")
        high = float(np.clip(high, lo_lim, hi_lim))
        low = float(np.clip(low, lo_lim, hi_lim))
        if low >= high:
            low = lo_lim
    aa = [r["key"] == "n" for r in log_rows if r["dim"] == dim and r["is_match"] and r["key"] is not None]
    out = dict(dim=dim, high=float(high), low=float(low), in_range=in_range,
               false_alarm=float(np.mean(aa)) if aa else None,
               n_used=len(use), fit_seconds=round(time.time() - t0), warnings=list(res["warnings"]),
               extra=dict(method="lnrm_quadratic"))
    if dim == "audio":                                           # 聲音再換回「混淆次數」給題本用
        out["extra"]["count_high"] = float(count_from_x(high))
        out["extra"]["count_low"] = float(count_from_x(low))
    print(f"[{dim}] H={out['high']:.2f} L={out['low']:.2f} in_range={in_range} FA={out['false_alarm']} "
          f"擬合 {out['fit_seconds']}s {out['warnings']}")
    if not ROBOT and not in_range:
        show(f"Note for the experimenter: the {dim} calibration fell outside the tested range "
             "and was clipped. Please write this down.\n\nPress the space bar to continue.")
    return out

# ════════════════════════════════════════════════════════════════════════════════
# 4. 用 H / L 產生主實驗的一組題：2 個目標色各有 H、L 版本；2 個目標音各有 H、L 干擾子音
# ════════════════════════════════════════════════════════════════════════════════
def make_study_set(calib):
    de_h, de_l = calib["colour"]["high"], calib["colour"]["low"]
    cnt_h, cnt_l = calib["audio"]["extra"]["count_high"], calib["audio"]["extra"]["count_low"]
    for _ in range(200):
        t1, t2, s1, s2, k1, k2 = two_targets()
        cols = [find_colour_at(t, de, rng) for t in (t1, t2) for de in (de_h, de_l)]
        if all(c is not None for c in cols):
            break
    else:
        raise RuntimeError("色域內找不到符合 ΔE_H / ΔE_L 的顏色")
    h1, l1, h2, l2 = cols

    def foil_near(cons, count):           # 目標子音的干擾子音裡，混淆次數最接近 count 的
        cands = CONFUSION[cons]
        return cands[int(np.argmin([abs(np.log10(c + 1) - np.log10(count + 1)) for _, c in cands]))][0]

    return dict(
        color1_target=lab_to_hex(t1), color1_H=lab_to_hex(h1), color1_L=lab_to_hex(l1),
        color2_target=lab_to_hex(t2), color2_H=lab_to_hex(h2), color2_L=lab_to_hex(l2),
        audio1_target_file=sound_file(k1, s1), audio1_H_file=sound_file(k1, foil_near(s1, cnt_h)),
        audio1_L_file=sound_file(k1, foil_near(s1, cnt_l)),
        audio2_target_file=sound_file(k2, s2), audio2_H_file=sound_file(k2, foil_near(s2, cnt_h)),
        audio2_L_file=sound_file(k2, foil_near(s2, cnt_l)),
    )

# ════════════════════════════════════════════════════════════════════════════════
# 5. 主實驗：每組學習後 9 個探測。條件編碼跟 VAWM_nobox.py 一樣（分析程式認這些數字）
#    名字：第一個字母 = 顏色、第二個 = 聲音；A = 跟目標相同、H = 容易分、L = 難分
# ════════════════════════════════════════════════════════════════════════════════
COND = {   # 編號: (名字, 探測色欄位, 探測音欄位)   偶數 = 探測第 1 個目標，奇數 = 第 2 個
    0: ("AA", "color1_target", "audio1_target_file"), 1: ("AA", "color2_target", "audio2_target_file"),
    2: ("HA", "color1_target", "audio1_H_file"),      3: ("HA", "color2_target", "audio2_H_file"),
    4: ("LA", "color1_target", "audio1_L_file"),      5: ("LA", "color2_target", "audio2_L_file"),
    6: ("AH", "color1_H", "audio1_target_file"),      7: ("AH", "color2_H", "audio2_target_file"),
    8: ("HH", "color1_H", "audio1_H_file"),           9: ("HH", "color2_H", "audio2_H_file"),
    10: ("LH", "color1_L", "audio1_H_file"),          11: ("LH", "color2_L", "audio2_H_file"),
    12: ("AL", "color1_L", "audio1_target_file"),     13: ("AL", "color2_L", "audio2_target_file"),
    14: ("HL", "color1_H", "audio1_L_file"),          15: ("HL", "color2_H", "audio2_L_file"),
    16: ("LL", "color1_L", "audio1_L_file"),          17: ("LL", "color2_L", "audio2_L_file"),
}


def nine_probes():
    """一組的 9 個探測：2 AA、1 HA、1 LA、1 AH、1 AL、3 個雙通道（HH/LH/HL/LL 隨機），打散。"""
    c = [int(rng.choice([0, 1])) for _ in range(2)]
    c += [int(rng.choice([2, 3])), int(rng.choice([4, 5])), int(rng.choice([6, 7])), int(rng.choice([12, 13]))]
    c += [int(rng.choice([8, 9, 10, 11, 14, 15, 16, 17])) for _ in range(3)]
    rng.shuffle(c)
    return c


def run_sets(n_sets, block, calib, rows, with_feedback):
    for s in range(1, n_sets + 1):
        st = make_study_set(calib)
        for p, cond in enumerate(nine_probes(), 1):
            name, col_key, snd_key = COND[cond]
            probe_col, probe_snd = st[col_key], st[snd_key]
            args = (st["color1_target"], st["color2_target"], st["audio1_target_file"], st["audio2_target_file"],
                    probe_col, probe_snd)
            key, rt = one_trial(*args, _info=("colour", 20.0 if name != "AA" else 0.0, name == "AA")) if ROBOT else one_trial(*args)
            correct = None if key is None else int(key == ("y" if name == "AA" else "n"))
            if with_feedback:
                feedback("No response" if key is None else ("Correct" if correct else "Wrong"))
            row = dict(participant=SUBJ, block=block, set=s, probe=p, condition=cond, cond_name=name,
                       targetCol=probe_col, targetAud=probe_snd, correct=correct)
            row["ResponseBox.keys"] = key          # 欄名照 VAWM_nobox.py，analyze_vawm 認這兩欄
            row["ResponseBox.rt"] = rt
            row.update(st)
            rows.append(row)
            write_csv(f"data/{SUBJ}_simple_main.csv", rows)

# ════════════════════════════════════════════════════════════════════════════════
# 6. 整個流程
# ════════════════════════════════════════════════════════════════════════════════
calib_rows = []
colour = calibrate("colour", calib_rows)
audio = calibrate("audio", calib_rows)
calib = dict(info=dict(participant=SUBJ, method="lnrm", h_targ=H_TARG, l_targ=L_TARG), colour=colour, audio=audio)
with open(f"data/{SUBJ}_simple_calib.json", "w", encoding="utf-8") as f:
    json.dump(calib, f, ensure_ascii=False, indent=2)

main_rows = []
show("Part 3 of 3: the main task. First, a short practice.\n\n"
     "Each round starts the same way: two colours and two sounds to remember. "
     "Then you will answer nine questions in a row, one colour-and-sound pair at a time.\n\n"
     "Both the same as before: press Y.   Anything different: press N.\n\n"
     "Feedback is shown during practice only.\n\nPress the space bar to begin.")
run_sets(N_PRACTICE_SETS, 0, calib, main_rows, with_feedback=True)
for b in range(1, N_BLOCKS + 1):
    show(f"Main task: block {b} of {N_BLOCKS}\n\nSame rules as the practice. There is no feedback from now on.\n\n"
         "Answer as accurately as you can; speed is not important.\n\nPress the space bar to begin.")
    run_sets(SETS_PER_BLOCK, b, calib, main_rows, with_feedback=False)
show("That is the end of the experiment. Thank you very much!\n\nPress the space bar to finish.")
print("done:", f"data/{SUBJ}_simple_calib.json", f"data/{SUBJ}_simple_main.csv")
if not ROBOT:
    win.close()
    core.quit()
