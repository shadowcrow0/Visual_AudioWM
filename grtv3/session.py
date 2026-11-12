"""把模組接成一次完整的 session：對話框 → 資料檔 → 介紹 → 校顏色 → 校聲音 → 練習 → 主實驗。

資料欄位名稱與 GRTv3_ada_EN.py 相同（phase, trial_i, colour_arc_lo/hi, snr_hi/lo, quad_content, time_perm, target_serial,
studyUR..BR, mask_*, snd_mask, cue_*, target_*, opt_perm, pressed_key, chosen_slot, chosen_item, outcome, err_rel(_name),
is_correct, rt, snd_item0..3, snd_probe, adapt_x/resp/corr, psi_*_alpha/beta, final_*），所以既有的分析腳本不用改。
"""
import os
import sys

import numpy as np

from . import config as C
from . import text_en as T
from .calibration import Calibration
from .colour import ColourLUT
from .masking import MaskNoise, mask_reference, visual_mask_textures
from .plan import (adapt_correct, adapt_response, build_plan, feedback_ok, layout_trial, option_labels,
                   option_permutation, rest_message, score_response)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def participant_dialog():
    from psychopy import core, data, gui
    info = {"participant": f"{np.random.randint(0, 999999):06d}", "session": "001"}
    dlg = gui.DlgFromDict(dictionary=info, sortKeys=False, title=C.EXP_NAME, alwaysOnTop=True)
    if not dlg.OK:
        core.quit()
    info["date"] = data.getDateStr()
    info["expName"] = C.EXP_NAME
    return info


def run_session(argv=None, require_headphones=True):
    argv = sys.argv if argv is None else argv
    pilot = "--pilot" in argv
    os.chdir(ROOT)

    # 1. 音訊裝置（名稱選取；沒有耳機就停）
    from audio_pick import register_label, setup_from_argv
    audio = setup_from_argv(argv, require_headphones=require_headphones)
    register_label(audio["name"], label=C.SPEAKER_LABEL)

    # 2. 對話框、資料檔、視窗
    from psychopy import core, logging
    from .display import Display, ExperimentAbort, make_handler, make_window
    info = participant_dialog()
    info["audio_device"] = audio["name"]
    exp, filename = make_handler(info, ROOT)
    win = make_window(pilot=pilot)
    info["frameRate"] = win._monitorFrameRate
    global_clock = core.Clock(format="float")
    logging.setDefaultClock(global_clock)
    disp = Display(win, exp, global_clock)

    # 3. 計畫與刺激來源
    rng = np.random.default_rng()
    plan = build_plan(rng)
    lut = ColourLUT()
    from snr_runtime import SNRStimulus
    snd = SNRStimulus(outdir=filename + "_snr", token=C.SND_TOKEN)
    mask_sr, mask_ref_rms = mask_reference()
    masker = MaskNoise(snd.outdir, mask_sr, mask_ref_rms)
    calib = Calibration(lut)
    print(f"[plan] calibration {'joint ' + str(plan.n_adapt) if plan.joint else f'colour {plan.n_adapt_col} + sound {plan.n_adapt_snd}'}"
          f" + practice {plan.n_practice} + main {plan.n_main} = {plan.n_trials} trials; main {plan.n_blocks} blocks x {C.BLOCK_SIZE}")
    print(f"[colour] {lut.meta}; usable arc [{lut.arc_min:.2f}, {lut.arc_max:.2f}]")
    print(f"[mask] {C.MASK_DUR}s visual={C.MASK_VISUAL} audio={C.MASK_AUDIO}; ref RMS={mask_ref_rms:.4f}")

    levels = None            # 主實驗的水準：校準收尾時設定
    practice_correct = 0
    try:
        disp.show_text("Intro", T.INTRO)
        exp.nextEntry()
        disp.show_text("instruction_normal", T.INSTRUCTION_JOINT if plan.joint else T.INSTRUCTION,
                       stim=disp.instruction_text, extra=[disp.instruction_img])
        exp.nextEntry()

        for trial_i, (cue_valid, target_item, relation, target_serial) in enumerate(plan.trials):
            phase = plan.phase_of(trial_i)
            is_adapt = phase.startswith("adapt")
            exp.addData("thisRow.t", global_clock.getTime(format="float"))

            # ── 水準與目標 ──
            adapt_x = None
            if is_adapt:
                adapt_x, lv, target_item = calib.propose(phase, rng)
            else:
                lv = levels
            item_wavs = snd.make_many([lv.snr_levels[C.ITEM_SNR[i]] for i in range(4)], tags=[f"item{i}" for i in range(4)])
            item_hex = lv.item_hex

            # ── study ──
            lay = layout_trial(rng, target_item, target_serial)
            disp.study(item_hex, item_wavs, lay)
            exp.addData("phase", phase)
            exp.addData("is_practice", phase == "practice")
            exp.addData("trial_i", trial_i)
            exp.addData("colour_arc_lo", lv.colour_arc[0])
            exp.addData("colour_arc_hi", lv.colour_arc[1])
            exp.addData("snr_hi", lv.snr_hi)
            exp.addData("snr_lo", lv.snr_lo)
            exp.addData("quad_content", str(lay.quad_content))
            exp.addData("time_perm", str(lay.time_perm))
            exp.addData("target_serial", int(target_serial))
            for q, nm in enumerate(C.SLOT_NAMES):
                it = lay.quad_content[q]
                exp.addData(f"study{nm}", f"{item_hex[it]}|{item_wavs[it]}|t={lay.start_times[q]}")

            # ── mask ──
            vis_seed = noise = None
            textures = None
            if C.MASK_VISUAL:
                import snr_audio
                vis_seed = snr_audio.new_seed()
                textures = visual_mask_textures(vis_seed)
            if C.MASK_AUDIO:
                noise = masker.make(trial_i)
            m = disp.mask(textures, noise["path"] if noise else None)
            on = [v for v in m["onsets"].values() if v is not None]
            off = [v for v in m["offsets"].values() if v is not None]
            exp.addData("mask_onset", min(on) if on else "")
            exp.addData("mask_offset", max(off) if off else "")
            exp.addData("mask_dur", C.MASK_DUR if (C.MASK_VISUAL or C.MASK_AUDIO) else 0.0)
            exp.addData("mask_visual", C.MASK_VISUAL)
            exp.addData("mask_audio", C.MASK_AUDIO)
            exp.addData("mask_vis_seed", vis_seed if C.MASK_VISUAL else "")
            exp.addData("mask_noise_seed", noise["seed"] if noise else "")
            exp.addData("mask_noise_db", noise["db"] if noise else "")
            exp.addData("mask_noise_rms", noise["rms"] if noise else "")
            exp.addData("mask_ref_rms", mask_ref_rms)
            if noise:
                exp.addData("snd_mask", noise["summary"])

            # ── cue：框在 cued_item 的位置，內容是 target_item ──
            cued_item = target_item ^ relation
            cue_idx = lay.quad_content.index(cued_item)
            cue_wav = snd.make(lv.snr_levels[C.ITEM_SNR[target_item]], tag="probe")
            disp.cue(C.POS[cue_idx], item_hex[target_item], cue_wav)
            exp.addData("cue_valid", int(cue_valid))
            exp.addData("relation", int(relation))
            exp.addData("relation_name", C.REL_NAME[relation])
            exp.addData("target_item", int(target_item))
            exp.addData("cued_item", int(cued_item))
            exp.addData("target_quad", int(lay.target_quad))
            exp.addData("cue_idx", int(cue_idx))
            exp.addData("cue_color", item_hex[target_item])
            exp.addData("cue_audi", cue_wav)
            exp.addData("target_snr", C.SNR_NAMES[C.ITEM_SNR[target_item]])
            exp.addData("cued_snr", C.SNR_NAMES[C.ITEM_SNR[cued_item]])
            exp.addData("POS_cue", str(C.POS[cue_idx]))
            exp.addData("target_pos", str(C.POS[lay.target_quad]))

            # ── task ──
            opt_perm = option_permutation(rng)
            disp.set_options(opt_perm, item_hex, option_labels(lv.snr_levels))
            r = disp.task()
            sc = score_response(r["key"], opt_perm, target_item, cued_item)
            outcome, is_correct = sc.outcome, sc.is_correct
            exp.addData("opt_perm", str(opt_perm))
            exp.addData("pressed_key", sc.pressed_key)
            exp.addData("chosen_slot", sc.chosen_slot)
            exp.addData("chosen_item", sc.chosen_item)
            exp.addData("err_rel", sc.err_rel)
            exp.addData("err_rel_name", C.REL_NAME[sc.err_rel] if sc.err_rel else "")
            exp.addData("rt", r["rt"])
            exp.addData("key_resp_2.keys", r["key"])
            exp.addData("key_resp_2.rt", r["rt"])
            for row in snd.drain_log():
                exp.addData("snd_" + row["tag"], row["summary"])
            if phase == "practice" and is_correct:
                practice_correct += 1

            # ── Psi 更新（校準時只看被校的維度）──
            adapt_resp = adapt_corr = None
            if is_adapt and sc.chosen_item is not None:
                adapt_resp = adapt_response(phase, sc.chosen_item)
                calib.update(phase, adapt_resp)
                adapt_corr = adapt_correct(adapt_resp, adapt_x)
                outcome, is_correct = ("correct" if adapt_corr else "other"), bool(adapt_corr)
            exp.addData("outcome", outcome)
            exp.addData("is_correct", is_correct)
            if is_adapt:
                if phase == "adapt_joint":                 # 兩維各一欄；分開校時 adapt_x 是單一數值（與原檔相同）
                    exp.addData("adapt_x", adapt_x[0])
                    exp.addData("adapt_s", adapt_x[1])
                    exp.addData("adapt_resp", adapt_resp[0] if adapt_resp else None)
                    exp.addData("adapt_resp_s", adapt_resp[1] if adapt_resp else None)
                else:
                    exp.addData("adapt_x", adapt_x)
                    exp.addData("adapt_resp", adapt_resp)
                exp.addData("adapt_corr", adapt_corr)
                exp.addData("calib_joint", int(plan.joint))

            # ── 校準收尾：主實驗的水準 ──
            if trial_i == plan.n_adapt - 1:
                levels, lams, raw = calib.finish(C.OVERALL_ACC, posterior_path=filename + "_posterior")
                exp.addData("psi_col_alpha", float(lams[0][0]))
                exp.addData("psi_col_beta", float(lams[0][1]))
                exp.addData("psi_snd_alpha", float(lams[1][0]))
                exp.addData("psi_snd_beta", float(lams[1][1]))
                exp.addData("final_colour_arc_lo", levels.colour_arc[0])
                exp.addData("final_colour_arc_hi", levels.colour_arc[1])
                exp.addData("final_snr_hi", levels.snr_hi)
                exp.addData("final_snr_lo", levels.snr_lo)
                exp.addData("snr_token", C.SND_TOKEN)
                exp.addData("snr_lim", C.SNR_LIM)
                print(f"[AGRT] colour lambda={lams[0]}  sound lambda={lams[1]}  raw={raw}")
                print(f"[AGRT] COLOUR_ARC={levels.colour_arc} -> {levels.colour_hex}; SNR hi={levels.snr_hi:+.2f} lo={levels.snr_lo:+.2f} dB")

            # ── 回饋（校準與練習）──
            if phase != "main" and C.FEEDBACK_PRE_MAIN:
                msg = T.FEEDBACK["correct" if is_correct else ("noresp" if sc.chosen_item is None else "wrong")] + T.FEEDBACK_SUFFIX
                disp.feedback(opt_perm, lambda it: feedback_ok(phase, it, target_item), msg)

            # ── 休息 ──
            msg = rest_message(trial_i + 1, plan, practice_correct)
            if msg:
                disp.show_text("rest", msg, stim=disp.rest_text)
            exp.nextEntry()                      # 手寫：沒有這行 600 試會疊成一列（CLAUDE.md）
    except ExperimentAbort as e:
        print(f"[abort] escape during {e}")
    finally:
        exp.saveAsWideText(filename + ".csv", delim="auto")
        exp.saveAsPickle(filename)
        logging.flush()
        win.flip()
        win.close()
    print("done:", filename)
    core.quit()
