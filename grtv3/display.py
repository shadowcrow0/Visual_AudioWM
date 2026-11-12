"""PsychoPy 層：視窗、刺激物件、一個通用的 routine 執行器。

Builder 把每個 routine 展開成幾百行的逐幀狀態機；這裡用一個 `run_timeline()` 取代：
給它一串 (stimulus, onset, duration)，它負責在正確的幀開、關、播放、收鍵、記 onset、處理 Esc。
時序規則與 Builder 相同：onset 以 routine 時鐘的下一次 flip 時間判斷（tThisFlip >= onset − tol），
offset 以全域時鐘的實際起始時間 + duration 判斷。
"""
import os

from psychopy import core, data, logging, sound, visual
from psychopy.hardware import keyboard

from . import config as C


class ExperimentAbort(Exception):
    """受試者或實驗者按了 Esc。"""


class Timed:
    """時間軸上的一個項目。kind = 'draw'（setAutoDraw）或 'sound'（play/stop）。duration None = 到 routine 結束。"""

    def __init__(self, stim, onset=0.0, duration=None, kind="draw", name=None):
        self.stim, self.onset, self.duration, self.kind = stim, onset, duration, kind
        self.name = name or getattr(stim, "name", "stim")
        self.t_on = self.t_off = None

    def start(self, win):
        if self.kind == "sound":
            self.stim.play(when=win)
        else:
            self.stim.setAutoDraw(True)

    def stop(self):
        if self.kind == "sound":
            self.stim.stop()
        else:
            self.stim.setAutoDraw(False)

    def finished_early(self):
        return self.kind == "sound" and getattr(self.stim, "isFinished", False)


class Display:
    def __init__(self, win, exp, global_clock, speaker=C.SPEAKER_LABEL):
        self.win, self.exp, self.clock = win, exp, global_clock
        self.kb = keyboard.Keyboard()
        self.routine_clock = core.Clock()
        self.frame_dur = 1.0 / round(win._monitorFrameRate) if getattr(win, "_monitorFrameRate", None) else 1.0 / 60.0
        self._build(speaker)

    # ── 刺激物件（與 GRTv3_ada_EN.py 的 Initialize components 相同的參數）──
    def _build(self, speaker):
        w = self.win
        txt = dict(font="Arial", color="white", colorSpace="rgb", wrapWidth=26, alignText="left", anchorHoriz="center")
        self.text = visual.TextStim(w, name="text", text="", pos=(0, 0), height=1.0, wrapWidth=24, **{k: v for k, v in txt.items() if k != "wrapWidth"})
        self.instruction_text = visual.TextStim(w, name="instruction_normal_text", text="", pos=(0, 3.0), height=0.65, **txt)
        self.instruction_img = visual.ImageStim(w, name="instruction_key_img", image="stimuli/box_keys_iokl.png",
                                                pos=(0, -7.0), size=(5.5, 5.0), interpolate=True)
        self.rest_text = visual.TextStim(w, name="rest_text", text="", pos=(0, 0), height=0.8, **txt)
        self.adapt_text = visual.TextStim(w, name="adapt_text", text="", pos=(0, 0), height=0.8, **txt)
        self.fixation = visual.ShapeStim(w, name="Fixation", vertices="cross", size=(0.5, 0.5), lineColor="white", fillColor="white", interpolate=True)
        sound.Sound.backend = "ptb"
        self.study_colour, self.study_sound = [], []
        for i, nm in enumerate(C.SLOT_NAMES):
            self.study_colour.append(visual.Rect(w, name=f"color{nm}", width=C.SQUARE, height=C.SQUARE, colorSpace="hex",
                                                 lineColor=None, fillColor="white", pos=C.POS[i], interpolate=True))
            snd = sound.Sound("A", secs=-1, stereo=True, hamming=True, speaker=speaker, name=f"audi{nm}")
            snd.setVolume(1.0)
            self.study_sound.append(snd)
        self.cue_frame = visual.Rect(w, name="Cue", width=C.SQUARE, height=C.SQUARE, lineColor="white", fillColor=None, interpolate=True)
        self.cue_colour = visual.Rect(w, name="targetC", width=C.SQUARE, height=C.SQUARE, colorSpace="hex", lineColor=None, fillColor="white", interpolate=True)
        self.cue_sound = sound.Sound("A", secs=1.0, stereo=True, hamming=True, speaker=speaker, name="targetAudi")
        self.cue_sound.setVolume(1.0)
        self.box = visual.ImageStim(w, name="BOX", image="stimuli/box.png", pos=(0, 0), size=(C.BOX_SIZE, C.BOX_SIZE), interpolate=True)
        self.opt_text = [visual.TextStim(w, name=nm, text="", font="Arial", height=4.0, color="white", colorSpace="rgb") for nm in C.SLOT_NAMES]
        self.opt_colour = [visual.Rect(w, name=f"v{nm}", width=C.SQUARE, height=C.SQUARE, colorSpace="hex", lineColor=None,
                                       fillColor="white", interpolate=True) for nm in C.SLOT_NAMES]
        self.fb_frame = visual.Rect(w, name="fb_frame", width=7.0, height=9.5, lineWidth=4.0, lineColor="white", fillColor=None, interpolate=True)
        import numpy as np
        self.mask_vis = [visual.GratingStim(w, name=f"mask{nm}", units="deg", tex=np.zeros((C.MASK_TEX_N, C.MASK_TEX_N)), mask=None,
                                            size=(C.MASK_SIZE, C.MASK_SIZE), sf=1.0 / C.MASK_SIZE, pos=C.POS[i], interpolate=False)
                         for i, nm in enumerate(C.SLOT_NAMES)]
        self.mask_sound = sound.Sound("A", secs=C.MASK_DUR, stereo=True, hamming=True, speaker=speaker, name="maskAudi")
        self.mask_sound.setVolume(1.0)

    # ── 通用 routine 執行器 ────────────────────────────────────────────────
    def run_timeline(self, name, items, keys=None, max_dur=None, end_when_all_done=True):
        """跑一個 routine。回傳 dict(key, rt, onsets={name: t_global}, offsets={...})。
        keys：有鍵按下就結束（Esc 永遠中止）。max_dur：到時就結束（cue 用 2.0）。"""
        win, exp = self.win, self.exp
        tol = C.FRAME_TOLERANCE
        exp.currentRoutine = data.Routine(name=name, components=[it.stim for it in items])
        exp.addData(f"{name}.started", self.clock.getTime(format="float"))
        self.routine_clock.reset()
        self.kb.clock.reset()
        self.kb.clearEvents()
        pressed = []
        for it in items:
            it.t_on = it.t_off = None
        running = True
        while running:
            t_flip = win.getFutureFlipTime(clock=self.routine_clock)
            t_flip_global = win.getFutureFlipTime(clock=None)
            for it in items:
                if it.t_on is None and t_flip >= it.onset - tol:
                    it.t_on = t_flip_global
                    it.start(win)
                    exp.addData(f"{it.name}.started", t_flip_global)
                elif it.t_on is not None and it.t_off is None and it.duration is not None:
                    if t_flip_global > it.t_on + it.duration - tol or it.finished_early():
                        it.t_off = t_flip_global
                        it.stop()
                        exp.addData(f"{it.name}.stopped", t_flip_global)
            if keys:
                # 同時收「仍按著」與「已放開」的鍵：Builder 只收前者，快速敲擊會漏（DECISIONS 17）
                got = self.kb.getKeys(keyList=keys, waitRelease=False, clear=True) + \
                      self.kb.getKeys(keyList=keys, waitRelease=True, clear=True)
                if got:
                    pressed.extend(got)
                    running = False
            if self.kb.getKeys(keyList=["escape"], waitRelease=False):
                raise ExperimentAbort(name)
            if max_dur is not None and self.routine_clock.getTime() >= max_dur:
                running = False
            if running and end_when_all_done and all(it.duration is not None and it.t_off is not None for it in items):
                running = False
            if running:
                win.flip()
        for it in items:
            if it.t_on is not None and it.t_off is None:
                it.stop()
                it.t_off = win.getFutureFlipTime(clock=None)
        exp.addData(f"{name}.stopped", self.clock.getTime(format="float"))
        key = min(pressed, key=lambda k: k.rt) if pressed else None
        return dict(key=key.name if key else None, rt=key.rt if key else None,
                    onsets={it.name: it.t_on for it in items}, offsets={it.name: it.t_off for it in items})

    # ── 具體畫面 ──────────────────────────────────────────────────────────
    def show_text(self, name, text, stim=None, extra=()):
        stim = stim or self.text
        stim.setText(text)
        items = [Timed(stim)] + [Timed(e) for e in extra]
        return self.run_timeline(name, items, keys=["space"], end_when_all_done=False)

    def study(self, item_hex, item_wavs, layout):
        """四個角落、各自的起始時槽；回傳 run_timeline 的結果。"""
        starts = layout.start_times
        items = [Timed(self.fixation, 0.0, C.FIX_DUR)]
        for q in range(4):
            it = layout.quad_content[q]
            self.study_colour[q].setFillColor(item_hex[it])
            self.study_colour[q].setPos(C.POS[q])
            self.study_sound[q].setSound(item_wavs[it], secs=C.STUDY_DUR, hamming=True)
            self.study_sound[q].setVolume(1.0, log=False)
            self.study_sound[q].seek(0)
            items.append(Timed(self.study_sound[q], starts[q], C.STUDY_DUR, kind="sound"))
            items.append(Timed(self.study_colour[q], starts[q], C.STUDY_DUR))
        res = self.run_timeline("study", items)
        for s in self.study_sound:
            s.pause()
        return res

    def mask(self, textures, noise_wav):
        items = []
        if C.MASK_VISUAL:
            for q, mk in enumerate(self.mask_vis):
                mk.setTex(textures[q])
                mk.setPos(C.POS[q])
                items.append(Timed(mk, 0.0, C.MASK_DUR))
        if C.MASK_AUDIO:
            self.mask_sound.setSound(noise_wav, secs=C.MASK_DUR, hamming=True)
            self.mask_sound.setVolume(1.0, log=False)
            self.mask_sound.seek(0)
            items.append(Timed(self.mask_sound, 0.0, C.MASK_DUR + self.frame_dur, kind="sound"))
        if not items:
            return dict(onsets={}, offsets={})
        res = self.run_timeline("mask", items)
        if C.MASK_AUDIO:
            self.mask_sound.pause()
        return res

    def cue(self, cue_pos, colour_hex, wav):
        """0–2 s 空白框在被提示的角落；1–2 s 框內出現目標的顏色與聲音。"""
        self.cue_frame.setPos(cue_pos)
        self.cue_colour.setFillColor(colour_hex)
        self.cue_colour.setPos(cue_pos)
        self.cue_sound.setSound(wav, secs=1.0, hamming=True)
        self.cue_sound.setVolume(1.0, log=False)
        self.cue_sound.seek(0)
        items = [Timed(self.cue_frame, 0.0, C.CUE_DUR), Timed(self.cue_colour, C.CUE_CONTENT_AT, 1.0),
                 Timed(self.cue_sound, C.CUE_CONTENT_AT, 1.0, kind="sound")]
        res = self.run_timeline("cue", items, max_dur=C.CUE_DUR)
        self.cue_sound.pause()
        return res

    def set_options(self, opt_perm, item_hex, labels):
        for q in range(4):
            x, y = C.POS[q]
            it = opt_perm[q]
            self.opt_text[q].setPos((x, y + C.TASK_GAP / 2))
            self.opt_text[q].setText(labels[it])
            self.opt_colour[q].setPos((x, y - C.TASK_GAP / 2))
            self.opt_colour[q].setFillColor(item_hex[it])

    def task(self):
        items = [Timed(self.box)] + [Timed(s) for s in self.opt_text] + [Timed(s) for s in self.opt_colour]
        return self.run_timeline("task", items, keys=C.RESPONSE_KEYS, end_when_all_done=False)

    def feedback(self, opt_perm, ok_fn, message):
        for s in self.opt_text + self.opt_colour:
            s.draw()
        for q in range(4):
            if ok_fn(opt_perm[q]):
                self.fb_frame.setPos(C.POS[q])
                self.fb_frame.draw()
        self.adapt_text.setText(message)
        self.adapt_text.draw()
        self.win.flip()
        core.wait(C.FEEDBACK_SEC)
        self.win.flip()
        core.wait(C.POST_FEEDBACK_BLANK)


def make_window(pilot=False):
    win = visual.Window(size=C.WIN_SIZE, fullscr=C.WIN_FULLSCR and not pilot, screen=0, winType="pyglet", allowGUI=False,
                        monitor=C.MONITOR, color=[0, 0, 0], colorSpace="rgb", blendMode="avg", useFBO=True, units=C.WIN_UNITS,
                        checkTiming=False)
    win._monitorFrameRate = win.getActualFrameRate(infoMsg="Measuring the screen frame rate, please wait...")
    win.hideMessage()
    return win


def make_handler(exp_info, data_dir):
    filename = os.path.join(data_dir, "data", "%s_%s_%s" % (exp_info["participant"], C.EXP_NAME, exp_info["date"]))
    exp = data.ExperimentHandler(name=C.EXP_NAME, version="", extraInfo=exp_info, runtimeInfo=None, originPath=__file__,
                                 savePickle=True, saveWideText=True, dataFileName=filename, sortColumns="time")
    logging.LogFile(filename + ".log", level=logging.EXP)
    logging.console.setLevel(logging.WARNING)
    return exp, filename
