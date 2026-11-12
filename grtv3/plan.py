"""純邏輯：試次計畫、每試版面、計分、休息訊息。不 import PsychoPy，tests/test_grtv3.py 直接測。

item 編碼（CLAUDE.md）：item = colour_bit + 2*sound_bit；target ^ 1 只差顏色、^ 2 只差聲音、^ 3 都差。
"""
from dataclasses import dataclass

from . import config as C
from . import text_en as T


# ── 計畫 ──────────────────────────────────────────────────────────────────

def build_main_plan(rng, n_valid=C.N_VALID_PER_CELL, n_invalid=C.N_INVALID_PER_CELL, block_size=C.BLOCK_SIZE):
    """主實驗：每格 (cue_valid, target_item, relation, target_serial) 的試次數固定，輪流發到各區塊再各自打散
    （GRTv3_ada_EN.py:438-456）。回傳 (plan, n_blocks)；每個區塊內 valid:invalid 都是 2:1。"""
    strata = []
    for it in range(4):
        for sp in range(4):
            strata.append(((1, it, 0, sp), n_valid))
            for rel in (1, 2, 3):
                strata.append(((0, it, rel, sp), n_invalid))
    n_main = sum(n for _, n in strata)
    n_blocks = n_main // block_size
    blocks = [[] for _ in range(n_blocks)]
    k = 0
    for cond, n in strata:
        for _ in range(n):
            blocks[k % n_blocks].append(cond)
            k += 1
    for b in range(n_blocks):
        blocks[b] = [blocks[b][i] for i in rng.permutation(len(blocks[b]))]
    return [t for b in blocks for t in b], n_blocks


def build_adapt_plan(rng, n):
    """校準試次：全部 valid，target_serial 0–3 平均；target_item 之後由 Psi 的提議決定（None 佔位）。"""
    sp = [i % 4 for i in range(n)]
    return [(1, None, 0, int(sp[i])) for i in rng.permutation(n)]


def build_practice_plan(rng, n=C.N_PRACTICE):
    """練習：2/3 valid、1/3 invalid，從全部格子中不重複抽，再打散。"""
    n_valid = round(n * 2 / 3)
    n_invalid = n - n_valid
    valid_pool = [(1, it, 0, sp) for it in range(4) for sp in range(4)]
    inval_pool = [(0, it, rel, sp) for it in range(4) for rel in (1, 2, 3) for sp in range(4)]
    plan = ([valid_pool[i] for i in rng.permutation(len(valid_pool))[:n_valid]]
            + [inval_pool[i] for i in rng.permutation(len(inval_pool))[:n_invalid]])
    return [plan[i] for i in rng.permutation(len(plan))]


@dataclass
class Plan:
    trials: list
    n_adapt_col: int
    n_adapt_snd: int
    n_practice: int
    n_main: int
    n_blocks: int
    joint: bool = False      # True：校準是一個 joint 區塊（n_adapt_col 放 joint 試次數，n_adapt_snd = 0）

    @property
    def n_adapt(self):
        return self.n_adapt_col + self.n_adapt_snd

    @property
    def n_pre(self):
        return self.n_adapt + self.n_practice

    @property
    def n_trials(self):
        return len(self.trials)

    def phase_of(self, trial_i):
        if trial_i < self.n_adapt:
            if self.joint:
                return "adapt_joint"
            return "adapt_col" if trial_i < self.n_adapt_col else "adapt_snd"
        return "practice" if trial_i < self.n_pre else "main"


def build_plan(rng, n_adapt_col=C.N_ADAPT_COL, n_adapt_snd=C.N_ADAPT_SND, n_practice=C.N_PRACTICE,
               joint=bool(C.CALIB_JOINT), n_adapt_joint=C.N_ADAPT_JOINT):
    main, n_blocks = build_main_plan(rng)
    if joint:
        adapt = build_adapt_plan(rng, n_adapt_joint)
        n_adapt_col, n_adapt_snd = n_adapt_joint, 0
    else:
        adapt = build_adapt_plan(rng, n_adapt_col) + build_adapt_plan(rng, n_adapt_snd)
    trials = adapt + build_practice_plan(rng, n_practice) + main
    return Plan(trials, n_adapt_col, n_adapt_snd, n_practice, len(main), n_blocks, joint=joint)


# ── 每試版面 ──────────────────────────────────────────────────────────────

@dataclass
class Layout:
    quad_content: list   # 四個角落（UR UL BL BR）各放哪個 item
    target_quad: int     # 目標 item 在哪個角落
    time_perm: list      # 四個角落各用 STUDY_START 的哪一個時槽

    @property
    def start_times(self):
        return [C.STUDY_START[k] for k in self.time_perm]


def layout_trial(rng, target_item, target_serial):
    """角落內容隨機；目標的時槽強制為 target_serial（GRTv3_ada_EN.py:1178-1183）。"""
    quad_content = [int(x) for x in rng.permutation(4)]
    target_quad = quad_content.index(target_item)
    time_perm = [int(x) for x in rng.permutation(4)]
    i = time_perm.index(target_serial)
    time_perm[i], time_perm[target_quad] = time_perm[target_quad], time_perm[i]
    return Layout(quad_content, target_quad, time_perm)


def option_permutation(rng):
    """選項畫面：四個 item 隨機分到 UR UL BL BR。"""
    return [int(x) for x in rng.permutation(4)]


def option_labels(snr_levels):
    """每個 item 的音節標籤：SNR > 0 → [bi]，否則 [pi]（GRTv3_ada_EN.py:2014-2015）。"""
    return [C.SND_LABELS[0] if snr_levels[C.ITEM_SNR[i]] > 0 else C.SND_LABELS[1] for i in range(4)]


# ── 計分 ──────────────────────────────────────────────────────────────────

@dataclass
class Score:
    pressed_key: object
    chosen_slot: object
    chosen_item: object
    outcome: str
    err_rel: object

    @property
    def is_correct(self):
        return self.outcome == "correct"


def score_response(pressed_key, opt_perm, target_item, cued_item):
    """GRTv3_ada_EN.py:2336-2361。opt_perm 的順序是 UR UL BL BR。"""
    slot_item = dict(zip(C.SLOT_NAMES, opt_perm))
    if isinstance(pressed_key, list):
        pressed_key = pressed_key[-1] if pressed_key else None
    if pressed_key in C.KEY_TO_SLOT:
        chosen_slot = C.KEY_TO_SLOT[pressed_key]
        chosen_item = slot_item[chosen_slot]
    else:
        chosen_slot = chosen_item = None
    if chosen_item is None:
        outcome = "noresp"
    elif chosen_item == target_item:
        outcome = "correct"
    elif chosen_item == cued_item:
        outcome = "intrusion"
    else:
        outcome = "other"
    err_rel = None if (chosen_item is None or outcome == "correct") else (chosen_item ^ target_item)
    return Score(pressed_key, chosen_slot, chosen_item, outcome, err_rel)


def adapt_response(phase, chosen_item):
    """校準試次餵給 Psi 的反應：顏色區塊 colour bit（1 = pink）、聲音區塊 sound bit（1 = [bi]）；
    joint 模式回傳 (colour_resp, sound_resp)，即 AGRTHandler.addResponse 要的 tuple。"""
    col = chosen_item & 1
    snd = 1 if (chosen_item >> 1) == 0 else 0
    if phase == "adapt_col":
        return col
    if phase == "adapt_snd":
        return snd
    return (col, snd)


def adapt_correct(resp, adapt_x):
    """Psi 提議 x > 0 時「正確」反應是 1（GRTv3_ada_EN.py:2386）。joint：兩維都對才算對。"""
    if isinstance(resp, tuple):
        return int(all(r == int(x > 0) for r, x in zip(resp, adapt_x)))
    return int(resp == int(adapt_x > 0))


def feedback_ok(phase, item, target_item):
    """回饋要框哪些選項：校準時只看被校的那個維度（另一維度不變，兩個選項內容相同），練習時只框目標。"""
    if phase == "adapt_col":
        return (item & 1) == (target_item & 1)
    if phase == "adapt_snd":
        return (item >> 1) == (target_item >> 1)
    return item == target_item


# ── 休息訊息 ──────────────────────────────────────────────────────────────

def rest_message(n_done, plan, practice_correct):
    """回傳要顯示的休息文字；不用休息時回傳 None（GRTv3_ada_EN.py:2465-2502）。"""
    if plan.joint:
        if n_done == plan.n_adapt:
            return T.rest_after_joint(plan.n_practice)
    else:
        if n_done == plan.n_adapt_col:
            return T.rest_after_colour()
        if n_done == plan.n_adapt:
            return T.rest_after_sound(plan.n_practice)
    if n_done == plan.n_pre:
        return T.rest_after_practice(practice_correct, plan.n_practice, plan.n_blocks)
    if n_done > plan.n_pre and (n_done - plan.n_pre) % C.BLOCK_SIZE == 0 and n_done < plan.n_trials:
        return T.rest_between_blocks((n_done - plan.n_pre) // C.BLOCK_SIZE, plan.n_blocks)
    return None
