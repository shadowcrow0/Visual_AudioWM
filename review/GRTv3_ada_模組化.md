# GRTv3_ada 模組化（branch `modular`，2026-11-12）

放棄 PsychoPy Builder 相容：`GRTv3_ada_EN.py` 的 2747 行（`run()` 一個函式 2300 行、每個 routine 展開成逐幀狀態機）
改成 `grtv3/` 套件 + 30 行的入口 `GRTv3_ada_modular.py`。流程、參數、資料欄位不變；邏輯層可以在沒有 PsychoPy 的機器上測試。

```
   GRTv3_ada_modular.py ──> grtv3/session.py ─┬─> grtv3/plan.py         試次計畫、版面、計分、休息訊息   ← 純 Python，有測試
                                              ├─> grtv3/calibration.py  兩個 Psi 的包裝（AGRT）           ← 水準規則純 Python，有測試
                                              ├─> grtv3/colour.py       弧長 → hex                        ← 有測試
                                              ├─> grtv3/masking.py      遮蔽噪音、視覺紋理                ← 有測試
                                              ├─> grtv3/config.py       所有常數
                                              ├─> grtv3/text_en.py      受試者看到的文字
                                              └─> grtv3/display.py      PsychoPy：視窗、刺激、run_timeline()
```

## 對應表（原檔行號 → 新位置）

| 原 `GRTv3_ada_EN.py` | 新 |
|---|---|
| 434–479 試次計畫（strata、區塊輪發、校準佔位、練習抽樣） | `plan.build_main_plan / build_adapt_plan / build_practice_plan / build_plan` |
| 481–493 顏色 LUT 與 `colour_for` | `colour.ColourLUT.hex_for`（超範圍一樣丟錯） |
| 866–968 Psi、SNR、遮蔽常數 | `config.py` |
| 892–924 遮蔽參考 RMS、`make_mask_noise` | `masking.mask_reference / MaskNoise.make` |
| 1139–1156 校準試次的水準與目標 | `calibration.levels_for_adapt`（純）、`Calibration.propose` |
| 1170–1183 角落內容、時槽、目標時槽 | `plan.layout_trial` |
| 1593–1606 study 的資料欄 | `session.py` 同名欄位 |
| 1784–1800 cue：`cued_item = target ^ relation`，框在 cued、內容是 target | `session.py`（同一行邏輯）+ `display.cue` |
| 2006–2033 選項畫面的排列與標籤 | `plan.option_permutation / option_labels` + `display.set_options` |
| 2336–2361 計分（correct / intrusion / other / noresp、err_rel） | `plan.score_response` |
| 2379–2396 Psi 更新與校準試次的 outcome 覆寫 | `plan.adapt_response / adapt_correct`、`Calibration.update` |
| 2398–2420 校準收尾（估 λ、反解、截範圍、存後驗） | `Calibration.finish` |
| 2422–2446 回饋（框出正確選項） | `plan.feedback_ok` + `display.feedback` |
| 2465–2502 休息訊息 | `plan.rest_message` + `text_en.rest_*` |
| 每個 routine 的逐幀狀態機（~150 行 × 7） | `display.run_timeline()` 一個函式 |

## 行為上刻意改的地方（其餘一律照舊）

1. **按鍵收集**：`run_timeline` 同時收 `waitRelease=False` 與 `True` 的鍵。Builder 只收前者，快速敲擊會漏（DECISIONS 17，SFT 線已修）。
2. **語音 token**：`config.SND_TOKEN = 'bi'`。原腳本寫 `'be'`，但 `snr_audio.SPEECH_FILES` 自 8a75be2 起只有 `'bi'`——
   **原 `GRTv3_ada(_EN).py` 在現在的 repo 會在 `_mask_ref_rms()` 就 KeyError**。這是模組化時才發現的既有問題，main 上的兩支原腳本與 `snr_runtime` 的預設值還沒改。
3. Esc 改成丟 `ExperimentAbort`，由 `session.run_session` 的 `finally` 存檔關窗；不再有 Builder 的 pause 機制。
4. 不再寫 `piloting`、`expVersion`、`psychopyVersion` 等 Builder 欄位；`--pilot` 只代表視窗模式。
5. 對話框只問 participant / session；`audio_device` 寫進 expInfo。

## 沒有驗證的

- 容器沒有 PsychoPy：`grtv3/display.py`、`grtv3/session.py` 只做了 `py_compile` 與 pyflakes，**沒有跑過**。用到的 API 都是原 Builder 檔已經在用的
  （`visual.Window/TextStim/Rect/ShapeStim/ImageStim/GratingStim`、`sound.Sound` ptb、`keyboard.Keyboard.getKeys`、`data.ExperimentHandler/Routine`、`win.getFutureFlipTime`）。
- `AGRT.AGRTHandler` 需要 PsychoPy 的 `StairHandler`，所以 `Calibration` 的測試用假 handler；真 Psi 的行為沒變（直接用 `_psi1/_psi2`）。
- 第一次在實驗機上跑建議 `--pilot`，對照 csv 欄位與 `GRTv3_ada_EN.py` 的舊資料。

## 測試

`pytest tests/test_grtv3.py`（11 個）：區塊內 2:1、每格次數、相位切點、目標時槽、計分四種結果與 err_rel、Psi 反應位元、回饋規則、標籤隨 SNR 正負、
休息訊息出現的試次、LUT 超範圍丟錯、校準水準規則、收尾截範圍、遮蔽噪音可重建。
