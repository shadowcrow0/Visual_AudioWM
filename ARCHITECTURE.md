# ARCHITECTURE.md — 系統架構、耦合規則、後續更新方法

2026-11-17。給之後任何人或任何模型：**改東西之前先讀這份，不要讀整個 repo**。
原則：一份程式只有一份來源；參數只定義一次；UI／受試者體驗不因重構而變；能不讀的大檔不讀。

---

## 1. 三個 repo、各自的角色

```
   Visual_AudioWM（私有，主開發）          adaptiveSFT（私有，模型套件）        VAWM_run（公開，實驗機用的快照）
   ─────────────────────────────         ──────────────────────────         ────────────────────────────────
   實驗腳本、刺激、校準邏輯、文件          LNRM 擬合、反解 H/L、SIC、DDM       Visual_AudioWM/ 的執行檔 + adaptiveSFT/ 套件
   受試者資料 data/（只放這裡）            scripts/ 檢定力掃描（研發用）       數字開頭的 .bat、README、RUN_ON_*.md
   review/ 決策與查核筆記                  tests/                              ⚠ 永遠不放 data/；永遠由 make_run_bundle.py 產生，不手改
```

VAWM_run 的內容 = `make_run_bundle.py --repo <VAWM_run>` 的輸出。要改實驗機上的檔案，改 Visual_AudioWM 再重打包；.bat 與 README 例外（它們只住在 VAWM_run）。

## 2. 執行層的檔案與相依（只有這些會在實驗機上跑）

```
   SFT 實驗                                          GRT 實驗
   ──────────────────────────────────────────       ───────────────────────────────────────────
   SFT_WM_nochange_EN.py   流程（校準→練習→主實驗）     GRTv3_ada_EN.py            Builder 單檔（main）
         │                                                 │ 或 GRTv3_ada_modular.py + grtv3/（branch modular）
         ├─> adaptive_vawm_EN.py   校準邏輯、常數、d′ 表     ├─> AGRT.py             兩個 Psi（numpy/scipy）
         │        └─> adaptiveSFT  fit_lnrm_v2、反解       ├─> snr_audio.py         即時混噪音
         ├─> audio_pick.py        選輸出裝置               │     └─> snr_runtime.py
         ├─> sft_noise.py         每次呈現即時混噪音        ├─> agrt_colour_lut.json 顏色弧長表
         │     └─> snr_audio.py（讀寫 wav、輸出位準）      └─> audio_pick.py
         └─> stimuli/consonant_dprime.csv、T??_*3.wav
   sound_perception_EN.py  純知覺測試 ─> adaptive_vawm_EN、stimuli/sound_pairs.csv
   debug_adaptivesft_EN.py 全鏈診斷   ─> adaptive_vawm_EN、audio_pick（exit code 1 = 有問題）
```

相依方向只能往下：腳本 → `adaptive_vawm_EN` → `adaptiveSFT`；腳本 → `audio_pick`。**`adaptive_vawm_EN` 不得 import 任何實驗腳本或 PsychoPy**（它在沒有 PsychoPy 的機器上要能測）。

## 3. 唯一來源表（Single source of truth）— 改數字只能改這一格

| 東西 | 唯一定義處 | 誰讀它 |
|---|---|---|
| 顏色／聲音校準水準 `LEVELS_COLOUR` / `LEVELS_AUDIO`（顏色 [8,14,20,28,45,70]，DECISIONS 23、25） | `adaptive_vawm_EN.py` | SFT 腳本（import）、純知覺測試、debug |
| 校準模型與規則 `CALIB_LINK` (ogival) / `CALIB_BIAS` (b₀) / `CALIB_PLATEAU` (.9)；L = 50% 點、H = 到頂點 | `adaptive_vawm_EN.py`（`fit_points`） | SFT 腳本 `fit_calibration`（DECISIONS 25） |
| Houpt 漂移差目標 `H_TARG_DEFAULT` / `L_TARG_DEFAULT`（8.0 / 1.3） | `adaptive_vawm_EN.py` | 只有 `LNRMCalibrator`（R 對齊測試、debug），實驗不用 |
| 聲音軸（每對子音的 d′） | `stimuli/consonant_dprime.csv` ← `make_sound_pairs.py` ← `consonant_distance.py`（離線） | `adaptive_vawm_EN.load_dprime`；實驗只讀 csv |
| 純知覺測試的 80 對 | `stimuli/sound_pairs.csv` ← `make_sound_pairs.py` | `sound_perception_EN.py` |
| 顏色弧長 → hex | `agrt_colour_lut.json` ← `agrt_setup.py`（離線，需 colour-science） | GRT 腳本、`grtv3/colour.py` |
| 語音 token 鍵名 `'bi'` | `snr_audio.SPEECH_FILES` | `snr_runtime` 預設、GRT 腳本的 `SND_TOKEN`、`grtv3/config.py` |
| SFT 的試次結構（探測比例、區塊數、反應窗）、`SND_SNR_DB`（整場 SNR，None = 乾淨） | `SFT_WM_nochange_EN.py` 檔頭 36–60 行 | 只有它自己 |
| 噪音種子規則 `seed_for(phase, 試次, slot)`，探測另一個種子 | `sft_noise.py` docstring | SFT 腳本 `snd_path`、`sound_perception_EN.py` |
| GRT 的常數（Builder 版） | `GRTv3_ada_EN.py` 434–479、866–889 行 | 只有它自己（modular 版在 `grtv3/config.py`） |
| 實驗機的 PsychoPy 路徑 | `VAWM_run/_env.bat` | 所有 .bat |
| 每個數字為什麼是這個值 | `DECISIONS.md` | 人 |

同一個值出現在第二個地方 = bug。新增常數時先問：它屬於上表哪一列？

## 4. 已排除的高耦合（2026-11-17 這次做的）

| 之前 | 問題 | 現在 |
|---|---|---|
| 每支腳本有 zh / `_EN` 兩份（7 對、~9000 行） | 每個修改要做兩次；AST 比對證實只差 print/raise 的語言 | 只留 `_EN`（實驗機跑的那份）；zh 版凍結在 `archive/zh_comments/` |
| `LEVELS_*`、`H_TARG`/`L_TARG` 在 SFT 腳本與 `adaptive_vawm` 各定義一次 | 改一邊忘另一邊 | SFT 腳本 import |
| 舊流程 `VAWM_nobox.py`(7462 行)+`VAWM_calibrate`+`make_adaptive_blocks`+`make_simple_psyexp`+`*.psyexp`+`generate_trials.py` 留在根目錄 | 讀 repo 時被當成現行程式；佔 token | 移到 `archive/legacy_vawm_nobox/`；bundle 不再帶 `generate_trials.py` |
| 11 個 .bat 各自寫死 PsychoPy 路徑 | 換機器要改 11 處 | `_env.bat` 一處 |
| `debug_adaptivesft_EN.py` 永遠回傳 0 | 批次檔分不出成敗 | 有 ✘ 就 exit 1 |
| 聲音軸是 `generate_trials.CONFUSION_DATA` 的對數 | 不是單位、把 voicing 對排最遠 | d′ 表（DECISIONS 21） |

## 5. 還存在、刻意不動的耦合（動了風險大於收益）

- `GRTv3_ada_EN.py` 是 Builder 輸出的 2748 行單檔。**不要在 main 上重構它**；要模組化的版本在 branch `modular`（`grtv3/`），等實驗機驗過再合併。
- `adaptive_vawm_EN.py` 裡的 `write_blocks` / `CSV_COLUMNS` 與 `stimuli/practice.csv` 只服務舊流程，但有測試覆蓋；留著不礙事。
- `adaptiveSFT` 有 `models.py`（v1）與 `models_v2.py`：**實驗用 v2**（`fit_lnrm_v2`，pm.LogNormal）；v1 給 `scripts/` 的模擬與 R 對照用。不要把 v1 刪掉，也不要在實驗裡用它。
- `AGRT.py` 是第三方（GPL），只打過一個 scipy 相容補丁；不要改它，要擴充就在外面包（`grtv3/calibration.py` 就是這樣做）。

## 6. 之後要改東西的標準流程（給任何模型）

```
   1 讀：CLAUDE.md（規則）→ 本檔 §3 找到唯一來源 → DECISIONS.md 對應那一條（為什麼是這個值）
   2 改：只改唯一來源那一格；UI 文字不動（除非任務就是改文字）
   3 驗：pytest tests/                 （Visual_AudioWM，27+ 個，含 PyMC 的 LNRM 回復測試；~1 分鐘）
          pytest tests/test_grtv3.py    （branch modular）
          python -m py_compile <改過的 .py>   （有 PsychoPy 的檔案這裡只能編譯，不能跑——要說清楚）
   4 記：DECISIONS.md 加一條（決定 / 為什麼 / 在哪），不要改舊條目的內容
   5 包：python make_run_bundle.py --repo <VAWM_run 的 checkout> → 在 VAWM_run commit + push
   6 推：Visual_AudioWM 的 main 與工作分支；改到 grtv3/ 或共用模組時把 main 合進 modular
```

## 7. 寫程式的規則

1. **一份來源**：不開 zh/EN 分支、不複製檔案改幾行。語言差異用字串表（`grtv3/text_en.py` 的做法）。
2. **新常數先進 §3 的表**，再寫程式。
3. **純邏輯與 PsychoPy 分層**：計畫、計分、尺度、擬合包裝不 import psychopy，放可測的模組；PsychoPy 只出現在流程檔與 `display` 層。
4. **資料欄位名是介面**：`outcome / is_correct / rt / phase / x / adapt_x / audio_device / snd_*` 等名稱不改；新增欄位可以，刪改要同時改分析腳本並記 DECISIONS。
5. **UI 不變**：受試者看到的文字、按鍵、時序、回饋是研究設計的一部分；重構不得改動，改動要走 DECISIONS。
6. **錯誤要出聲**：找不到檔案、超出範圍、裝置開不了 → 丟錯或印清楚的 ✘ 並 exit 1；不要悄悄退回預設值（`colour_for` 丟錯而不是截掉，就是這個意思）。
7. **註解寫為什麼**，中英皆可；特別是「試過更簡單的做法、為何不行」要留著。
8. **資料不進公開 repo**：`data/` 只在 Visual_AudioWM；VAWM_run 的 `.gitignore` 擋 data，但人為 `git add -f` 擋不住——不要。
9. **archive/ 是凍結區**：不改、不打包、不 import（例外：`consonant_distance.py` 讀 `archive/legacy_vawm_nobox/generate_trials.py` 的舊次數表，只為了 `confusion` 這個對照指標）。
10. commit 訊息說「改了什麼、為什麼」，不放模型名稱。

## 8. Token 預算：做什麼任務讀哪些檔（不要讀的也列出來）

| 任務 | 讀 | 不要讀 |
|---|---|---|
| 改 SFT 的試次數／比例／反應窗 | `SFT_WM_nochange_EN.py` 1–60 行、DECISIONS 14–18 | 其餘 400 行 |
| 改校準水準／規則 | `adaptive_vawm_EN.py` 105–120 行與 `fit_points`、DECISIONS 23、25 | `LNRMCalibrator` 內部（R 對齊用） |
| 改聲音軸 | `make_sound_pairs.py`、`consonant_distance.py` 的 docstring、`review/GRT擬合MN_B路線_D2.md` §4 | 擬合細節 |
| 聲音裝置問題 | `audio_pick.py` 檔頭 docstring、`_env.bat` | 其他 |
| SFT 噪音／SNR／種子 | `sft_noise.py` docstring、DECISIONS 24 | `snr_audio.py` 內部（它只借 wav 讀寫與位準常數） |
| 擬合不收斂／點找不到 | `adaptivesft/salience.py` `find_points`、`models_v2.py`（`bias`）、DECISIONS 25 | `models.py`（v1）、`find_salience`（Houpt 對齊） |
| GRT 流程 | branch modular 的 `grtv3/session.py`（200 行）與 `review/GRTv3_ada_模組化.md` 的對照表 | `GRTv3_ada_EN.py` 全文（2748 行）——用 grep 定位行號即可 |
| 寫論文方法 | `review/paper_draft_sound_scale_EN.md`、`論文規格書_聲音軸.md` | 其他 review 筆記（它們是 dated records） |
| 任何事 | — | `archive/`、`review/papers/*.pdf`（除非任務是查原文）、`data/*.psydat` |

grep 比讀全文便宜：`grep -n "^[A-Z_]* *=" SFT_WM_nochange_EN.py` 就能列出所有旋鈕。

## 9. 分支

```
   main                                    現行；實驗機用的就是這裡打包出去的
   claude/grtv3-ada-adaptivesft-feasibility-btx6w8   工作分支，與 main 等價（每次 push 兩邊都推）
   modular                                 GRTv3_ada 的 grtv3/ 套件版；main 定期合進來；驗過再反向合併
```
