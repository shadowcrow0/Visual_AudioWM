# VAWM 的自適應校準（adaptive SFT）—— 怎麼跑

```
   ① VAWM_calibrate.py            區塊 1 校顏色差距、區塊 2 校子音混淆度（作業 = VAWM_nobox 的 study → probe → y/n）
          │                       METHOD = "lnrm"（原始 adaptiveSFT）或 "psi"
          │  data/<subj>_calib.json（ΔE_H、ΔE_L、count_H、count_L、假警報率、α、β）
          ▼
   ② make_adaptive_blocks.py      → stimuli/block1..6.csv（欄位與 practice.csv 完全相同）
          │
          ▼
   ③ VAWM_nobox.py                主實驗照舊（它讀 stimuli/block*.csv）
          │  data/<subj>_VAWM_<date>.csv
          ▼
   ④ python -c "from adaptive_vawm import analyze_vawm; print(analyze_vawm('data/<subj>_VAWM_….csv')['report'])"
                                  SIC / MIC / dominance → 預測架構
```

## 兩種校準方法

| | `METHOD = "lnrm"`（**原始 adaptiveSFT**，預設） | `METHOD = "psi"` |
|---|---|---|
| 強度怎麼選 | 定值刺激法：6 層固定，各 `N_PER_LEVEL`（8）試，打散 | Psi 逐試選熵最小的 x |
| 用到的資料 | 正確率 **+ RT**（「不同」試） | 只有正確率 |
| 模型 | lnrm2（`adaptivesft.models.fit_lnrm`，PyMC NUTS，區塊結束時擬合 20–40 s） | 累積常態心理計量函數 |
| 目標 | 漂移差 `H_TARG` / `L_TARG`（z₂ − z₁；預設 2.0 / 0.5，Houpt 8.0 / 1.3） | 「答不同」機率 `P_HIGH` / `P_LOW` |
| 反解 | `adaptivesft.salience.find_salience`（`adaptiveSFT_functions.R:229-232` 的公式，只用 α₂ < 0 的 draw） | `salience_levels` |
| 實驗機器要什麼 | **完整 adaptivesft 套件（PyMC）** + PsychoPy 在同一個 Python ≥ 3.10 | 只要 `psi.py`（numpy + scipy） |
| 每區塊試次 | 6 × 8 + 24 AA = 72 | 72（含 1/3 AA） |

兩者共用同一個試次產生器（`ColourCalibrator._make_trial` / `AudioCalibrator._make_trial`），差別只在強度怎麼來、
結束時怎麼算。json 的 `colour.p_high` / `p_low` 在 lnrm 下放的是 `H_TARG` / `L_TARG`；`extra.method` 標明方法，
`extra.params` 是 lnrm2 五個參數的後驗平均，`extra.high_median` / `low_median` 是逐 draw 反解的中位數（平均的對照）。

```
   lnrm：  6 層 × 8 試 ──► (x, correct, rt) ──► fit_lnrm(link="quadratic") ──► {μ, α, α₂, varZ, ψ}
                                                                                    │
                                        ΔE_H, ΔE_L  ◄── 換回物理單位 ◄── 解 2·(α·u + α₂·u²) = H_TARG / L_TARG
```

強度送進模型前縮放：顏色 `u = ΔE / 10`、聲音 `u = x + 2.7`（`x = −log10(count+1)`，平移到 ≥ 0，否則 R 的
反解公式取錯根）。聲音的層級會貼到目標子音可用的 foil，所以實際的 u 是散的（測試裡 29 個不同值），lnrm2 直接
對實際值擬合。

## 一次性

```
   實驗機器（PsychoPy）                                  分析機器（Arc 或任何有 Python ≥ 3.10 的地方）
   ─────────────────────────────────────────────         ──────────────────────────────────────────
   ① VAWM_calibrate.py  ② make_adaptive_blocks           ④ analyze_vawm、adaptiveSFT 的 pytest
   ③ VAWM_nobox.py
   lnrm：Python ≥ 3.10 + psychopy + adaptiveSFT 同一環境   pip install -e <adaptiveSFT 路徑>
   psi ：只要 adaptiveSFT/adaptivesft/psi.py（numpy+scipy）
         clone 在隔壁或設 ADAPTIVESFT_PATH，不用裝
```

實驗機器（lnrm）建環境，Windows / macOS 都一樣（Arc 不行：沒螢幕沒聲卡，`psychopy-env` 又是 Python 3.6）：

```bash
conda create -n vawm python=3.10
conda activate vawm
pip install psychopy                                         # 2023.2+ 支援 3.10
git clone https://github.com/shadowcrow0/adaptiveSFT.git     # 放在 Visual_AudioWM 隔壁，或任何地方
pip install -e ../adaptiveSFT                               # 帶 PyMC、numba
python -c "import psychopy, pymc; print(psychopy.__version__, pymc.__version__)"
python VAWM_calibrate.py
```

`adaptive_vawm.py` 先試 `import adaptivesft.psi`（裝好的套件）；沒裝或 PyMC 不在時，改成直接按檔案載入
`psi.py`，不執行套件的 `__init__.py`，所以 PsychoPy standalone 的 Python（沒有 PyMC）也能跑 psi 版校準。
`METHOD = "lnrm"` 與分析端 `analyze_vawm` 才需要完整套件。

注意：本 repo 自己有一個舊的 `adaptivesft/`（PyMC LNRM 版）會遮住裝好的套件，`adaptive_vawm.py` 會自動繞過。

## 校準的設計

| | 顏色（區塊 1） | 聲音（區塊 2） |
|---|---|---|
| 軸 x | 探測色與目標色的 CIELAB 距離（`colorpool.py` 的 `delta_e`，與 csv 的 `*_deltaE` 同單位），2–50 | −log10(混淆次數 + 1)（`generate_trials.py` 的 Miller & Nicely 表），Psi 提的值貼到目標子音可用的 foil |
| 每試 | 兩個目標色 + 兩個目標音，探測 = 其中一個目標的顏色 ± x 與它自己的聲音 | 探測 = 目標色 + 目標子音的 foil（同 talker） |
| AA 試 | 1/3（探測 = 目標），估假警報率當心理計量函數的下漸近線 | 同 |
| 模型 | `P(答「不同」\| x) = FA + (1 − FA − lapse)·Φ((x − α)/β)`，Psi 逐試選熵最小的 x | 同 |
| 輸出 | ΔE_H、ΔE_L（目標「答不同」機率 .90 / .75，可改） | count_H、count_L |
| 試次 | 72 | 72 |

H = 高 salience = 容易分：顏色 ΔE 大、子音混淆次數少（同 csv 現有的 `audio1_H_count = 1`、`L_count = 91`）。

## 參數（`VAWM_calibrate.py` 最上面）

`METHOD`；lnrm：`N_PER_LEVEL`（8）、`H_TARG` / `L_TARG`（2.0 / 0.5）、`FIT_KW`（NUTS 1000/1000/4 鏈）、層級
`LEVELS_COLOUR`（ΔE 2–45）/ `LEVELS_AUDIO` 在 `adaptive_vawm.py`；psi：`N_COLOUR` / `N_AUDIO`（72）、
`P_HIGH` / `P_LOW`（.90 / .75）、Psi 網格在 `ColourCalibrator` / `AudioCalibrator` 預設參數；共用 `P_MATCH`（1/3）。
漂移差目標怎麼選見 `adaptiveSFT/results/power_scan.csv` 與 `p6_results.md`：8.0 / 1.3 在 a = 3、v = 2 下 H 超出
範圍，2.0 / 0.5 在範圍內且 SIC 判對率最好。

## 驗證

`tests/test_adaptive_vawm.py`（7 個）：模擬受試者跑完兩個區塊回復 α/β/FA；audio 的 foil 一定是該子音表裡的；
產生的 block csv 欄位 = `practice.csv`、ΔE 在 ±0.5、H 的 count ≤ L 的；VAWM 格式的輸出 → 判對 ParallelOR
（最後這個要 PyMC，沒有就 skip；前六個只要 numpy + scipy）。
PsychoPy 的 `VAWM_calibrate.py` 在這個容器裡只做過 `py_compile`。

## 順便修的 bug

`VAWM_nobox.py` 練習回饋：`rule.csv` 的 `ResponseBox` 欄是 Cedrus 按鍵碼 3/4，鍵盤版收到 `'y'`/`'n'`，查表永遠失敗，
練習階段每一試都顯示 Incorrect。已改成 `{'y': '4', 'n': '3'}` 對照。主實驗六個 block 不算 Acc（離線算），不受影響。
