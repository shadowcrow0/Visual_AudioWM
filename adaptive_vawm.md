# VAWM 的自適應校準（adaptive SFT）—— 怎麼跑

```
   ① VAWM_calibrate.py            區塊 1 校顏色差距、區塊 2 校子音混淆度（作業 = VAWM_nobox 的 study → probe → y/n）
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

## 一次性

```bash
git clone https://github.com/shadowcrow0/adaptiveSFT.git     # 放在 Visual_AudioWM 隔壁，或任何地方
pip install -e ../adaptiveSFT                               # 或 set ADAPTIVESFT_PATH=<路徑>
```

注意：本 repo 自己有一個舊的 `adaptivesft/`（PyMC LNRM 版）會遮住裝好的套件，`adaptive_vawm.py` 會自動繞過，
但要有 `../adaptiveSFT` 或 `ADAPTIVESFT_PATH`。

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

`N_COLOUR` / `N_AUDIO`（72）、`P_HIGH` / `P_LOW`（.90 / .75）、`P_MATCH`（1/3）、Psi 的網格在 `adaptive_vawm.py` 的
`ColourCalibrator` / `AudioCalibrator` 預設參數。目標「答不同」機率要多高才有 SFT 檢定力，見
`adaptiveSFT/results/power_scan.csv` 與 `p6_results.md`：正確率目標拉不開 RT，分離要夠大。

## 驗證

`tests/test_adaptive_vawm.py`（7 個）：模擬受試者跑完兩個區塊回復 α/β/FA；audio 的 foil 一定是該子音表裡的；
產生的 block csv 欄位 = `practice.csv`、ΔE 在 ±0.5、H 的 count ≤ L 的；VAWM 格式的輸出 → 判對 ParallelOR。
PsychoPy 的 `VAWM_calibrate.py` 在這個容器裡只做過 `py_compile`。

## 順便修的 bug

`VAWM_nobox.py` 練習回饋：`rule.csv` 的 `ResponseBox` 欄是 Cedrus 按鍵碼 3/4，鍵盤版收到 `'y'`/`'n'`，查表永遠失敗，
練習階段每一試都顯示 Incorrect。已改成 `{'y': '4', 'n': '3'}` 對照。主實驗六個 block 不算 Acc（離線算），不受影響。
