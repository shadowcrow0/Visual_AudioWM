# archive/legacy_vawm_nobox — 舊的 SFT 流程（凍結）

`VAWM_nobox.py`（Builder 輸出，7462 行）+ `VAWM_calibrate(_EN).py` + `make_adaptive_blocks(_EN).py`（校準 → block csv → 主實驗）
與 `make_simple_psyexp.py` / `VAWM_simple.psyexp`（Builder 版的單檔實驗）、`GRTv1/GRTv2.psyexp`（GRTv3 系列的 Builder 來源，不再重生）、
`generate_trials.py`（舊聲音軸的混淆次數表 `CONFUSION_DATA`；`consonant_distance.py` 的 `confusion` 指標仍從這裡讀）。

2026-11-17 移入。現行的 SFT 實驗是根目錄的 `SFT_WM_nochange_EN.py`（單檔，校準 → 練習 → 主實驗），不依賴這裡任何東西。
`adaptive_vawm_EN.py` 裡的 `write_blocks` / `CSV_COLUMNS` 與 `stimuli/practice.csv` 是為這條舊流程留的，測試仍覆蓋，暫不移除。
