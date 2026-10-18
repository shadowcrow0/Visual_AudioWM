# archive/ —— 不再執行、但決策文件引用過的歷史檔（2026-10-19 整理）

搬進來的理由分三類，對應 `review/` 與 `DECISIONS.md` 的脈絡：

- **色彩探索叢集**（`colorpool.py`、`generate_colors*.py`、`generate_bv_*.py`、`color_tools.py`、`lc_*.py`、`psychopy_pickcolor.py` 與 `data/`、`figure/`、`stimuli/` 裡它們的輸出）：
  `review/決策脈絡_顏色維度.md` 引用。結論已進 `agrt_setup.py` / `agrt_colour_lut.json`。
- **聽覺維度嘗試叢集**（`GetAudioStim.py`、`generate_b3_p3_auditory_baseline.py`、`adaptive_duration_staircase.py`、
  `generate_speech_stimulus.py`、`psychopy_*_adaptive.py`、`psychopy_integration_example.py`、相關 csv）：
  `review/聽覺維度_嘗試與放棄紀錄.md` 記錄的對象。
- **舊版本**：`VAWM.psyexp`（`VAWM_nobox.py` 的 Builder 原始檔，也是 `make_simple_psyexp.py` 的底稿）、
  `GRTv3_a.py`（CLAUDE.md：GRTv3_ada 的前身）與它用的 `stimuli/box_keys.png`、六張舊 pilot 分析圖。

同一次整理直接刪掉的（零引用或逐 byte 重複，git 歷史裡找得回來）：十支舊 Builder 編譯檔、測試殘渣、
根目錄與 `data/`/`figure/` 重複的 csv/png、誤入版控的 `.pyc`、孤兒音檔。
