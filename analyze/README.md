# analyze/ — 從 data/ 的 .csv / .json 算出筆記裡那些數字的程式

2026-11-24 整理。這些是事後分析，不是實驗的一部分：實驗程式不 import 這裡，這裡不進 run bundle、不上實驗機、不進公開的 VAWM_run。

## 怎麼跑

從 repo 根目錄：

```
python analyze/snr_pilot_summary.py                       # 預設 yyc1116、SNR 0 6 12
python analyze/snr_pilot_summary.py --participant EN01 --snrs 0 6 12
python analyze/main_task_summary.py data/EN01_s002_2026-11-24_08h49.32_simple_main.csv
python analyze/calibration_refit.py data/EN01_s001_2026-11-18_10h16.05_simple_calib_trials.csv --seed 42
python analyze/<name>.py --help                           # 每支都有
```

輸出（表、圖、重算的 json）預設寫到 `analyze/out/`（gitignore；`--out` 可改）。要放進筆記的圖另外複製到 `review/figures/`。

相依：numpy、scipy、pandas、matplotlib；adaptiveSFT repo 要找得到（跟實驗一樣：放在隔壁、或設 `ADAPTIVESFT_PATH`）。`stimuli_voice_check.py` 的母音共振峰另需 praat-parselmouth（沒裝會略過那一項）。
標「PyMC」的三支另外要 pymc + nutpie（`setup_pymc_coder.bat` 裝的那套）。中文字型 WenQuanYi Zen Hei；沒有的話圖上的中文會變方塊，數字不受影響。

## 每支程式

| 程式 | 讀 | 算什麼 | 對應的筆記 / 數字 |
|---|---|---|---|
| `snr_pilot_summary.py` | `data/<id>_sound_perception_dprime_snr<dB>.csv` | 每個 SNR 的無反應、誤報、前後半誤報 / 命中、六級命中率、logistic 的 50% / 90% 點 | `review/pilot_2026-11-18_snr_pilot.md`「結果」表、讀法 3、5；fig19、fig20 |
| `snr_pilot_pairs.py` | 同上 | 每個有序對的命中率、r(命中, d')、依特徵數 / 說話者的命中率、命中最低的對 | 同一份筆記讀法 1、2、4；fig21 |
| `native_category_false_alarms.py` | 同上 | 「華語有 / 沒有相近類別」兩組子音的誤報（× SNR、Fisher exact）與命中 | 2026-11-24 對話回報的數字（18/72 對 17/35、p = .027；86/109 對 77/101） |
| `consonant_acoustics.py` | `stimuli/T??_<子音>3.wav` | 子音段長度、80-350 Hz 能量、1.5-7 kHz 噪音強度、頻譜重心；成對頻譜圖 | 2026-11-24 對話的圖 S1-S4、N1-N3 |
| `sound_scale.py` | `stimuli/consonant_dprime.csv`、校準 trials、M&N 矩陣 | d' 表分布（幾對低於 0.8）、校準各級落在尺的哪裡、各 SNR 的尺 | DECISIONS 24「為什麼 (1)(2)」；`review/paper` fig17、fig18 |
| `calibration_summary.py` | `data/*_simple_calib_trials.csv` + `_simple_calib.json` | 各級 P(不一樣)、誤報、實驗當場記下的 H / L 與警告（不擬合） | `review/pilot_2026-11-24_EN01_s002.md` §2「舊版算出 / 資料」；DECISIONS 23、25 引用的比例 |
| `calibration_refit.py` **PyMC** | `data/*_simple_calib_trials.csv` | 用現行管線 `adaptive_vawm_EN.fit_points` 重算 L / H、兩個點的區間、模型對資料 | DECISIONS 25「結果」；s002 筆記 §2「新版重算」 |
| `calibration_link_comparison.py` **PyMC** | 同上 | 二次式 / 直線 / ogival × 有無 b0，加飽和指數：50% 點、90% 點、斜率跨零的比例 | DECISIONS 25「為什麼」1-3 |
| `calibration_seed_stability.py` **PyMC** | 同上（預設 EN01 s001 顏色） | 舊規則（二次式、反解目標、取平均）換 8 個取樣種子：平均與中位數 | DECISIONS 25「為什麼」4；「-994 怎麼來的」；`review/paper` fig10 |
| `main_task_summary.py` | `data/*_simple_main.csv` | 九個條件的正確率與 RT、AA 誤報逐 block、存活函數序（兩個方向的單尾 KS）、Houpt dominance、KS-SIC | s002 筆記 §3；DECISIONS 24「為什麼 (1)」；fig16、fig22、fig23 |
| `acoustic_scale_compute.py` | `stimuli/T??_<子音>3.wav`；噪音用 `sft_noise` | 每一對 × 每個 SNR（乾淨、+12、+6、0、-6 dB）：頻譜圖差、噪音底、扣掉噪音底後的「差別」、留一位說話者的樣板觀察者 d′ | `review/proposal_2026-11-24_language_independent_sound_scale.md` §2、§5；約 25 秒 |
| `acoustic_scale_validate.py` | 上一支的輸出、`stimuli/consonant_dprime.csv`、`data/<id>_sound_perception_dprime_snr<dB>.csv`、EN01 的校準 trials | 聲學尺對 M&N（r、ρ）；對非母語聽者逐題比三把尺（AUC + 拔靴區間、AIC、四等分命中）；對 EN01 乾淨條件 | 企劃書 §2 的 E2、E3、E5-E7、E10 |
| `acoustic_scale_use.py` | 前兩支的輸出 | 差別在時間上的位置、視窗敏感度、差別 → 命中的邏輯曲線、各 SNR 下 120 對的分布、差別最小 / 最大的對、與 M&N 最難 20 對的重疊 | 企劃書 §2 的分布表、E8、E12；§5 的換算 |
| `acoustic_scale_figs.py` | 前三支的輸出 | 七張圖 | `review/figures/fig24-fig29` |
| `acoustic_scale_design_check.py` | 前三支的輸出 | 驗證實驗建議的 40 對；模擬三種情境下各判準的值（每人 AUC、組間 / 折半 ρ、清濁對殘差） | 企劃書 §6 的參考值、附錄 A |
| `acoustic_scale_listener_curve.py` | `acoustic_scale_trials_<聽者>.csv`（validate 的輸出）、該聽者的 pilot csv | 換算曲線：x50、x90、bootstrap 區間、各 SNR 能否共用一條；與另一位聽者的四等分對照 | `review/pilot_2026-11-24_WEB01_snr_pilot.md` §4、`review/figures/fig31-fig32` |
| `acoustic_scale_criteria.py` | `acoustic_scale_trials_<聽者>.csv`、該聽者的 pilot csv；`--fit-points` 需要 PyMC | 企劃書 §6 的逐人判準 C1（排序力）、C2（加 SNR 的 AIC 改善）、C5（換算曲線與 `fit_points` 的 L / H 是否在範圍內；`--by-snr` 每場各跑一次） | `review/evaluation_2026-11-24_current_vs_proposed_sound_scale.md` §5 |
| `stimuli_voice_check.py` | `stimuli/T??_*3.wav`、`acoustic_scale_pairs.csv`；母音共振峰需要 praat-parselmouth | 12 個說話者編號兩兩的音量包絡相關、各編號的音高、母音 F1–F3、聲學尺上同一對在組內 / 組間的差異 | 同上 §3、`review/figures/fig34` |
| `evaluation_figs.py` | 上兩支的 json | 三把尺在兩位聽者上的排序力；說話者相關矩陣 | `review/figures/fig33-fig34` |
| `_common.py` | — | 路徑、字型、讀檔與過濾規則、不經 PyMC 載入 `sic.py`、用 ast 讀實驗的 `FIT_KW` | — |

聲學尺的五支要依序跑（compute → validate → use → figs / design_check），後面的讀前面寫在 `analyze/out/` 的檔。

## 共用的過濾規則（都在 `_common.py`）

- 校準：只取 `other == "T"` 的試次；擬合用「不同」試且 RT 在 0.15-3.0 s；誤報 = 「相同」試按 n 的比例。
- 主實驗：`block >= 1`（block 0 是練習）；RT 分析只取正確且 RT 在 0.15-3.0 s 的試次。
- 知覺測驗：沒按鍵的試次不進命中 / 誤報的分母。

## 程式字面算的 vs 解讀

程式只算比例、相關、擬合參數與檢定的 p。下面這些是解讀，不是程式的輸出：

- 「尺對非母語者沒有排序力」：程式給的是一位聽者 75 對的 r = .07。
- 「沒有類別就只能比原始聲音」：程式給的是事後分組的誤報差與未校正的 p；θ ð v 同時也是最弱的音，兩種解釋分不開。
- 「這批錄音的 80-350 Hz 差 9-18 dB」是量測；「母語者用這條線索」是文獻，不是這裡量的。
- `calibration_refit.py` 的 L / H 是「現行版對舊資料會給什麼」；舊資料的層級沒有涵蓋轉彎處，區間很寬。
- 聲學尺（`acoustic_scale_*`）量的是「聲音裡有多少差別」，不是「聽者用了多少」；排序力 .74 來自一位非母語聽者的事後比較。「難 / 易」的界線是用同一位聽者、同一批資料換算的，只是示意。`acoustic_scale_design_check.py` 的數字是模擬。

## 驗證（2026-11-24，全部用 `/usr/bin/python3` 實跑；PyMC 三支都是完整取樣 1000 × 4 鏈）

| 程式 | 對到的數字 |
|---|---|
| `snr_pilot_summary.py` | 誤報 .47 / .28 / .23；50% 點 0.19 / 0.36 / 0.57；90% 點 5.54 / 4.59 / 2.42；+12 dB 斜率 1.19；無反應 4 / 0 / 3 ✓ |
| `snr_pilot_pairs.py` | 75 對、r = .07；特徵數 .70 / .79 / .92 / .94；T03 .57、T09 .55、T01 .92、T12 .90；ð-g 1/6、θ-k 1/4、d-b 2/4 ✓ |
| `native_category_false_alarms.py` | 18/72 對 17/35、Fisher p = .027；逐 SNR .35/.69、.17/.46、.23/.22；命中 86/109 對 77/101；f-v 類 1/3 ✓ |
| `consonant_acoustics.py` | 80-350 Hz 清濁差 9 / 18 / 13 / 9 dB；θ 重心 3.47 kHz、f 3.16 kHz；z 比 s 弱 6 dB ✓ |
| `sound_scale.py` | 240 對中 14 對 d' < 0.8 ✓ |
| `calibration_summary.py` | s002 顏色 .40 .20 .20 .75 1.00 1.00、誤報 .10、當場 L 62.8 / H 90；聲音 1.5 級 .6、L 1.13 ✓；s001 當場 H = L = 2 ✓ |
| `calibration_refit.py`（seed 42） | EN01 s001 顏色 L 25.2 [15.4, 41.8]、H 36.3 [18.3, 80.9]；yyc1110 L 15.2 [8.6, 21.2]、H 26.9 [18.2, 47.7]；s002 顏色 L 18.7 [10.7, 28.9]、H 41.6 [21.7, 117]；聲音 50% 點 -1.07 / -0.77 / -2.47 ✓ |
| `calibration_link_comparison.py`（seed 15） | 50% 點 EN01 二次式 23.3、飽和指數 26.8；yyc1110 12.8 / 12.9 / 15.5；沒有 b0 的二次式 P(α<0) = .47 ✓（ogival 見下） |
| `calibration_seed_stability.py` | seed 15 的 H 平均 -993.9；8 個種子中位數 422-493、平均在 seed 12 / 15 翻成負的 ✓ |
| `main_task_summary.py` | s002 正確率表、AA 誤報 .21（block 1 .36 → block 6 .12）、Houpt p .13-.98、SIC D+ .085 p = .74 / D- .134 p = .47；s001 正確率 .92 / .90 ✓ |
| `acoustic_scale_compute.py` | 與整理前的版本逐格相同（600 列 × 36 欄，最大差 0）；各 SNR 的平均差別 10.09 / 5.95 / 4.35 / 2.89 / 1.66 dB ✓ |
| `acoustic_scale_validate.py` | yyc1116 210 題、漏 47；排序力 M&N .56 [.48, .64]、特徵數 .63 [.55, .70]、聲學尺 .74 [.67, .81]、樣板 d′ .69；AIC 221.8 / 215.2 / 201.5；四等分 .53 / .77 / .87 / .94；對 M&N 表 r = .21（+6 dB）；EN01 71 題中 67 題可反推、漏 5 ✓ |
| `acoustic_scale_use.py` | 難 / 中 / 易：乾淨 0/1/119、+12 12/43/65、+6 51/27/42、0 67/40/13、-6 95/25/0；.75 在 3.1 dB、.90 在 5.3 dB；最難 20 對重疊 4；視窗敏感度 .73-.75（只看母音開頭 100 ms：.56）✓ |
| `acoustic_scale_figs.py` | +6 dB 時 0.4 kHz 以下剩 17%、3 kHz 以上剩 62%；d 對 z（T01）12.0 → 4.3 dB ✓ |
| `acoustic_scale_design_check.py`（500 次，seed 20261124） | 兩組相同：組間 ρ .90 [.86, .94]、清濁殘差 0.01 [-0.31, 0.35]；一組清濁對差 2 logit：ρ .81 [.74, .86]、殘差 -2.00 [-2.31, -1.71]（模擬，不是資料） |

## 與筆記對不上的地方（重跑時發現的）

第 1-3 點已在 2026-11-24 更正進筆記（`review/pilot_2026-11-24_EN01_s002.md` §3、DECISIONS 24、`review/pilot_2026-11-18_snr_pilot.md` 讀法 2、論文附錄的 Selective-influence 段與 fig16、`review/figures/fig22-23`），原文旁都留了更正註記。第 4-5 點是定義或先驗不同，筆記維持原樣。下面保留當初的差異紀錄。

1. **存活函數序的方向標反了。** `review/pilot_2026-11-24_EN01_s002.md` §3 寫「聲音：S_H < S_L 預期方向 p = .44」「顏色：預期方向 p = .42」，DECISIONS 24 寫 EN01 s001「KS p = .55」。這三個數字是 `_ks_one_sided(H, L, "less")`，它的統計量是 max(F_L − F_H)，檢的是「L 較快」（違反的方向），不是預期方向。預期方向（H 較快，`"greater"`）是：

   | | H 較快（預期） | L 較快（違反） |
   |---|---|---|
   | s001 聲音，雙變格 | p = .39 | p = .55 |
   | s001 顏色，雙變格 | p = .19 | p = .85 |
   | s002 聲音，雙變格 | p = **.074** | p = .44 |
   | s002 聲音，單變 HA vs LA | p = **.011**（中位數 .712 vs .772） | p = .99 |
   | s002 顏色，雙變格 | p = .95 | p = .42 |
   | s002 顏色，單變 AH vs AL | p = .59 | p = .55 |

   s001 的結論不變（兩個方向都不顯著）。s002 的聲音要改寫：雙變格接近但沒過 .05，單變條件 H 明顯較快（未校正，這張表有六個預期方向的檢定）。s002 顏色的結論不變。Houpt 八個檢定的數字本身沒錯，但「前四個不該顯著」的說法也是反的：前四列是「較快那一格的 CDF 在上」，通過時應該顯著（`adaptivesft.sic.classify` 的判法）。
2. **SNR pilot 筆記讀法 2 的「m-n / n-m 三個 SNR 合計 2/12」** 對不上資料：n-m 命中 3/11、m-n 3/5，合計 6/16。它仍是命中最低的幾對之一。
3. **s002 筆記 §3 標題的「block 1-6，1374 / 1376 有反應」** 含 32 個練習試；正式 block 是 1344 / 1344。
4. **DECISIONS 25「為什麼」第 2 點的 ogival 50% 點（EN01 27、yyc1110 16）** 是探索時用另一組先驗算的；用定案管線的先驗是 24.7 / 15.5（`calibration_refit.py` 給 25.2 / 15.2，差在 seed）。
5. 定義不同、不是錯：筆記裡各條件的 RT 中位數是「所有有反應的試次」（表裡的 `rt_median_all`），SNR pilot 表的 RT 中位數是「不同」試的；s002 筆記的發散數 10 / 24 是 `target_accept` 還是 0.9 時的，現在 0.95 是 2 / 10。

## 沒有收進來的

session 裡還有一批純示意、不讀受試者資料的畫圖腳本（模擬為什麼地板不是 50%、層級怎麼選、連結函數形狀），以及被後來版本取代的中間版本（refit / refit2 / refit3、bias、nob0、ppinv、linear、cmp_links、satexp —— 內容已併進上面三支 PyMC 程式）。論文附錄用的圖（`review/paper/figures/fig1-18`）有幾張是那些示意腳本畫的，這裡不重現。
