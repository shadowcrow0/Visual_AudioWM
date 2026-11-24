# 線上「聲音校準前測」試作（jsPsych 7，Pavlovia）

子音同 / 異判斷，**單一 SNR（+6 dB）**，216 題不同 + 108 題相同，題目依「頻譜圖上的差別」均勻鋪開。
目的：看一位受試者做一場，能不能定出他自己的 L（曲線中點附近）與 H（到頂點）。
這是提案的試作（`review/evaluation_2026-11-24_current_vs_proposed_sound_scale.md` §7 第 1 步），不是實驗的一部分；實驗主程式沒有因它改動。

## 與 `online/snr_pilot/` 的差別

| | `snr_pilot/`（WEB01 用的，保持原樣） | `calib_pilot/`（這裡） |
|---|---|---|
| SNR | 0 / +6 / +12 dB 三段，各 108 題 | 只有 +6 dB，324 題（`online/make_calib_pilot_stimuli.py:41-42`） |
| 題目的軸 | d′ 表的六個層級 | 頻譜圖差別，每 1 dB 一格、0.5–8.5 dB 共 8 格（`:43`） |
| 題目的單位 | 子音對，說話者隨機 | 子音對 × 說話者；x 用該說話者兩個音檔的值 |
| 只差清濁的對子 | 碰巧抽到（WEB01 只有 f–v 3 題） | 7 對全放，每對 6 題（`:52`、`:96-102`） |
| 說話者 | 12 個編號隨機 | 三組聲音之間平衡，組別記進資料（`:49`） |
| 題目順序 | 三段各自洗牌，種子沒記 | 整場洗一次，`order_seed` 記進資料，可重建（`experiment.js:21`、`:27`） |
| 開頭 | 只有網址帶的受試者編號 | 不問任何問題。編號來自網址；沒帶或是 `test` 這類佔位字，就用當下的日期時間產生一個（`experiment.js:39`）。母語等背景由研究者另外記錄 |
| 噪音種子 | 數字（被瀏覽器四捨五入過） | 字串（`make_calib_pilot_stimuli.py:151`、`experiment.js:137`） |

試次的時間結構、按鍵（Y 相同 / N 不同）、回饋、耳機檢查都沿用。

## 題目組成（程式產生的實際結果，`design.csv`、`design.png`）

「不同」216 題。x = 該說話者在 +6 dB 的頻譜圖差別（`analyze/acoustic_scale_compute.py` 的 `ex_T??` 欄），範圍 0.83–8.37 dB，112 個不同的子音對。

| x (dB) | 題數 | 其中清濁對 | 不同的對 | 塞–塞 | 塞–擦 | 塞–鼻 | 擦–擦 | 擦–鼻 | 鼻–鼻 | G1 / G2 / G3 | 最常出現的子音 | 含 ʃ 的比例 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5–1.5 | 27 | 6 | 14 | 9 | 9 | 3 | 3 | 3 | 0 | 15 / 7 / 5 | b 13、v 12、k 7 | 0 |
| 1.5–2.5 | 27 | 18 | 13 | 6 | 4 | 2 | 12 | 2 | 1 | 9 / 9 / 9 | ð 9、f 8、θ 7 | 0 |
| 2.5–3.5 | 27 | 2 | 25 | 5 | 5 | 5 | 5 | 5 | 2 | 9 / 9 / 9 | n 8、m 6、f 6 | 0 |
| 3.5–4.5 | 27 | 4 | 22 | 4 | 6 | 5 | 7 | 4 | 1 | 9 / 8 / 10 | z 10、n 8、d 6 | .15 |
| 4.5–5.5 | 27 | 10 | 19 | 6 | 6 | 4 | 7 | 4 | 0 | 9 / 9 / 9 | t 11、z 9、ʒ 7 | .07 |
| 5.5–6.5 | 27 | 2 | 23 | 6 | 7 | 4 | 6 | 4 | 0 | 10 / 9 / 8 | t 13、ʒ 10、n 4 | .04 |
| 6.5–7.5 | 27 | 0 | 24 | 0 | 9 | 1 | 9 | 8 | 0 | 9 / 9 / 9 | s 9、ʃ 9、ʒ 7 | .33 |
| 7.5–8.5 | 27 | 0 | 20 | 0 | 10 | 0 | 9 | 8 | 0 | 10 / 8 / 9 | ʃ 16、s 11、n 4 | .59 |

![題目組成](design.png)

- **均勻鋪開**：8 格各 27 題。清濁對先放進去（落在哪一格由它們的 x 決定），其餘用「補到一樣高」的方式填（`make_calib_pilot_stimuli.py:104-129`）。
- **清濁對**（每對 6 題 = 三組聲音 × 兩個方向，共 42 題）：b–p 1.33–1.75、g–k 1.08–2.28、f–v 1.87–2.32、θ–ð 2.01–2.46、s–z 3.31–5.30、ʃ–ʒ 3.68–4.57、d–t 4.60–6.50 dB。d′ 表給它們的值是 2.88–4.58（表上算容易）。
- **三組聲音**：G1（T01–T06）80 題、G2（T07–T09）68 題、G3（T10–T12）68 題。最低一格 G1 偏多（15 / 7 / 5），因為落在那一格的音檔多半是 G1 的。
- **子音類別分散到什麼程度**（做不到的地方）：
  - 6.5 dB 以上沒有塞–塞的對，也幾乎沒有塞–鼻：材料裡差別這麼大的對一定含擦音。最高一格 59% 的題目含 /ʃ/。
  - 最低一格以 b、v、k 為主（27 題裡 b 出現 13 次、v 12 次）。
  - 1.5–2.5 dB 那格擦–擦偏多（12 題），因為清濁對有 18 題落在這裡，其中 f–v、θ–ð 都是擦音。
  - 所以「x 大 = 含擦音、x 小 = 含 b / v / k」在兩端仍然成立；中間五格（2.5–6.5 dB）六種類別大致平均。這是材料的限制，抽樣只能緩和、不能消除。
- 「相同」108 題：16 個子音 × 三組聲音 × 2 = 96 題，另 12 題分給 12 個不同的子音、每組聲音 4 題（`:131-138`）。兩個音節是同一個音檔配不同的噪音。

## 檔案

```
online/calib_pilot/                 上傳這個資料夾裡的 15 個檔（每個 < 5 MB）
   index.html、experiment.js        實驗本體
   conditions.json                  固定的 324 題：對子、說話者、組別、x、種子、檔名
   js/（5 個檔）                    jsPsych 與三個 plugin、css（與 snr_pilot 相同的檔）
   bundles/（7 個檔）               聲音：stim_hp.json（耳機檢查）、stim_p1..p6.json（各 54 題的兩個音節，約 4 MB）
   design.csv、design.png           題目組成（不用上傳）
   stim/                            產生 bundles 用的 666 個 wav（不用上傳；不進 git）
online/make_calib_pilot_stimuli.py  產生 stim/、conditions.json、design.*（固定的 PLAN_SEED，結果可重現）
online/pack_stimuli.py calib_pilot  stim/ → bundles/
online/pavlovia_to_perception_csv.py   資料檔 → data/<id>_sound_perception_dprime_snr6.csv + .json
online/calib_pilot_browser_test.js  用無頭瀏覽器把整場跑一遍的測試程式（不用上傳）
```

重新產生（結果應與 repo 裡的相同）：

```bash
python analyze/acoustic_scale_compute.py          # 產生 analyze/out/acoustic_scale_pairs.csv（約 25 秒）
python online/make_calib_pilot_stimuli.py
python online/pack_stimuli.py calib_pilot
```

## 放上 Pavlovia

1. pavlovia.org → Dashboard → **Experiments** → New：選 *jsPsych*，取名（例如 `calib_pilot`）。Pavlovia 會建一個 GitLab repo。
2. 在 gitlab.pavlovia.org 打開那個 repo，用網頁上傳（`+` → *Upload file*）：
   - 根目錄 3 個：`index.html`、`experiment.js`、`conditions.json`
   - `+` → *New directory* 建 `js`，上傳 5 個：`jspsych.js`、`jspsych.css`、`plugin-html-keyboard-response.js`、`plugin-audio-keyboard-response.js`、`plugin-preload.js`
   - 建 `bundles`，上傳 7 個：`stim_hp.json`、`stim_p1.json` … `stim_p6.json`
   - 路徑必須正好是 `js/...` 和 `bundles/...`。**不要建 `lib/`**：Pavlovia 保留這個名字，有它就會拒絕同步。
3. 實驗狀態設成 **Piloting** 可以免費試跑；**Running** 才能正式收（要點數或授權）。
4. 給受試者的連結：`https://run.pavlovia.org/<帳號>/<實驗名>/?participant=CAL01`
   - 編號只從 `participant=` 來，頁面不會問受試者任何問題。不帶、或帶的是 `test`、`anon` 這類佔位字（`experiment.js:23`），頁面會自己產生 `P年月日_時分秒` 當編號，資料的 `id_source` 欄寫 `generated`。
   - **母語等背景頁面不記錄**，請在發連結時自己記下「編號 ↔ 是誰、母語是什麼」。
   - 測試用：`&skipcheck=1` 跳過耳機檢查；`&maxtrials=12` 只跑前 12 題；`&order_seed=數字` 固定題目順序。
5. 資料：Dashboard → 該實驗 → Download results，每位受試者一個 CSV。

Pavlovia 的橋接程式沿用 `snr_pilot` 的寫法（`index.html` 載入 `https://lib.pavlovia.org/jspsych-7-pavlovia-2022.1.1.js`）；WEB01 那一場用它存到了資料。

## 轉檔與分析

```bash
python online/pavlovia_to_perception_csv.py <下載的檔>.csv              # 寫 data/<id>_sound_perception_dprime_snr6.csv 與 .json
python analyze/acoustic_scale_validate.py --participant <id> --snrs 6    # 三把尺的排序力；寫 analyze/out/acoustic_scale_trials_<id>.csv
python analyze/acoustic_scale_listener_curve.py --participant <id>       # 換算曲線：x50、x90、區間
python analyze/acoustic_scale_criteria.py --participants <id> --fit-points   # C1、C5（含實驗自己的校準擬合，需要 PyMC）
```

- 檔名沿用實驗室版的樣式，所以 `analyze/` 的程式直接讀得到；`axis` 欄寫 `acoustic`，`x` / `count` 欄放的是該對在 d′ 表上的值（給「現在的尺」那一欄的比較用），聲學差別在新增的 `x_acoustic` 欄。
- json 會保留 `order_seed`、耳機檢查結果（`pavlovia_to_perception_csv.py:102-108`）。母語等欄位頁面不再收集，json 裡會是空的。
- **同一個人做過 `snr_pilot` 的 +6 dB 再做這個，檔名會撞**。轉檔程式遇到「檔案已存在而內容不同」會停下來（`:44`），請用 `--id 新編號` 存成另一個編號（例如 `WEB01C`）。
- 資料檔若帶的是佔位字編號，轉檔會拒絕，要用 `--id` 指定（`:64`）。
- 單一 SNR 的場次，判準 C2（加 SNR 有沒有幫助）不適用，程式會印「不適用」（`analyze/acoustic_scale_criteria.py:45`）；`acoustic_scale_validate.py` 的迴歸不放 SNR（`:139`）。這三支程式對 WEB01、yyc1116 印出的結果改動前後逐字相同（json 的數值相同）。

## 驗證

**瀏覽器實跑**（`node online/calib_pilot_browser_test.js`；無頭 Chromium，本機靜態伺服器，沒有 Pavlovia）：

- 耳機檢查 6 題跑完並通過。（2026-11-24 拿掉開頭表單後重測：網址帶 `test` 時頁面改用自動產生的編號、頁面上沒有任何表單；含耳機檢查跑 14 題通過。整場 324 題的實跑是拿掉表單之前做的，題目迴圈沒有改。）
- 324 題全部跑到，題目清單的每一列各出現一次；216 不同 + 108 相同。
- 資料檔欄位齊全；種子是 17–19 位數的完整字串，324 題與題目清單逐位相同。
- 題目順序可以從記下的 `order_seed` 重建，與資料一致。
- 計分正確：測試程式每 7 題故意答錯、每 50 題不作答，記錄的對錯與預期 324 / 324 相符。
- 轉檔後 324 列、26 欄，json 帶著問卷與 `order_seed`。

**模擬聽者**（`analyze/calib_pilot_design_check.py`；**模擬，不是資料**。聽者 = WEB01 的換算曲線：x50 1.98、s 0.57、λ .017、誤報 .08，真值 x90 = 3.23 dB）：

| | 估回來的中位數 | 中間 90% | 寬度 |
|---|---|---|---|
| 曲線中點 x50（200 場） | 1.98 dB | 1.68–2.26 | 0.58 dB |
| 到頂九成 x90（200 場） | 3.15 dB | 2.63–3.77 | 1.14 dB |
| 逐題排序力 AUC（200 場） | .90 | .85–.94 | |

- x90 落在真值 ±1 dB 內的比例 .99。x50 到 x90 的距離是 1.25 dB，x90 的不確定範圍（1.14 dB）跟它差不多寬：L 和 H 分得開，但不寬裕。
- 三場模擬資料走完整條管線（轉檔 → validate → listener_curve → criteria `--fit-points`）：實驗自己的校準擬合三場都在範圍內、0 發散。

  | 場 | L（50% 點）[90% 區間] | H（到頂點）[90% 區間] | in_range | 發散 |
  |---|---|---|---|---|
  | 1 | 1.70 [1.40, 1.94] | 4.14 [2.81, 5.53] | 是 | 0 |
  | 2 | 1.67 [1.45, 1.87] | 3.53 [2.02, 4.82] | 是 | 0 |
  | 3 | 1.59 [1.17, 1.88] | 4.14 [2.79, 6.17] | 是 | 0 |

  `fit_points` 的 x 傳入前除以 2 再乘回來，這是 `analyze/acoustic_scale_criteria.py` 的做法，不是實驗的設定。它的 H 定義（ogival 到頂九成）與換算曲線的 x90 不是同一個量，所以數值不同。

**時間**：WEB01 做 `snr_pilot` 的 324 題含說明與耳機檢查是 19.7 分鐘（實測）；這裡題數相同，多一次休息，估計約 20 分鐘。瀏覽器測試（每題 0.3 秒就作答、不讀說明）跑完是 14.9 分鐘，這是下限。

**沒有驗證到的**：

- **Pavlovia 上傳與存檔**這一段在這裡測不了；頁面在沒有 Pavlovia 時走的是「結束後提供下載」的分支。
- 真人的作答速度、頁面在手機或其他瀏覽器上的樣子、實際耳機上的音量。
- 模擬假設聽者就是那條曲線、每題獨立、沒有練習效果。斜率更平或誤報更高的聽者（例如 yyc1116：誤報 .23–.47、x90 約 5.7 dB）區間會更寬；可以用 `--x50 --s --lam --fa` 重跑。
- 題目的 x 來自 `analyze/acoustic_scale_compute.py`（每個音檔 6 個噪音種子的平均），不是每一題實際播放那兩段噪音的值。先前的檢查顯示兩者相關 1.00（那份檢查還沒進 repo）。

## 線上版控制得到與控制不到的

- SNR 在檔案裡（語音和噪音混好了），不受播放音量影響。絕對音量控制不到；耳機檢查（Woods et al., 2017；6 題，答對 5 題以上才通過）擋掉喇叭。
- 瀏覽器的 RT 精度約 ±10–20 ms。這個前測以對錯為主；`fit_points` 會用到 RT，線上的 RT 能不能直接拿來校準還沒有比較過。
- 題目清單固定（`PLAN_SEED`），每位受試者只有順序不同。
