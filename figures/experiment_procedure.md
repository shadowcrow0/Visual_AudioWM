# Figure 1. 實驗流程 / Experimental procedure

圖檔：
- `experiment_procedure_en.svg` / `.png`：英文、斜階梯版式、APA 7 §7.26 字型（圖內 sans serif 8–14 pt：SVG 裡字型名為 Arial，容器用同字寬的 Liberation Sans 排版）；B 面板以顏色校準試次為例（聲音與學習項目相同，只變顏色）；D 面板是校準迴圈（72 試定值刺激 → lnrm2 擬合 → 反解 H/L）、72 試的 ΔE 層級序列與擬合曲線示意；只有標籤與時間，沒有說明文字。重畫：`python figures/make_procedure_figure_en.py`
- `experiment_procedure_zh.svg` / `.png`：中文版，照 GRTv3_ada 流程圖的版式（A 整場流程、B 單一試次斜階梯螢幕序列、C 時間軸）。重畫：`python figures/make_procedure_figure_zh.py`
- `experiment_procedure.svg` / `.png`：英文版（A 整場流程、B 試次、C 探測類型與 2×2 設計）。重畫：`python figures/make_procedure_figure.py`
所有數字來自 `VAWM_nobox.py` 與 `VAWM_calibrate.py`。

## 圖說（中文）

**圖 1.** 實驗流程。**(A)** 單一場次的結構。受試者先完成兩個校準區塊（各 40 試：6 個強度層級 × 5 試「不同」探測，加 10 試
「相同」探測；每試後給回饋；另有 4 試暖機與 8 試填充，不進擬合）。每個區塊結束時以 lnrm2 模型擬合正確率與反應時間（20–40 s），反解出該受試者的
高 / 低顯著度（H / L）層級：顏色為 CIELAB 距離 ΔE_H / ΔE_L，聲音為子音混淆次數 count_H / count_L。接著是練習
（2 組 study × 16 個探測，回饋 1 s）與主作業（6 個 block × 14 組 study × 16 個探測 = 1344 個探測，無回饋，block 之間
自行決定休息長度）。**(B)** 單一試次。Study 階段（4.3 s）：注視點 300 ms，左側顏色 1000 ms，右側顏色 1000 ms，
子音 1 1000 ms，子音 2 1000 ms。每組 study 之後連續出現 16 個探測（各 3.3 s）：注視點 300 ms，顏色方塊與子音同時
出現 1000 ms，作答窗從探測出現後 500 ms 開始、持續 2000 ms（y = 與記住的相同，n = 不同）。練習與校準試次另有回饋
畫面（1000 ms；校準為 800 ms）。校準試次的作答窗最長 3 s。**(C)** 每組 16 個探測的組成（隨機排列）：AA（顏色與聲音
都與某個 study 項目相同）4 個；HA / LA（只有顏色不同，H = 容易、L = 困難）各 2 個；AH / AL（只有聲音不同）各 2 個；
HH / HL / LH / LL（兩者都不同）各 1 個。即 Houpt et al. (2014) Table 3 的 16 格各 1/16，把 target（相同）併成一個水準：
4/16、2/16 × 4、1/16 × 4。最後四類構成
2 × 2 雙因子設計（double-factorial paradigm），以其「不同」反應的反應時間計算 SIC 與 MIC，以推論顏色與聲音的處理架構
（作業為「兩者皆相同才答 y」，即 AND 作業；處理架構與停止規則由 SIC 的形狀判定，不預設）。
每場次共 336 個雙因子探測，每格 84 個。

## Caption (English)

**Figure 1.** Experimental procedure. **(A)** Structure of one session. Participants first complete two calibration
blocks (40 trials each: 6 intensity levels × 5 "different" probes plus 10 "same" probes; feedback after every
trial; 4 warm-up and 8 filler trials per block are not fitted). At the end of each block the lnrm2 model is fitted to accuracy and response time (20–40 s) and inverted to
obtain that participant's high- and low-salience levels (H / L): for colour the CIELAB distances ΔE_H / ΔE_L, for
sound the consonant-confusion counts count_H / count_L. Practice (2 study sets × 16 probes, 1 s feedback) and the
main task (6 blocks × 14 study sets × 16 probes = 1344 probes, no feedback, self-paced rest between blocks) follow.
**(B)** One trial. Study (4.3 s): fixation 300 ms, colour 1 on the left 1000 ms, colour 2 on the right 1000 ms,
consonant 1 1000 ms, consonant 2 1000 ms. Each study set is followed by sixteen probes (3.3 s each): fixation 300 ms,
a colour patch and a consonant presented together for 1000 ms, and a response window opening 500 ms after probe
onset and lasting 2000 ms (y = same as studied, n = different). Practice and calibration trials add a feedback
screen (1000 ms; 800 ms in calibration), and calibration trials allow up to 3 s to respond. **(C)** Composition of
the sixteen probes per study set (shuffled): AA (colour and sound both match a studied item) × 4; HA / LA (only
colour differs; H = easy, L = hard) × 2 each; AH / AL (only sound differs) × 2 each; HH / HL / LH / LL (both differ)
× 1 each. This is Table 3 of Houpt et al. (2014), 16 cells at 1/16 each, with the target (same) collapsed to one
level: 4/16, 2/16 × 4, 1/16 × 4. The last four types form
the 2 × 2 double-factorial paradigm; the response times of their "different" responses yield the SIC and MIC used to
infer the processing architecture of colour and sound (the task is "y only if both match", an AND task; the
architecture and stopping rule are inferred from the shape of the SIC, not assumed); 336 double-factorial probes per session, 84 per cell.

## 要注意的一件事 / One caveat

每格 84 個探測，低於 `adaptiveSFT/results/power_scan.csv` 建議的 250（Table 3 的配比只有 1/4 的探測進雙因子格，
這是換取「相同」佔 25%、反應偏誤較小的代價）。要提高 SIC 的檢定力，增加 block 數或每 block 的組數
（`N_BLOCKS` / `SETS_PER_BLOCK`），或跑多個場次：250 / 格約需 4000 個探測。

84 probes per DFP cell is below the 250 per cell recommended by `adaptiveSFT/results/power_scan.csv` (with Table 3
rates only a quarter of the probes fall in the double-factorial cells; the price of keeping "same" at 25 % and the
response bias small). Add blocks or sets per block (`N_BLOCKS` / `SETS_PER_BLOCK`), or run several sessions:
250 per cell needs about 4,000 probes.

## APA 7 格式的圖說（給 `experiment_procedure_en`）

APA 7（§7.22–7.36）的規定：圖的四個部分是 **圖號**（Figure 1，粗體）、**標題**（斜體，另起一行）、**圖本體**、**註**（Note.，在圖下方）。
說明性的文字放在 **Note**，不放在圖裡；圖裡只放標籤（label）——能辨識元件的最短文字：名稱、時間、單位、軸名。
圖例（legend）只用來解釋符號，放在圖的邊框內。另外 §7.26：**圖本體內的字要用 sans serif（Arial、Calibri 等），8–14 pt**；
圖號、標題與 Note 用內文字型（Times New Roman 12 pt）。`experiment_procedure_en` 已照此設定（圖內 Arial 8–14 pt）；圖號、標題、Note 貼進稿件時用內文字型。

以下可直接貼進稿件（圖號與標題在圖的上方，Note 在下方）：

**Figure 1**

*Session Structure, Trial Sequence, and Timeline of the Audiovisual Working-Memory Task*

[圖]

*Note.* Panel A shows the order of phases in one session. In the two calibration parts, each trial varied only the
dimension being calibrated (color in Part 1, sound in Part 2); the other dimension was identical to the studied
item. Each calibration part comprised 40 fitted trials: 6 intensity levels × 5 "different" probes plus 10 "same" probes,
with feedback after every trial. At the end of each part the lognormal race model (lnrm2) was fitted to accuracy and
response time and inverted to obtain the participant's high- and low-salience levels (color: CIELAB ΔE; sound:
consonant-confusion count). Practice (2 study sets × 16 probes, feedback 1 s) and the main task (6 blocks × 14 study
sets × 16 probes = 1,344 probes, no feedback, self-paced rest between blocks) followed. Panel B shows one
color-calibration trial. Study: fixation 300 ms, color 1 on the left for 1,000 ms, color 2 on the right for
1,000 ms, the syllable [bi] for 1,000 ms, the syllable [pi] for 1,000 ms (screen blank during sounds). Probe: fixation 300 ms,
then a color patch and a syllable presented together for 1,000 ms; here the color is color 1 shifted by x ΔE
and the syllable is [bi], unchanged. The response window opened 500 ms after probe onset and lasted 2,000 ms
(up to 3,000 ms in calibration); participants pressed y for "same as studied" or n for "different." Feedback
(800 ms in calibration, 1,000 ms in practice) was shown only in calibration and practice. In the main task the
probe sequence (fixation, probe, response) repeated sixteen times after each study set, with one of nine probe types
on each repetition. Panel C shows the same trial on a single time axis. Panel D shows the calibration loop for Part 1: the 72 trials present six ΔE levels (2, 6, 12, 20, 30, 45) eight times each plus 24 same probes in shuffled order (middle, schematic); after the last trial the lnrm2 model is fitted to accuracy and response time, and the drift-separation curve 2·d(ΔE) is inverted at the targets H = 2.0 and L = 0.5 to give ΔE_H and ΔE_L (right, schematic observer), which are then used in practice and the main task. The calibration trial is identical to the trial in Panel B; only the probe color varies, and the sounds are the studied sounds on every trial. Part 2 follows the same loop with the consonant-confusion level in place of ΔE and the colors unchanged.

