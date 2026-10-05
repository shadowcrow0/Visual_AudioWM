# Figure 1. 實驗流程 / Experimental procedure

圖檔：
- `experiment_procedure_en.svg` / `.png`：英文、斜階梯版式、Times New Roman（SVG 裡字型名為 Times New Roman，容器用同字寬的 Liberation Serif 排版）；B 面板以顏色校準試次為例（聲音與學習項目相同，只變顏色）；只有標籤與時間，沒有說明文字。重畫：`python figures/make_procedure_figure_en.py`
- `experiment_procedure_zh.svg` / `.png`：中文版，照 GRTv3_ada 流程圖的版式（A 整場流程、B 單一試次斜階梯螢幕序列、C 時間軸）。重畫：`python figures/make_procedure_figure_zh.py`
- `experiment_procedure.svg` / `.png`：英文版（A 整場流程、B 試次、C 探測類型與 2×2 設計）。重畫：`python figures/make_procedure_figure.py`
所有數字來自 `VAWM_nobox.py` 與 `VAWM_calibrate.py`。

## 圖說（中文）

**圖 1.** 實驗流程。**(A)** 單一場次的結構。受試者先完成兩個校準區塊（各 72 試：6 個強度層級 × 8 試「不同」探測，加 24 試
「相同」探測；每試後給回饋）。每個區塊結束時以 lnrm2 模型擬合正確率與反應時間（20–40 s），反解出該受試者的
高 / 低顯著度（H / L）層級：顏色為 CIELAB 距離 ΔE_H / ΔE_L，聲音為子音混淆次數 count_H / count_L。接著是練習
（5 組 study × 9 個探測，回饋 1 s）與主作業（6 個 block × 24 組 study × 9 個探測 = 1296 個探測，無回饋，block 之間
自行決定休息長度）。**(B)** 單一試次。Study 階段（4.3 s）：注視點 300 ms，左側顏色 1000 ms，右側顏色 1000 ms，
子音 1 1000 ms，子音 2 1000 ms。每組 study 之後連續出現 9 個探測（各 3.3 s）：注視點 300 ms，顏色方塊與子音同時
出現 1000 ms，作答窗從探測出現後 500 ms 開始、持續 2000 ms（y = 與記住的相同，n = 不同）。練習與校準試次另有回饋
畫面（1000 ms；校準為 800 ms）。校準試次的作答窗最長 3 s。**(C)** 每組 9 個探測的組成（隨機排列）：AA（顏色與聲音
都與某個 study 項目相同）2 個；HA / LA（只有顏色不同，H = 容易、L = 困難）各 1 個；AH / AL（只有聲音不同）各 1 個；
HH / HL / LH / LL（兩者都不同）共 3 個。最後四類構成 2 × 2 雙因子設計（double-factorial paradigm），其反應時間用
於計算 SIC 與 MIC，以推論顏色與聲音的處理架構。每場次共 432 個雙因子探測，每格約 108 個。

## Caption (English)

**Figure 1.** Experimental procedure. **(A)** Structure of one session. Participants first complete two calibration
blocks (72 trials each: 6 intensity levels × 8 "different" probes plus 24 "same" probes; feedback after every
trial). At the end of each block the lnrm2 model is fitted to accuracy and response time (20–40 s) and inverted to
obtain that participant's high- and low-salience levels (H / L): for colour the CIELAB distances ΔE_H / ΔE_L, for
sound the consonant-confusion counts count_H / count_L. Practice (5 study sets × 9 probes, 1 s feedback) and the
main task (6 blocks × 24 study sets × 9 probes = 1296 probes, no feedback, self-paced rest between blocks) follow.
**(B)** One trial. Study (4.3 s): fixation 300 ms, colour 1 on the left 1000 ms, colour 2 on the right 1000 ms,
consonant 1 1000 ms, consonant 2 1000 ms. Each study set is followed by nine probes (3.3 s each): fixation 300 ms,
a colour patch and a consonant presented together for 1000 ms, and a response window opening 500 ms after probe
onset and lasting 2000 ms (y = same as studied, n = different). Practice and calibration trials add a feedback
screen (1000 ms; 800 ms in calibration), and calibration trials allow up to 3 s to respond. **(C)** Composition of
the nine probes per study set (shuffled): AA (colour and sound both match a studied item) × 2; HA / LA (only
colour differs; H = easy, L = hard) × 1 each; AH / AL (only sound differs) × 1 each; HH / HL / LH / LL (both
differ) × 3. The last four types form the 2 × 2 double-factorial paradigm whose response times yield the SIC and
MIC used to infer the processing architecture of colour and sound; 432 double-factorial probes per session, about
108 per cell.

## 要注意的一件事 / One caveat

每格約 108 個探測，低於 `adaptiveSFT/results/power_scan.csv` 建議的 250。要提高 SIC 的檢定力，可把
`generate_stratified_conditions()`（`VAWM_nobox.py:137`）裡雙因子探測的數量從 3 提高，或增加 block 數。

About 108 probes per DFP cell is below the 250 per cell recommended by `adaptiveSFT/results/power_scan.csv`.
Raise the number of double-factorial probes per set in `generate_stratified_conditions()` (`VAWM_nobox.py:137`)
or add blocks to increase SIC power.
