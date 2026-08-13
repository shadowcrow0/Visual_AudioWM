---
tags: [literature-note, GRT, 教學論文, 知覺維度, 建模假設]
citekey: silbert-hawkins2016
---

# Silbert & Hawkins (2016) — GRT 教學:知覺維度「對應」物理維度是個建模慣例

**DOI / URL** https://doi.org/10.1016/j.jmp.2016.04.011 (經 Crossref 核實) | PDF 來源 https://rdhawkins.com/wp-content/uploads/2013/11/silberthawkins16_grttutorial.pdf
**閱讀狀態** **全文 PDF 已下載並轉為文字,由我親自檢索**(非通讀)。書目資料取自 PDF 首頁頁眉原文 "Journal of Mathematical Psychology 73 (2016) 94–109",並經 Crossref 獨立核實。

```bibtex
@article{silbert2016tutorial,
  author  = {Silbert, Noah H. and Hawkins, Robert X. D.},
  title   = {A tutorial on General Recognition Theory},
  journal = {Journal of Mathematical Psychology},
  volume  = {73}, pages = {94--109}, year = {2016},
  doi     = {10.1016/j.jmp.2016.04.011}
}
```

## 研究問題
GRT 的概念與數學結構是什麼?非參數與參數(高斯)兩類工具各能回答什麼?(教學論文,無受試者。)

## 結果與限制
**本卡只處理一件事:GRT 如何看待「物理維度 → 知覺維度」的對應。**

檢索全文後,**唯一**正面談這件事的句子是(原文,經 PDF 轉文字取得):
> "The dimensions along which the perceptual distributions and decision bounds are defined
> are modeled perceptual dimensions corresponding to the physical dimensions of the
> stimuli."

**這句話把 [[silbert2012]] 的顧慮講成了 GRT 的建模慣例:模型裡的知覺維度,是「對應到刺激物理維度」而定義的。**

也就是說,GRT 的輸出永遠是**相對於實驗者所選的物理維度**而言的。若把 VOT 設成唯一變動的參數、把 F1 釘死,那麼模型估出來的「聽覺知覺維度」就是「VOT 這條軸」—— 模型**在結構上沒有辦法**告訴你受試者其實靠別的東西在做判斷。

**但必須誠實標明兩點**:

1. **這句話是以中性的建模說明語氣寫的,不是警告。** 教學論文並沒有從這個慣例推出「所以要用自然刺激」。我全文檢索過 `synthetic`、`natural`、`manipulat`、`stimulus set`、`ceiling` 等關鍵詞:
   - `synthetic` / `naturally produced` / `natural speech`:**0 次命中**
   - `manipulat`:**0 次命中**
   - 沒有任何一節在談刺激製作方式的選擇
   
   **→ 因此:「刺激沿單一參數變動會預先決定估到的知覺維度」這個前提,在 GRT 教學文獻裡是被明白寫出來的建模結構;但「所以應該用自然刺激」這個規範性推論,不在教學論文裡,是 Silbert 在自己的實證論文中的做法。**

2. 這是一句在解釋圖 1(2×2 高斯 GRT 模型視覺化)時的說明文字,不是理論宣言。過度解讀有風險。

**另一個對 AVWM 有用的點**:教學文中的 2×2 範例用的是 **Silbert, Townsend et al. (2009)** 的非語音實驗 —— 寬頻噪音中的**頻率 × 時長**:
> "listeners simultaneously identified the frequency range and duration of the stimuli,
> and the stimulus set consisted of the factorial combination of low and high frequency
> levels (490–1490 Hz and 510–1510 Hz, respectively) and short and long duration levels
> (250 ms and 300 ms, respectively)."

注意這裡**也是把刺激埋在寬頻噪音裡**來製造誤差。

## 可連結脈絡
- 提出規範性推論的實證論文 —— [[silbert2012]]、[[silbert2014]]
- 獨立提出同一顧慮的第二個聲音 —— [[roark2019]]
- 另一份 GRT 教學(講難度操弄)—— [[soto2017]]
- 綜合建議 —— [[natural-vs-synthetic-speech]]

---
標籤note:[[literature-note]] [[GRT]] [[AVWM]]

## 回查線索
**GRT 的知覺維度跟物理維度是什麼關係?** → 本篇:「對應」是建模慣例,不是實證發現。
**我在哪查證過「某個規範性主張其實不在教學文獻裡」?** → 本卡(用關鍵詞 0 命中當證據,同 [[mbrola-cannot-do-vot]] 的推論模式)。
