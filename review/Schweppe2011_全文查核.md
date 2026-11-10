# Schweppe, Grice & Rummer (2011) 全文查核

*Journal of Memory and Language, 64*(3), 256–269. https://doi.org/10.1016/j.jml.2010.11.006
2026-11-10 依使用者上傳的 PDF 全文（14 頁）查核；取代先前只到摘要層級的描述，並**修正**企劃書軸表裡寫錯的那一列
（「操弄共享特徵數、voicing/nasality/manner/place 組合預測最好」是他們轉述 Wickelgren 1966，不是這篇自己的做法）。

另一份上傳的 219 頁 PDF 是 Schweppe 2006 年的博士論文《Shared representations in language processing and verbal
short-term memory: The case of grammatical gender》，主題是句子回憶中的文法性別，目錄裡沒有子音特徵的實驗，與本題無關，未再細讀。

## 一句話

不是算距離，是**選哪一個特徵共享**：三組音節，各組三個子音，只差一個「類型」的特徵。
共享 MANNER 的組他們叫「聲學相似」，共享 PLACE（同一主動構音器）的組叫「構音相似」。結果：共享 MANNER 在任何呈現／回憶方式下都傷回憶；共享 PLACE 只有口說回憶時才傷。

```
   組              音節           共享什麼                  差什麼            標籤
   ─────────────── ───────────── ──────────────────────── ───────────────── ──────────────
   acoustic        pa  ta  ka    manner = plosive          place L/C/D       「聲學相似」
   articulatory    da  la  za    place = CORONAL (舌尖)    manner 塞/邊/擦   「構音相似」
   distinct        fa  na  ga    —                         manner 與 place   不相似
   prefix/suffix   ja            glide, palatal            與三組皆不同      (Exp 1 邊界音)
```
（Table 1, p. 260）

## 六個問題

### 1. 用哪些特徵、來源

兩個特徵類型：**MANNER** 與 **PLACE（主動構音器 / place node）**。理論依據不是 Chomsky–Halle 的一張特徵表，而是兩個音韻學框架：
feature geometry（Clements 1985, 1992；Sagey 1986；Halle 1992）與 articulatory phonology（Browman & Goldstein 1986–1992）。

> "Based on the above considerations we take acoustic similarity to correspond to shared MANNER features and articulatory similarity to shared PLACE features." (p. 259)

> "An important articulatory dimension for consonants is the active articulator, that is, that part of the supralaryngeal vocal tract that is actively moved in order to produce the consonant ... In feature geometry terms, this involves the PLACE nodes LABIAL, CORONAL, DORSAL and RADICAL" (p. 258)

為什麼 manner 算「聲學」：manner 對聲學訊號的影響通常比 place 大，而且 Browman & Goldstein 與 Clements 都說 manner 特徵的定義混了聲學／空氣動力學成分、在特徵樹上掛得比 PLACE 高 (p. 259)。
voicing 與 nasality 沒有被當成操弄變項；distinct 組「部分」還差 voicing (p. 264)。

### 2. 相似度怎麼算

不算。每組內三個子音**在目標特徵上完全相同、其餘不同**，組間比較的是「共享哪一種特徵」，不是「共享幾個」。作者反而明說特徵數不是重點：

> "In terms of feature counts, the distinct list was expected to lead to better recall, as the number of feature distinctions was greater." (p. 262) — 結果 distinct 與 articulatory 沒差。

> "The additional place differences in the distinct lists did not improve recall, as might be expected if a simple count of differences across feature values were involved." (p. 264)

他們引 Brady, Mann & Schmidt (1987) 支持同一點：共享 0/1/2 個特徵與回憶之間沒有一致關係，而**共享 voicing 時最差** (p. 258)。

刺激常模：63 人以 1–10 分評 30 對音節的相似度；articulatory 組 4.8 vs distinct 4.3，acoustic 組 4.9 vs 4.3，兩個相似組之間沒差 (p. 261)。所以主觀相似度相同、記憶效果卻不同，是他們的重點。

### 3. 刺激

- 三組封閉集合，各 3 個 CV 音節、母音皆 /a/；每個試驗 5 個音節**從同一組**抽，所以一個試驗內必有重複 (p. 260)。
- 男聲錄音、每音節約 500 ms、ISI 500 ms (p. 261)；每條件 14 串列，共 42 試驗 (p. 261)。
- Exp 1 另加 /ja/ 當 prefix/suffix 測 Jones 的邊界線索說 (p. 260–261)。
- 受試者皆德語母語（Saarland／Erfurt 大學生，每實驗 36 人）(p. 260, 263, 264, 265)。
- 德語 biphone/bigram 頻率已核對，不是相似組較佔便宜 (p. 261)。

### 4. 作業

```
   Exp 1   聽覺呈現 · 書寫序列回憶 · 有／無 prefix-suffix
   Exp 2   視覺呈現 · 書寫回憶
   Exp 3   聽覺 vs 視覺呈現 · 口說回憶
   Exp 4   聽覺呈現 · 口說 vs 書寫回憶（同一受試者內）
```

> "In Experiment 1, we use written recall of auditory lists, in the second one written recall of visually presented lists. In the third experiment we investigate oral recall of both auditory and visual lists and in the fourth experiment we compare oral and written recall of auditory lists." (p. 260)

依變項：正確位置回憶率。

### 5. 哪些特徵重要（結果）

```
                              acoustic (pa ta ka)   articulatory (da la za)   distinct (fa na ga)
   Exp 1 聽覺→書寫                 81.6%                 87.9%                    87.4%     art = dist
   Exp 2 視覺→書寫                 92.6%                 96.7%                    96.0%     art = dist
   Exp 3 聽/視→口說                75.9%                 84.3%                    88.0%     dist > art (p = .005)
   Exp 4 聽覺→口說                   —                   84.9%                    88.0%     dist > art (p = .025)
   Exp 4 聽覺→書寫                   —                   87.3%                    88.0%     ns
   Exp 4 合併                       76.2%                 86.5%                    87.7%
```
(p. 261–262, 263, 264, 266)

> "acoustic similarity impairs performance no matter if the input is auditory or visual and regardless of whether recall is written or oral. Performance with articulatorily similar items, on the other hand, is equivalent to performance with dissimilar items, whatever the input mode and when recall is written. Only when recall requires overt articulation do we find a disadvantage caused by articulatory similarity." (p. 266)

> "we can conclude that PLACE similarity impairs recall only when recall is overtly articulated." (p. 266)

> "The effect of a MANNER feature, which is attached higher up in a feature tree (e.g. [continuant]), is thus greater than a PLACE feature, which is attached lower down." (p. 268)

Exp 2 視覺呈現下效應仍在，作者用它排除「只是聽錯」的知覺解釋 (p. 264)。

### 6. 跟 M&N 噪音混淆的關係

他們的「聲學相似」組 pa ta ka 就是 M&N 在噪音中最常互混的那一簇（同 voicing、同 manner、只差 place）；
「構音相似」組 da la za 在 M&N 裡是跨簇、幾乎不混的。記憶結果：前者傷、後者（書寫時）不傷。
這與「記憶混淆結構 ≈ 知覺混淆結構」是**同向**的——但作者自己的解釋是「特徵樹上較高的 MANNER 節點」，不是聲學混淆度；
而且他們引 Hintzman (1967)／Wickelgren (1969) 的爭論說聽覺 vs 構音編碼在記憶中「仍未解決」(p. 258)。

## 對本研究的意義

1. **企劃書軸表那一列要改**：Schweppe 2011 = 2 個特徵類型（MANNER vs PLACE），三組固定音節，不算特徵數。
2. **「共享幾個特徵」這條線在這篇是被反對的**（p. 262, 264；Brady 1987）。我們的 hamming/weighted 距離要說成「知覺（M&N）的工作假設」，不能說記憶文獻支持。
3. **對 (c)「記憶 ≈ 知覺混淆結構」**：Schweppe 2011 同向（pa-ta-ka 在兩邊都是最難分的），Roodenrys 2025 反向（視覺呈現下構音相似一樣干擾、一半 minimal pair 聲學不混）。兩篇合起來：記憶裡的相似至少有一部分不是聲學的，但「只差 place 的塞音組」在知覺與記憶裡都最脆弱是穩的。
4. **voicing**：這篇沒操弄 voicing；它轉述的 Brady 1987 說共享 voicing 時回憶最差——再一次不支持「記憶丟 voicing」。
5. **方法可借**：63 人 1–10 分相似度常模，用來證明兩個相似組主觀上一樣像、記憶上卻不一樣——我們的 `sound_pairs.csv` 若要分「知覺距離」與「記憶干擾」，可以照做一個短常模。
