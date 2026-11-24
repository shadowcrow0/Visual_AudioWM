---
review: 需要
review_note: 2026-11-24 查聲學尺文獻先例時建卡；只到摘要層級，引用前要讀全文
tags: [literature-note, 聽覺模型, microscopic, 樣板比對, 子音混淆預測, 聲學尺, AVWM]
citekey: zaar2017predicting
---

# Zaar & Dau (2017) — 聽覺模型前端 + 樣板比對後端，預測噪音下的子音辨識率與混淆

**和我們的「樣板觀察者」最接近的已發表方法。** 聽覺模型的內部表徵做互相關樣板比對，預測各 SNR 的子音辨識率與混淆。

**DOI / URL** https://doi.org/10.1121/1.4976054 ｜ 機構典藏全文 https://core.ac.uk/download/84004202.pdf

**閱讀狀態** **未讀全文。** 書目與下列內容來自 2026-11-24 網路搜尋回傳的出版社／PubMed／機構典藏紀錄與引用它的論文之摘要層級文字；這個工作環境的網路白名單開不了這些頁面，所以沒有逐字核對原文、沒有頁碼引句。引用前要讀全文。

```bibtex
@article{zaar2017predicting,
  author  = {Zaar, Johannes and Dau, Torsten},
  title   = {Predicting consonant recognition and confusions in normal-hearing listeners},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {141}, number = {2}, pages = {1051--1064}, year = {2017},
  doi     = {10.1121/1.4976054}
}
```

## 研究問題
一個以聽覺模型為前端的「微觀」模型，能不能逐子音預測正常聽力者在噪音下的辨識率和混淆？

## 方法與族群
- 前端：Dau、Kollmeier 與 Kohlrausch（1997）聽覺訊號處理模型的延伸；後端：以互相關為基礎的樣板比對。
- 驗證資料：[[zaar-dau2015]] 的 15 個 CV（白噪音、多個 SNR）。

## 結果與限制
- 辨識率在很寬的 SNR 範圍預測準確。
- 混淆的預測與聽者資料方向一致；但一個資料驅動的對照方法在預測**混淆**上較好，本模型在預測**辨識率**上較好。
- **對我們的意義**：白噪音、辨認作業、15 個子音接 /i/；沒有語音形噪音、沒有同/異作業、沒有非母語聽者。它是可以拿來對照我們頻譜圖差別的現成模型，但要先確認程式是否公開（**待查證**）。

## 可連結脈絡
- 驗證用的資料與變異來源分析 —— [[zaar-dau2015]]
- 另一個微觀模型（語音形噪音）—— [[jurgens-brand2009]]
- 可聽度／glimpse 類模型 —— [[cooke2006]]
- 我們的樣板觀察者遠優於人（0 dB 仍有 104 對 ≥ .90）—— [[literature_2026-11-24_acoustic_distance_and_consonant_confusion]]

---
標籤note:[[literature-note]] [[speech-perception]] [[AVWM]]

## 回查線索
**有沒有已發表的「聽覺表徵 + 樣板比對」子音模型？** → 本篇與 [[jurgens-brand2009]]。
**這類模型預測混淆準不準？** → 辨識率準，混淆不如資料驅動的方法。
