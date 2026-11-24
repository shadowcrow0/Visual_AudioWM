---
review: 需要
review_note: 2026-11-24 查聲學尺文獻先例時建卡；只到摘要層級，引用前要讀全文
tags: [literature-note, 聽覺模型, microscopic, 語音形噪音, DTW, 音素辨識預測, AVWM]
citekey: jurgens2009microscopic
---

# Jürgens & Brand (2009) — 聽覺模型 + 動態時間校正辨識器，在語音形噪音下逐音素預測辨識

**語音形噪音下的微觀模型先例。** 把模型的音素辨識表現和正常聽力者在語音形噪音下的表現直接比。

**DOI / URL** 作者機構 PDF https://medi.uni-oldenburg.de/download/pdfs/J%fcrgens_Brand_JASA2009_personalcopy.pdf ｜ DOI **待查證**

**閱讀狀態** **未讀全文。** 書目與下列內容來自 2026-11-24 網路搜尋回傳的出版社／PubMed／機構典藏紀錄與引用它的論文之摘要層級文字；這個工作環境的網路白名單開不了這些頁面，所以沒有逐字核對原文、沒有頁碼引句。引用前要讀全文。 期號**待查證**。

```bibtex
@article{jurgens2009microscopic,
  author  = {J{\"u}rgens, Tim and Brand, Thomas},
  title   = {Microscopic prediction of speech recognition for listeners with normal hearing in noise using an auditory model},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {126}, pages = {2635--2648}, year = {2009}
}
```

## 研究問題
不看平均可懂度、而是逐音素預測：一個聽覺模型加簡單辨識器，在語音形噪音下的音素辨識和人差多少？

## 方法與族群
- 聽覺模型前端；後端是簡單的動態時間校正（DTW）辨識器。
- 無意義語音、封閉式作業、語音形噪音。

## 結果與限制
- 搜尋摘要只確認了研究設計與比較的對象；模型與人的差距有多大、混淆矩陣對不對得上，**待讀全文**。
- **對我們的意義**：這是「聽覺表徵上的距離 → 語音形噪音下的音素辨識」的直接先例，噪音種類和我們相同。

## 可連結脈絡
- 白噪音下的同類模型 —— [[zaar-dau2017]]
- 可聽度／glimpse 類 —— [[cooke2006]]
- [[literature_2026-11-24_acoustic_distance_and_consonant_confusion]]

---
標籤note:[[literature-note]] [[speech-perception]] [[AVWM]]

## 回查線索
**有沒有在語音形噪音下用聽覺模型距離預測音素辨識的研究？** → 本篇。
