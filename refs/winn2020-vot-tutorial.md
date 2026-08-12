# Winn (2020) — VOT 操弄的方法學參考文獻

**出處** Winn, M. B. (2020). Manipulation of voice onset time in speech stimuli:
A tutorial and flexible Praat script. *The Journal of the Acoustical Society of America*,
147(2), 852-866.
**DOI** 10.1121/10.0000692 (CC-BY,全文與腳本 https://github.com/ListenLab/VOT)

## 核心方法:progressive cutback and replacement
取 burst 對齊的自然 /bV/ 與 /pV/,每一階**漸進刪除有聲 token 的起始段,換成等量的
無聲 token 的 burst+送氣**。腳本實測參數:crossfade 6 ms、`vowel_cutback_to_VOT_ratio = 0.65`
(每 1 ms 的 VOT 對應切掉 0.65 ms 母音)。

## ⚠️ 重要:這個方法**不是**固定網格
Winn 明確說明為什麼用 crossfade 而非直接接合:
> "First, it frees the process from the need to concatenate only at zero crossings,
> **which would place limitations on the minimal change in VOT that could be included in
> a continuum**."

送氣段擷取長度會補償一半的混合時間,"so that the output VOT should be exactly as the user
intends"。**因此 VOT 是 sample-accurate 的任意實數**,腳本裡的 N 階只是介面選擇。
(本地實測:任意 VOT 如 13.70 / 41.30 ms 皆可,Δ時長/ΔVOT = 0.35 = 1−0.65,**0.47 ms/token**。)

## 一個直接影響刺激選擇的建議:用 /i/ 不要用 /ɑ/
§II.D:低母音 /ɑ/ 的 F1 起始在 /dɑ/(~400 Hz)與 /tɑ/(~700 Hz)差約 300 Hz —— 約
3 mm 耳蝸距離。
> "the /ɑ/ vowel context could be a particularly unfortunate choice for experimenters
> hoping to isolate auditory processing of a purely temporal nature"

**這正好支持 AVWM 用 be/pe(/i/ 母音)而非 b3/p3(/ɜː/)** —— /i/ 的 F1 本來就低且穩定,
母音切除不會產生共變的 F1 線索。

## 明確不推薦的三種做法
預貼送氣(Iverson 2003; Gordon-Salant et al. 2006)、選擇性刪除送氣(Andruski 1994)、
送氣時間扭曲(Schoonmaker-Gates 2015; Schertz & Hawthorne 2018) —— 都會讓 F1 線索與
VOT 不匹配而偏移 voicing 知覺。

## 對 MBROLA 的沉默
全文沒有 "MBROLA"、"diphone"、"concatenative synthesis" 任何一詞。

相關:[[mbrola-cannot-do-vot]] [[winn2013-vot-noise]] [[mcmurray2022-cp-myth]]
