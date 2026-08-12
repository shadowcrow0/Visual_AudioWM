# Klatt (1980) — 串聯/並聯共振峰合成器

**出處** Klatt, D. H. (1980). Software for a cascade/parallel formant synthesizer.
*The Journal of the Acoustical Society of America*, 67(3), 971-995.

## 核心
參數式合成器。VOT 是**定義性參數**(嗓音振幅開始上升的時刻相對於 burst),不是副作用。

## 使用實例
Fox, N. P., Leonard, M., Sjerps, M. J., & Chang, E. F. (2020). Transformation of a
temporal speech cue to a spatial neural code in human auditory cortex. *eLife*, 9, e53051.
DOI 10.7554/eLife.53051
> "Stimuli were generated with a parallel/cascade Klatt-synthesizer KLSYN88a... all
> stimulus parameters were identical across stimuli, with the exception of the time at
> which the amplitude of voicing began to increase (in 10 ms steps from 0 ms to 50 ms
> after burst onset)."

## 對 AVWM 的意義
若最終需要 VOT 連續體,這是精度最高、可每試次即時生成的路(Praat KlattGrid 實測 8 ms/token,
斜率 1.0、單調)。但若採用 SNR 方案則不需要。

相關:[[mbrola-cannot-do-vot]] [[winn2013-vot-noise]]
