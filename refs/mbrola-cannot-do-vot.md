# MBROLA 為何做不出 VOT 連續體(綜合卡)

## 結論
MBROLA 是 diphone 串接合成器,**控制面只有三個:音素身分、音素時長、F0 目標點**。
VOT 是音素**內部**的時序(嗓音起始相對於 burst 釋放),不在這三者之內。

## 三層證據

**1. 架構 —— MBROLA 官方 README**
> "Each line contains a phoneme name, a duration (in ms), and a series (possibly none)
> of pitch targets composed of two float numbers each."

**2. 作者自己承認次音素限制**
D'Alessandro, N., Sebbe, R., Bozkurt, B., & Dutoit, T. (2005). MaxMBROLA: A Max/MSP
MBROLA-based tool for real-time voice synthesis. *Proc. EUSIPCO 2005*, Antalya.
https://zenodo.org/records/39427

Dutoit(MBROLA 創作者)在 "Problems and perspectives" 寫道:
> "Unavailability of 'subphonemic' pitch control: ... **phoneme duration is the smallest
> subdivision of time for control.**"

**3. 已發表的明確陳述:MBROLA 的時間操弄會把 VOT 一起拉伸**
Vojtech, J. M., Noordzij, J. P., Cler, G. J., & Stepp, C. E. (2019). *American Journal of
Speech-Language Pathology*, 28, 875-886. DOI 10.1044/2019_AJSLP-MSC18-18-0052
> "decreasing speech rate using MBROLA 2.00 only stretches the spectral content of the
> sentence in time. Thus, acoustically relevant properties of speech, such as voice onset
> time, are linearly stretched in time..."

## 文獻計量佐證
全文檢索 `"VOT continuum"` 的共現次數:MBROLA **0**、Klatt **114**、Praat **200**
(OpenAlex / Europe PMC)。

## 本地實測佐證(2026-08,us1 語者,F0=180Hz)
- `b` 的 VOT 在時長 8-200 ms 全程固定在 -6.5 ~ -2.5 ms(49 個請求值只落在 7 個相異結果)
- voicing onset 量化到**一個音高週期**(180Hz -> 階梯 5.5 ms;300Hz -> 3.3 ms)
- 請求時長 +1 ms 常讓實測 VOT **下降**(非單調 -> 對 Psi 致命)
- 插入 `h` 段的 workaround 無效:VOT 飽和在 ~7 ms(MBROLA 把 h 合成成有聲的 [ɦ])

相關:[[winn2020-vot-tutorial]] [[zuk2013-musicians-vot]] [[dutoit1996-mbrola]]
