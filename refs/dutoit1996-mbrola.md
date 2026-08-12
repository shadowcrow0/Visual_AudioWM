# Dutoit et al. (1996) — MBROLA 專案原始文獻

**出處** Dutoit, T., Pagel, V., Pierret, N., Bataille, F., & Van der Vrecken, O. (1996).
The MBROLA project: Towards a set of high quality speech synthesizers free of use for
non-commercial purposes. *Proc. ICSLP '96*, 3, 1393-1396.
**DOI** 10.1109/ICSLP.1996.607874

## MBROLA 的真正定位
Europe PMC 中 91 篇 MBROLA 論文,絕大多數是**超音段**用途:韻律、節奏、語調、
音節時長、統計學習、神經夾帶。它的價值是「大量音節/語者/語言,而音節時長與音高輪廓
被精確且一致地指定」。

代表性用法(直接引句):
- Zhang et al. (2010) *Hum Brain Mapp* 31(7):1106-1116 — "The **duration and F0**
  information was fed into MBROLA"(注意:餵進去的就只有這兩樣)
- Peter et al. (2022) *Sci Rep* 12:13477 — "independent imposition of appropriate prosody
  including intonation, duration, and shift in spectral quality";其刺激規格寫
  "When the first consonant was a stop, that consonant was 20 ms in duration"
  —— 一個塞音就是一個時長數字,沒有 burst/送氣/嗓音起始的次結構
- Fló et al. (2025) *eLife* 13:RP101802 — "consonants 90 ms, vowels 160 ms"
- Martinez-Alvarez et al. (2023) *Sci Adv* 9:eade4083 — "duration of consonants was set
  to 120 ms and vowels to 150 ms"

## 對 AVWM 的意義
MBROLA 適合的是「音段內容當作要控制住的干擾變項」的研究 —— 這與 VOT 研究
(音段細結構**就是**自變項)剛好相反。但用來產生**乾淨的 be/pe token 再加噪音**完全適用。

相關:[[mbrola-cannot-do-vot]]
