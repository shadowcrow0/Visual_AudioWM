# 子音配對該選哪一組:/b/–/p/、/d/–/t/、還是 /g/–/k/

**綜合回顧 · 2026-08-12**
AVWM 的聽覺維度需要一組塞音 voicing 對比。三個發音部位怎麼選?

單篇文獻卡在 `90_Sources/`。**每一條主張標注證據強度與閱讀狀態;我的推論與原文陳述嚴格分開。**
姊妹回顧:[[natural-vs-synthetic-speech]](自然 vs 合成)· [[synthetic-speech-cognitive-load]](認知負荷)

---

## 0. 結論先講

**選 /b/–/p/(唇音),配 /i/ 母音 —— 也就是維持現有的 be/pe。證據強度:中等偏強。**

而且結論比預期乾淨:**軟顎音在五條獨立的證據上都比較差,唇音沒有任何一條專屬缺點。**這不是「勉強挑一個」,是三組裡有一組被明確排除、另一組只是次佳。

**關鍵發現是一筆直接的組內比較**([[goldenberg2022]]):同一批受試者、同一個實驗,**軟顎音的辨識函數寬度是唇音的近兩倍**(acuity 2.0 vs 1.1),而且邊界偏斜。**對一個要估斜率的適應式程序,這是本回顧中最有決策價值的一個數字。**

⚠️ **但仍要標明:沒有任何一篇文獻正面論證過三個部位在 VOT 實驗中的優劣。**[[winn2020]] 是這領域的標準方法學教學,經全文查證,**他對發音部位不做任何推薦**。本回顧是把散落的部位特定證據拼起來的**推論**。

---

## 1. 基本參數:VOT 範圍與知覺邊界

### 1.1 知覺邊界 —— 研究者的印象獲得證實

[[winn2020]] §II.A(全文已讀)原文:

> "In simple single-syllable recognition situations for English, one could expect the perceptual boundary between **/b/ and /p/ to fall somewhere around 20–25 ms**, the boundary between **/d/ and /t/ to fall near 35 ms**, and the boundary between **/g/ and /k/ to fall near 40 ms.**"

⚠️ Winn 給這些數字時**沒有引用特定來源**,是以「一般預期」的語氣寫的。

**獨立的唇音實證值**([[fox2020]],全文已讀):
> "A psychophysical category boundary between 20 ms and 30 ms divided the continuum into stimuli most often perceived as voiced (/b/: 0 ms, 10 ms, 20 ms VOTs) or as voiceless (/p/: 30 ms, 40 ms, 50 ms VOTs)."

**Winn Table I 的端點建議**:/b p/ 3→60 ms;/d t/ 10→70 ms;/g k/ 15→70 ms。

### 1.2 產出數值:軟顎音的兩端**最擠**

[[chodroff2017]](*J. Phonetics* 61, 30–47,全文已讀)是現代大語料庫版本,Table 1(孤立語)與 Table 6(連續語)的平均 VOT(ms):

| | pʰ | tʰ | kʰ | b | d | g |
|---|---|---|---|---|---|---|
| 孤立語 平均 | 89 | 98 | 99 | 13 | 21 | **28** |
| 連續語 平均 | 51 | 61 | 56 | 8 | 14 | **17** |
| 孤立語 talker 平均的 SD | 27 | 28 | 24 | 5 | 7 | **10** |

**由這些已發表平均值算出的有聲–無聲間距(這是我的算術,不是原作者的主張)**:

| | 唇音 | 舌尖音 | 軟顎音 |
|---|---|---|---|
| 孤立語 | 76 | 77 | **71** |
| 連續語 | 43 | 47 | **39** |

**→ 軟顎音的可用區間最窄**,因為 /g/ 的 short-lag VOT 最長(28 / 17 ms),把有聲端往上頂。而且**有聲端的說話者間變異隨部位後移而變大**(b 5 → d 7 → g 10)。

**對 AVWM 的意義(我的推論)**:適應式程序需要一段乾淨、夠寬的搜尋範圍。唇音的有聲錨點最低(8–13 ms)、間距最寬,**給適應程序最大的操作空間**。

### 1.3 經典的部位序列其實不穩

[[chodroff2017]] 原文直接修正了教科書說法:
> "The present study observed a strong tendency for the ranking of [pʰ]<[kʰ] ... and **little difference between the means of [tʰ] and [kʰ]** within or across talkers in both studies. Among the voiced stops, the overwhelming majority of speakers had increasing VOT with more posterior places of articulation ([b]<[d]<[g])."

[[chodroff2019]] 的 100+ 語言調查也讓 [[winn2020]] 加了但書:
> "**a more recent cross-linguistic study by Chodroff et al. (2019) challenges the long-held notion of this ordering** of VOT based on place of articulation"

**→ 無聲端的三分序列不可靠;有聲端的 b<d<g 才穩。**這正好強化 §1.2 的重點:唇音的優勢在**有聲端**,而有聲端的序列是可靠的那一半。

### 1.4 為什麼 VOT 隨部位後移而增加

[[abramson2017]](全文已讀,原始 XML 逐字核對)§3:
> "Using VOT data from 18 languages, **Cho and Ladefoged (1999)** ... focusing their attention on the increasing values of voicing lag as the place of stop articulation moves from the lips to the back of the mouth ... Cho and Ladefoged discuss such possible causative factors as **cavity size behind the occlusion and in front of it, the size of the area of contact, and the aerodynamics in the region of the larynx.**"

直接的空氣動力學證據 —— Eshghi, Alemi & Zajac (2016), *Folia Phoniatr. Logop.* 68(5), 239–246, doi 10.1159/000478523(⚠️ **僅讀摘要**):
> "Aerodynamically, Po was greatest for the velar stop, intermediate for the alveolar stop, and smallest for the bilabial stop."

⚠️ 注意這支持的是**「唇音 vs 其餘」的二分**,而不是乾淨的三分序列。

⚠️ **Cho & Ladefoged (1999) 全文未取得**(確認為封閉取用),機制細節僅透過 [[abramson2017]] 的轉述。

---

## 2. 決定性證據:軟顎音的辨識函數又淺又偏

**這是本回顧唯一一筆「同一批受試者、同一個實驗、直接比較兩個部位」的資料,也是最有決策價值的一條。**

[[goldenberg2022]](*Front. Hum. Neurosci.* 16:879981,全文已讀)原文:

> "The bilabial category boundary is approximately centered between its endpoints, that is, its bias (4.2) is close to its midpoint (4.5). ... **Acuity (a measure of boundary slope) was computed as the difference between the 25 and 75% probabilities** for the discrimination function. **The velar category boundary is not as centralized and is skewed toward voicelessness (bias = 3.6)**; that is, longer VOTs were necessary for /ka/ responses. **The velar acuity (2.0) is shallower than that of the bilabial (1.1)**, possibly due to this skew."

**因為 acuity 定義為 25–75% 的寬度,數值越大代表越淺:軟顎音的辨識函數寬度約為唇音的兩倍。**

**對 AVWM 的直接後果(我的推論)**:
1. **適應式程序估斜率會更慢、更不準。**函數越淺,同樣的試次數換到的 β 精度越差。
2. **邊界偏斜代表對稱性假設不成立。**AGRT 的雙極結構預期邊界大致落在中點;軟顎音的 bias(3.6)明顯偏離中點(4.5),唇音(4.2)則貼近。
3. GRT 要估的知覺分布,在一個偏斜、淺平的維度上更難與高斯假設相容。

⚠️ **限制**:這是**單一研究**的附帶觀察,不是專門為比較部位而設計的實驗;作者自己用 "possibly due to this skew" 這種試探語氣。**證據強度:中等**(直接比較是強項,單一來源是弱項)。

---

## 3. 軟顎音的三個結構性問題

### 3.1 雙重 burst,而且在無聲端更嚴重

[[kingston1983]](*JASA* 74(S1), S51–S52,⚠️ **僅讀摘要,且該摘要本身是會議摘要**)原文:

> "**Multiple intensity peaks, 'double bursts,' are common in the release of velar stops.** ... If a high rate of air flow causes this second, brief closure, then **double bursts should occur more often after /k/ than /g/**, since intraoral air pressure during closure is higher for the voiceless stop than the voiced one."

> "Intensity measurements ... show **vowel quality determines burst intensity for velars—least intense before [u], most before [i]** ... more than the voiced/voiceless or aspirated/unaspirated contrasts do. **Alveolar releases vary much less across vowels.**"

**三件事一次到位:**
1. 雙重 burst 在軟顎音「common」;
2. 預期**在 /k/ 比 /g/ 更常見** —— 正好落在要建的連續體的無聲端;
3. 軟顎音的 burst 強度**由母音決定的程度超過由 voicing 決定的程度**,而舌尖音「varies much less across vowels」。

**為什麼這對 AVWM 要緊**:VOT 是**從 burst onset 起算**的。burst 結構不單純,burst onset 的定位就不單純。[[abramson2017]] 已記錄 burst/送氣界線難劃是主要量測誤差來源。

⚠️ **重要的負面結果**:[[abramson2017]] 是現代 VOT **量測**問題的標準回顧,而它**全文從未出現 "velar" 一詞**,也沒有討論多重 burst。**所以「軟顎音比較難量」是從 Kingston (1983) 推出的合理推論,不是量測文獻的既有陳述。**

⚠️ **查無**:多重 burst 的**發生率統計**、以及各部位 **burst 時長**的數值。這是本回顧覆蓋最弱的一項 ——「軟顎音 burst 較長」這個說法**沒有任何取得到的來源支持**,不應寫進論文。

### 3.2 軟顎音的 burst 頻譜隨母音劇烈變動 —— 對合成是致命的

[[frisch2016]](*J. Phonetics* 56, 52–65,PMC4805126,全文已讀)用超音波確認了英語軟顎音在前母音前的前移:
> "the difference in closure location on the palate between onsets in the words k[ey] and c[ough] ... is large enough to be noticed by naïve speakers despite being allophonic"

而該文轉述 Keating & Lahiri (1993)(⚠️ **二手**,原文未讀)給出了關鍵數字:
> "Keating and Lahiri (1993) ... conclude that **the prominent frequency peak in the burst spectrum is distinct for all five contexts and varies systematically with vowel frontness.** ... A closer examination of the frequency peaks shows a rather large difference between **front vowel contexts (about 3,000 Hz) and back vowel contexts (1,000–1,500 Hz).**"

**→ 軟顎音的 burst 頻譜峰值在前母音與後母音間差了兩倍以上。**Keating & Lahiri 的結論是舌體前後對軟顎音**沒有內在目標**,完全由協同構音決定 —— 這正是固定 burst 的合成器會失敗的原因。

**對照組**:[[fox2020]] 用 Klatt 合成 **/ba/–/pa/**,方法段原文:
> "**The onset noise-burst was 2 ms in duration and had constant spectral properties across all stimuli.**"

**一個 2 ms、頻譜固定的 burst,對唇音是站得住的設計;依上述數據,對軟顎音則不然。**(這個對照是我的推論,Fox 等人沒有討論部位選擇。)

### 3.3 軟顎音 + 前母音在知覺上靠近 /tʃ/

Guion, S. G. (1998). *Phonetica* 55(1–2), 18–52, doi 10.1159/000028423(⚠️ **僅讀摘要**):
> "Voiceless velar stops may become palatoalveolar affricates before front vowels. ... It is shown that **velars before front vowels are both acoustically and perceptually similar to palatoalveolars.**"

**對一個需要子音身分毫不含糊的 GRT 設計,這是實質風險。**(此推論為我所加。)

---

## 4. 母音脈絡:/i/ 是對的 —— 而且它與軟顎音衝突

[[winn2020]] §II.D(全文已讀)原文:

> "**the /ɑ/ vowel context could be a particularly unfortunate choice** for experimenters hoping to isolate auditory processing of a purely temporal nature. **Conversely, the /i/ context would be far less affected by VOT cutback**, since (1) its formants are more stable across time in general and (2) the F1 of /i/ is already low ... **F1 for /i/ simply remains at a low frequency regardless of the amount of vowel cutback, thus offering no covarying cue for VOT.**"

他要避開的混淆有多大:
> "The difference of 300 Hz in F1 ... occupies roughly 3 mm of cochlear space, which is roughly **10% of the tonotopic range** of the basilar membrane in an adult human"

**這個建議是通用的、不限部位**,而且他明確點名兩種脈絡:"e.g., for **/bɑ/-/pɑ/** sounds or **/dɑ/-/tɑ/** sounds"。

**F1 混淆的量化參考**(⚠️ **二手**,轉引自 Benkí 2005):Kluender (1991) *JASA* 90, 83–96 報告 F1 起始差 400 Hz 約值 **7 ms 的 VOT 邊界位移**(跨三個部位平均)。

### 4.1 ⚠️ 這個衝突沒有人寫過 —— 但它只咬軟顎音

**明確的負面結果**:[[winn2020]] 全文檢索,**"palatal" 出現 0 次**;"velar" 只出現 3 次(兩次在 VOT 範圍段、一次在 burst 頻譜段)。**他推薦 /i/,卻從未提及軟顎音前移、顎化、或他的母音建議與子音選擇之間的任何交互作用。兩個建議並存於同一篇論文而從未被調和。**

targeted 檢索(Europe PMC、Crossref、四輪網路搜尋)也找不到任何討論此權衡的論文。**這是真實的文獻空白。**

**但實務上這個衝突會自己消失 —— 只要選唇音。** /bi/–/pi/ 完整享有 Winn 的 F1 好處,而且完全沒有顎化問題。**被雙重詛咒的只有 /gi/–/ki/。**

⚠️ **一個與部位無關的 /i/ 注意事項**([[chodroff2017]] 原文):
> "Longer VOTs are observed before high and tense vowels, particularly [i], for voiceless stops (Klatt, 1975; ...)"

**→ /i/ 脈絡的連續體端點應比 Winn Table I 的(非母音特定的)數值稍微往長的方向調。**

---

## 5. 文獻先例:唇音是合成 VOT 連續體的主場

| 研究 | 連續體 | 部位 | 方法 | 閱讀狀態 |
|---|---|---|---|---|
| Abramson & Lisker(Haskins legacy set) | /ba/–/pa/ | **唇音** | Haskins 共振峰合成器 | 官網頁面已讀 |
| [[fox2020]] *eLife* 9:e53051 | /ba/–/pa/ | **唇音** | Klatt KLSYN88a | 方法段已讀 |
| [[zuk2013]] *PLoS ONE* 8:e80546 | /ga/–/ka/ | **軟顎音** | Klatt-based | 方法段已讀 |
| [[goldenberg2022]] | pa/ba **與** ka/ga | **兩者** | 自然,漸進縮短送氣 | 全文已讀 |
| *Sci. Rep.* 14:28825 (2024), PMC11582665 | **beach/peach** + dime/time | **唇音(+/i/!)** | Praat 自然漸進 cross-splicing | 全文已讀 |
| Benkí 2005, ISB4 240–248 | bossy/posse | **唇音** | Klatt & Klatt (1990) cascade | PDF 已讀 |
| [[silbert2012]] / [[silbert2014]] | — | **唇音 / 唇音+舌尖音** | 自然錄音 + 噪音 | 見各卡 |
| [[mcmurray2008]] Exp 1 | /b/–/p/ 系列詞 | **唇音** | 自然 cross-splicing | 見卡 |

**兩個觀察:**

1. **領域史上最常被重用的合成 VOT 連續體(Haskins 的 Abramson/Lisker 組)就是唇音。**
2. ⭐ **`beach/peach` —— 唇音 + /i/ + 自然漸進 cross-splicing —— 已經有已發表的使用先例**(*Sci. Rep.* 2024, PMC11582665)。**AVWM 想做的那個設計不是自創的。**

**粗略的分佈指標**(⚠️ Europe PMC 的**詞彙共現**計數,**不是**經整理的研究計數,含偽陽性,**不可當作調查數據引用**):「ba+pa+VOT continuum」51、「da+ta」39、「ga+ka」13。方向與上表一致。

**Winn 自己的例子刻意保持部位中立**(§IV.A):"like **deer/tier, big/pig, goat/coat**, etc." —— 三個部位各給一個。

⚠️ **明確的負面結果**:**沒有任何論文說某個部位比較好合成或比較難合成,也沒有任何方法段以合成品質為理由解釋部位選擇。**

---

## 6. 沒有查到的東西

1. **各部位 burst 時長的數值** —— 完全查無。本回顧最弱的一項。
2. **軟顎音多重 burst 的發生率統計** —— 查無。
3. **Lisker & Abramson (1964) 原文** —— 出版社 403。英語平均值僅有二手來源(Chen, Chao & Peng 2007 的 Table 1 重製:/b/ 1, /d/ 5, /g/ 21;/p/ 58, /t/ 70, /k/ 80 ms)。**引用須標明轉引。**
4. **Lisker & Abramson (1970)** 的知覺邊界值 —— 會議論文集,遍尋不獲。
5. **Benkí (2001)** *J. Phonetics* 29(1), 1–22, doi 10.1006/jpho.2000.0128,「Place of articulation and first formant transition pattern both affect perception of voicing in English」—— **這是唯一一篇正面交叉「部位 × F1」的論文,付費牆擋住。強烈建議透過館藏取得。**光是標題就說明部位對 F1 混淆不是中性的。
6. **Volaitis & Miller (1992)** *JASA* 92, 723–735 —— 經典的唇音 vs 軟顎音類別內部結構比較,出版社 403。
7. **Cho & Ladefoged (1999)** 全文 —— 封閉取用。
8. **Keating & Lahiri (1993)** 全文 —— 僅二手。
9. **任何正面比較三部位在 VOT 實驗中優劣的研究** —— **查無,真實空白。**

---

## 7. 建議

### 7.1 選 /b/–/p/ + /i/(維持 be/pe)

依證據強度排序:

| # | 理由 | 來源 | 強度 |
|---|---|---|---|
| 1 | **軟顎音辨識函數寬約兩倍且邊界偏斜**;唇音邊界置中 | [[goldenberg2022]] 直接組內比較 | **中等**(直接但單一來源) |
| 2 | **軟顎音 burst 頻譜隨母音差兩倍以上**,固定 burst 的合成必然失真;唇音可用 2 ms 固定 burst | [[frisch2016]]、Keating & Lahiri(二手)、[[fox2020]] | **中等偏強** |
| 3 | **軟顎音雙重 burst 常見,且在 /k/ 端更甚**,威脅 burst onset 定位 | [[kingston1983]](僅讀會議摘要) | **弱到中等** |
| 4 | **/i/ 有明確方法學背書**,而 /i/ 正是軟顎音顎化最嚴重的環境;唇音完全繞開 | [[winn2020]] §II.D + Guion 1998 | **強**(Winn)/**中等**(衝突為我的推論) |
| 5 | **唇音有聲錨點最低、間距最寬、有聲端變異最小** → 適應程序空間最大 | [[chodroff2017]](我的算術) | **中等** |
| 6 | **唇音是合成與自然 VOT 連續體的主場**,且 beach/peach 已有先例 | §5 | **弱**(便利樣本) |
| 7 | 實務:be/pe 是現有素材,`snr_audio.py` 已完成 | 專案內部 | — |

**舌尖音 /d/–/t/ 是合格的第二順位。**沒有查到針對它的專屬缺點,而且是 Keating/Nittrouer 那條 VOT 文獻的主場([[burst-vot-tradeoff]])。若 be/pe 錄音有問題,換 /d/–/t/ 不需重新論證。

**軟顎音 /g/–/k/ 應排除。**

### 7.2 這次的結論不再與路線選擇耦合

初稿曾推測:走合成路線時軟顎音反而較誠實(因為它本來就沒有 burst 頻譜的 voicing 線索,[[chodroff2014]])。**查證後這個推測站不住** —— §3.2 顯示軟顎音的 burst 頻譜**隨母音**劇烈變動,固定 burst 的合成器對它失真最大。**軟顎音在兩條路線上都差。**

**唇音在兩條路線上都好:**SNR 路線上它的 burst 頻譜 voicing 線索最強([[chodroff2014]] 預測 /p/–/b/「if anything stronger」);合成路線上它的 burst 可以合理地固定([[fox2020]] 的實作證明)。

### 7.3 論文寫作注意

1. **不要寫「文獻建議用唇音」。**沒有這回事。正確寫法:「唇音在 X、Y、Z 上有優勢,且未發現針對它的特定缺點」。
2. **引用 [[chodroff2014]] 的軟顎音結果時必須寫成產出/聲學層次**,不是知覺虛無結果(詳見該卡的引用查核)。
3. **引用部位 VOT 數值時**:優先用 [[chodroff2017]](現代大語料庫、可直接取得);要用 Lisker & Abramson (1964) 的數字必須標明轉引。
4. **引用 3 ms burst trading relation 時標明它來自波蘭語資料**([[burst-vot-tradeoff]])。
5. **/i/ 脈絡的端點要比 Winn Table I 稍往長調**(§4.1)。
6. **不要寫「軟顎音 burst 較長」** —— 查無來源(§6.1)。

---

**相關卡片**:[[goldenberg2022]] · [[kingston1983]] · [[frisch2016]] · [[chodroff2017]] · [[chodroff2014]] · [[chodroff2019]] · [[fox2020]] · [[burst-vot-tradeoff]] · [[winn2020]] · [[abramson2017]] · [[silbert2012]] · [[silbert2014]] · [[mcmurray2008]] · [[zuk2013]] · [[klatt1980]]
**其他回顧**:[[natural-vs-synthetic-speech]] · [[synthetic-speech-cognitive-load]]
**專案決策脈絡**:[[決策脈絡_聽覺維度]]

---
標籤note:[[literature-note]] [[speech-perception]] [[AVWM]]
