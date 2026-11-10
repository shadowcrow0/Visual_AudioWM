# Miller & Nicely (1955) 噪音下的子音混淆：逐 SNR 看哪些對會混、哪些分得開

來源：Miller & Nicely (1955), *JASA* 27, 338–352，Tables I–VI（S/N = −18、−12、−6、0、+6、+12 dB，頻寬 200–6500 cps，每格 4000 次判斷）。
矩陣由我從掃描 PDF 手抄（2026-11-09），存在 `review/data/miller_nicely_1955_tables_I-VI.json`；六張表每列加總都 ≈ 250、全表 = 4000，和論文說的一致，抄寫應無大錯。
每一對的「混淆率」= (a 聽成 b + b 聽成 a) ÷ (a 與 b 的呈現次數)，存在 `review/data/miller_nicely_1955_pair_confusion_rates.json`。

## 1. 每個子音的答對率（%）

```
   SNR       p    t    k    f    θ    s    ʃ    b    d    g    v    ð    z    ʒ    m    n
    -18      5   12   10    9    6    8    5    9    8    5    7    3    9    5   10   14
    -12     20   20   26   33   17   23   40   23   21   12   30   12   17   33   51   56
     -6     34   34   38   61   22   43   79   55   36   26   54   24   39   56   75   69
      0     51   77   50   75   49   75   97   75   73   59   75   31   59   92   97   95
      6     69   92   79   75   62   95  100   82   89   70   76   68   84   96   99   99
     12     85   98   91   88   69  100  100   86   90   85   82   87   94  100   99  100
```

最耐噪：ʃ、m、n、s（0 dB 就 ≥ 95%）；最脆弱：ð、θ、g、p（+12 dB 仍只有 69–87%）。

## 2. 每個 SNR 最常混的對（混淆率）

```
    -6 dB   f–θ 26%  v–ð 26%  p–k 25%  t–k 25%  p–t 24%  d–g 23%  m–n 20%  z–ʒ 19%  b–v 18%  g–ʒ 15%
     0 dB   p–k 30%  ð–z 30%  f–θ 26%  d–g 23%  t–k 14%  p–t 12%  θ–s 10%  b–v 10%  v–ð 9%  z–ʒ 8%
     6 dB   f–θ 25%  p–k 21%  v–ð 18%  d–g 15%  b–v 9%  ð–z 8%  t–k 4%  p–t 3%  b–ð 3%  g–z 2%
    12 dB   f–θ 18%  v–ð 12%  d–g 12%  p–k 11%  b–v 6%  b–ð 2%  ð–z 2%  p–θ 2%  g–ð 1%  f–b 1%
```

規律很一致：會混的全是**同 voicing、同 manner、只差 place（或 duration）**的對——
```
   無聲塞音  p t k        有聲塞音  b d g        鼻音  m n
   無聲擦音  f θ | s ʃ    有聲擦音  v ð | z ʒ    （b–v、d–ð 是塞音/擦音跨群但同 voicing 同 place）
```
到 +12 dB 還在混的只剩 f–θ 18%、v–ð 12%、d–g 12%、p–k 11%、b–v 6%：這幾對在安靜環境也不容易。

## 3. 跨情境都分得開的對

0、+6、+12 dB 一次都沒混、−6 dB 混淆率 < 1% 的有 **55 對**（120 對中）；其中到 −12 dB 仍 < 2% 的有 **46 對**：

```
   p–b  p–d  p–g  p–v  p–z  p–ʒ  p–m  p–n  t–b  t–d  t–v  t–ð
   t–z  t–ʒ  t–m  t–n  k–b  k–d  k–g  k–v  k–ð  k–z  k–ʒ  k–m
   f–d  f–g  f–z  f–ʒ  f–n  θ–ʒ  θ–m  θ–n  s–d  s–v  s–ð  s–ʒ
   s–m  s–n  ʃ–b  ʃ–v  ʃ–ð  ʃ–ʒ  ʃ–m  ʃ–n  z–m  ʒ–m
```

全部是跨群的對：差 voicing、差 nasality、或塞音 vs 擦音再加 place。注意**只差 voicing 的 p–b、t–d、k–g 也在名單裡**——噪音蓋不掉 voicing，所以它們在噪音裡「分得開」；這正是這份資料不能直接拿來當記憶相似度的地方（見下、§5）。

只差 voicing 的七對，噪音裡的混淆率：

```
   對      −18   −12    −6     0    +6   +12 dB
   p–b      5.1   1.2   0.2   0.0   0.0   0.0
   t–d      6.6   0.6   0.0   0.0   0.0   0.0
   k–g      7.0   0.4   0.0   0.0   0.0   0.0
   f–v      5.5   4.1   0.7   0.6   0.4   0.4
   θ–ð      2.6   2.0   1.9   0.0   0.4   0.4
   s–z      6.0   4.3   0.8   0.0   0.0   0.0
   ʃ–ʒ      3.1   1.9   0.0   0.0   0.0   0.0
```
−12 dB 以上幾乎不混（voicing 是噪音裡最耐的特徵，論文的結論），但這是「聽」的結果；它們在記憶裡、對非母語者是另一回事（見 §5）。

## 4. 現在 repo 裡那張表是哪裡來的

`generate_trials.py` 的 `CONFUSION_DATA` 寫「三個 SNR 加總」。對過：最接近 −6 / 0 / +6 dB 三張表的和（96 格裡 44 格完全相同，其餘差 1–5，像是抄寫差異或把空格當 1），不是 −18 / −12 / −6。
所以那張表描述的是「中等噪音下」的混淆，而且每個子音只保留 6 個 foil，§3 那些跨群、完全不混的對大多不在表裡——校準的「最不像」那端被截掉了（DECISIONS 19）。

## 5. 對刺激選擇的意思

- 要「清楚分得開」的聲音對：從 §3 的名單挑（跨 voicing / nasality / manner 的對，例如 m–s、n–ʃ、f–n、θ–m、p–z、k–z、b–s）。
- 要「容易混」的對：§2 的 f–θ、v–ð、d–g、p–k、t–k、p–t、z–ʒ、m–n、s–ʃ。
- 只差 voicing 的對（p–b、k–g…）在噪音裡算「分得開」，但 11-03 的純知覺測試裡它們就在「最不像」那層，d′ 只有 2.3；非母語者對 voicing 對比不敏感（中文沒有這個對比，只有送氣）。噪音資料不能直接當記憶／非母語的相似度，要用 `sound_perception_EN.py --axis features` 或直接拿 §3 的名單再測一次。

## 6. 來源（2026-11-09 子代理查核；網路只能用搜尋，除 M&N 外都只到摘要／書目層級）

**讀過原文、可引用**

- Miller, G. A., & Nicely, P. E. (1955). An analysis of perceptual confusions among some English consonants. *JASA, 27*(2), 338–352. doi:10.1121/1.1907526
  - p. 349：「voicing and nasality are much less affected by a random masking noise than are the other features. … Voicing and nasality are discriminable at signal-to-noise ratios as poor as −12 db whereas the place of articulation is hard to distinguish at ratios less than 6 db, a difference of some 18 db in efficiency.」
  - p. 348：「the features were perceived almost independently of one another.」
  - p. 352：「there are really just five distinguishable phonemes left: |ptk|, |fθsʃ|, |bdg|, |vðzʒ|, and |mn|.」
  - 注意：M&N 沒有逐字說「差的特徵越多越不會混」；他們說的是五個特徵近乎獨立的通道（p. 348–349），「距離越大越不混」是從這裡推出來的。

**只確認到書目／摘要**

- Wickelgren, W. A. (1966). Distinctive features and errors in short-term memory for English consonants. *JASA, 39*(2), 388–398. doi:10.1121/1.1909900 — STM 的那一半：子音在短期記憶裡「不是一個單位，而是一組可各自遺忘的特徵」（voicing、nasality、manner、place）。
- Wickelgren, W. A. (1965). Similarity and intrusions in short-term memory for consonant-vowel digrams. *QJEP, 17*, 241–246. doi:10.1080/17470216508416437 — 侵入錯誤的頻率是「與呈現項目相似度的單調遞增函數」。
- Conrad, R. (1964). Acoustic confusions in immediate memory. *Brit. J. Psychol., 55*, 75–84 — 記憶錯誤循聲音相似，但用的不是構音特徵。
- Wang, M. D., & Bilger, R. C. (1973). Consonant confusions in noise: A study of perceptual features. *JASA, 54*, 1248–1266 — 噪音與安靜、CV 與 VC；結論部分反對「固定的一組知覺特徵」，引用要小心。
- Shepard, R. N. (1972). Psychological representation of speech sounds. In David & Denes (Eds.), *Human Communication: A Unified View*. McGraw-Hill — 對 M&N 矩陣做 MDS，主軸是 voicing 與 nasality（頁碼未核）。
- Soli, S. D., & Arabie, P. (1979). Auditory versus phonetic accounts of observed confusions between consonant phonemes. *JASA, 66*, 46–59 — 重新分析 M&N，主張聲學屬性比音韻特徵更能解釋混淆。
- Phatak & Allen (2007) *JASA, 121*, 2312–2326；Phatak, Lovitt & Allen (2008) *JASA, 124*, 1220–1233 — 現代方法重做 M&N；子音依耐噪度分三群，不完全照 M&N 的特徵。
- **Cutler, Weber, Smits & Cooper (2004).** Patterns of English phoneme confusions by native and non-native listeners. *JASA, 116*, 3668–3678. doi:10.1121/1.1810292 — 荷蘭 vs 美國聽者，babble 0/8/16 dB。全文已讀（2026-11-10，`Cutler2004_全文查核.md`）：母語者 manner ≈ voicing > place；荷蘭聽者 manner > place > voicing（p. 3674），voicing 的流失集中在詞尾（荷語詞尾無濁音對比，p. 3671, 3675）；非母語 ≈ 母語的 80%，差距不隨 SNR 變。
- Flege (1989) *JASA, 86*, 1684–1697 — 中文母語者對英語詞尾 /t/–/d/ 的辨識。

**未確認**：中文母語者英語子音噪音混淆的逐特徵分析（只搜到 ICPhS 2023 的 Sheng et al. 與一張海報）；Shepard 1972 頁碼；Wang & Bilger、Soli & Arabie 的 DOI。

**結論**：「voicing / nasality 耐噪、place 先丟、塌成五群」直接出自 M&N p. 348–352；「差幾個特徵」這個量化是從 M&N 的獨立通道加 Wickelgren（1965, 1966）推出來的，沒有一篇原文逐字這樣說；對非母語者 Cutler et al. (2004) 指出 voicing 反而最弱。

## 7. 更正（2026-11-10 第二次查核：三個論述分開查）

我在 §3、§5 寫過「噪音裡分得開 ≠ 記憶裡分得開 ≠ 非母語者分得開」。查完文獻，三句只有第一句站得住：

**(1) 噪音裡 voicing 最耐、place 先丟 — 成立，且有重做。**
- M&N p. 349（見 §6）。注意 −12 dB 處 voicing 的相對資訊傳遞只剩約 52%（Table XXI），不是「幾乎全對」。
- Woods, D. L., Yund, E. W., Herron, T. J., & Ua Cruadhlaoich, M. A. I. (2010). Consonant identification in consonant-vowel-consonant syllables in speech-spectrum noise. *JASA, 127*(3), 1609–1623. https://doi.org/10.1121/1.3293005 — 摘要：「Single-feature place-of-articulation errors predominated at the highest SNR, while combined-feature errors predominated at the lowest SNR.」
- Wang & Bilger (1973) https://doi.org/10.1121/1.1914417 — 「Voicing and nasality are well perceived in the presence of masking noise, but their intelligibility drops relative to that of other features in quiet.」（排序不是跨情境固定的。）
- Phatak, Lovitt & Allen (2008) https://doi.org/10.1121/1.2913251 — 重做 M&N，型態相符，例外是擦音的 voicing 混淆不對稱（偏向答有聲）。

**(2) 「記憶裡只差 voicing 的對反而最難」— 沒有文獻支持，方向還可能相反。**
- Wickelgren (1966) 的摘要只說子音在 STM 裡是「一組可各自遺忘的特徵」；引用他的文獻說他的 STM 錯誤「大多落在同一 voicing 類別內」、「即使視覺呈現也反映 voicing 區分」——即記憶裡 voicing 也是被保留的特徵，和 M&N 一致，不是相反。
- Conrad (1964) https://doi.org/10.1111/j.2044-8295.1964.tb00899.x：「A highly significant correlation was found between letters which confused in the listening test, and letters which confused in recall.」——聽覺混淆和記憶混淆高度相關。
- 所以「噪音裡分得開 ≠ 記憶裡分得開」是我的推論，文獻傾向兩者相似。撤回。

**(3) 「非母語者（尤其中文母語）voicing 最弱」— 只有對比特定的證據，也有相反的。**
- Cutler et al. (2004) 全文（11-10 補讀）：「荷蘭聽者 voicing 最差」**屬實**（p. 3674），但由詞尾位置驅動（荷語詞尾無濁音對比）；p. 3674 同段也給母語者結果 manner ≈ voicing > place。
- Flege (1989) https://doi.org/10.1121/1.398599：中文母語者對詞尾 /t/–/d/ 未剪接的刺激辨識「接近母語者」，拿掉除阻爆發才變差——是線索特定的缺損，不是整體 voicing 缺損。
- Yang, Y., Yang, F., Porter, D. A., & Zhu, J. (2026). *Phonetica, 83*(1), 19–46. https://doi.org/10.1515/phon-2025-0029：中文母語者對 /s–z/、/θ–ð/（voicing）的區辨**優於** /s–θ/、/z–ð/（place）。與「voicing 最弱」相反。
- de Jong & Hao (2018) *J. Phonetics, 66*, 15–27 https://doi.org/10.1016/j.wocn.2017.09.002：台灣國語學習者的 voicing 正確率跨位置相關，詞尾例外。
- 所以「你分不出 p–b 是因為非母語」目前沒有文獻背書；中文母語者在詞首塞音與擦音的 voicing 上並不特別差。撤回為「待驗證」。

**對 11-03 資料的解讀要改**：純知覺 d′ 2.3、WM 裡掉到猜，不能歸給「voicing 對非母語者難」。剩下的候選：(a) 記憶負荷（2 + 2 項目）本身；(b) 「最不像」層混了不同類型的對，平均掉了；(c) 假警報 .28 的判準偏差；(d) 這套 wav 的個別 token 品質。`sound_perception_EN.py --axis features` 的結果能分 (b)；(a) 要靠減負荷的版本。

## 8. 修正後的敘述（2026-11-10，第三次查核後由子代理依三份查核結果寫成；只用查過的證據）

p–b、t–d、k–g 這些只差 voicing 的對，在噪音裡確實分得開：Miller & Nicely (1955, p. 349) 明言 voicing 與 nasality 在 −12 dB 仍可辨、place 在 6 dB 以下即難分；我們轉錄其 Table I–VI，三對的對稱混淆自 −12 dB 起皆 ≤ 1.2%，−6 dB 最混的 f–θ、p–k、t–k、d–g 全是只差 place 的對。但「混淆隨差異特徵數下降」不是 M&N 原話，而是從「特徵近乎獨立知覺」(p. 348) 推出的推論；Woods et al. (2010) 更指出低 SNR 時以多特徵錯誤為主——決定混淆的是「哪個特徵」，非特徵數。記憶層面：Wickelgren (1966) 顯示子音在短期記憶中以特徵集合編碼、各特徵半獨立遺忘；Klatt (1968) 重分析得出的特徵含 voiced 而無 place；Conrad (1964) 的聽覺／回憶相關只到字母層級。沒有研究把記憶混淆矩陣與 M&N 數值對照，「記憶裡的混淆結構等於知覺裡的」是推論。非母語：Cutler et al. (2004) 全文顯示荷蘭聽者 voicing 確實傳得最差 (p. 3674)，但那是荷語詞尾無濁音對比造成、集中在詞尾 (p. 3671, 3675)，不是非母語者的通性；華語母語者在 Flege (1989) 與 Yang et al. (2026) 中反而 voicing 辨識近母語、優於 place，只在特定線索被移除時崩潰。因此 11-03 單一受試者純知覺 d′ ≤ 2.3、虛報 .28，不能直接歸因於「非母語者 voicing 弱」（華語聽者的 voicing 缺口是推論，未測）；該軸「最不像」層平均僅差 2.4 個 M&N 特徵且混入 p–b、k–g 與 p–k 類對，軸的定義更可能是主因。WM 內降至猜測水準只表示此對比在目前負荷下量不到，不表示記憶未保留 voicing。

| 論述 | 證據等級 | 來源 | 一句話證據 |
|---|---|---|---|
| (a) 噪音中 voicing／nasality 耐噪，place 最先丟 | **已確立** | M&N 1955 (read)；Wang & Bilger 1973 (snippet)；Phatak et al. 2008 (abstract)；Woods et al. 2010 (abstract) | M&N p. 349；Table XXI −12 dB 相對資訊 voicing 52%、nasality 49%、place 4%。限制：CV + /ɑ/、白噪音、無安靜條件；Phatak：摩擦音 voicing 混淆不對稱。 |
| (b) 「差幾個特徵」決定混淆 | **推論**；「哪個特徵」才是已確立 | M&N (read)；Woods et al. 2010 (abstract) | M&N 只說特徵「幾乎獨立」；Woods：高 SNR 以單一 place 錯誤為主、低 SNR 以多特徵錯誤為主。 |
| (c) 記憶混淆結構＝知覺混淆結構 | **部分支持 + 推論**；無數值比較；**Schweppe 2011 全文同向、Roodenrys 2025 全文反向** | Wickelgren 1966、Klatt 1968、Conrad 1964、Hintzman 1967（abstract）；Schweppe et al. 2011 與 Roodenrys et al. 2025（全文，見 `Schweppe2011_全文查核.md`、`Roodenrys2025_全文查核.md`） | 子音在 STM 以特徵集合編碼、半獨立遺忘；Klatt 的特徵表含 voiced 無 place；Conrad 的相關只在字母層級；沒有記憶中特徵遺失的排序。Schweppe 2011 p. 266：只差 place 的 pa-ta-ka（M&N 最混的一簇）在任何呈現／回憶模式都傷回憶，差 manner 的 da-la-za 只口說時傷——與知覺結構同向，但作者歸因於特徵樹層級而非聲學混淆，並反對「數特徵」(p. 262, 264)。Roodenrys 2025 p. 254：干擾在視覺呈現下同樣出現，且依 Wang & Bilger 1973 只有約一半 minimal pair 聲學可混——作者明言效應來自構音特徵相似而非聲學相似。 |
| (d) 非母語者 voicing 最弱 | **荷蘭聽者屬實（全文）；對華語聽者仍是推論**，且 abstract 級證據方向相反 | Cutler et al. 2004（全文，`Cutler2004_全文查核.md`）；Flege 1989、Yang et al. 2026、de Jong & Hao 2018（abstract） | Cutler p. 3674：荷蘭聽者 voicing 傳遞最差、顯著低於 place；機制是 L1 詞尾無濁音對比（p. 3671, 3675），詞首劣勢小。華語任何位置都無英語式 voicing，預測詞首也弱——但 Flege：華語系聽者詞尾 /t–d/ 近母語；Yang：/s–z/、/θ–ð/ 優於 /s–θ/、/z–ð/。 |
| (e) 11-03 資料 | **一位受試者的描述性資料**；歸因是推論 | 本研究 | 可說：純知覺各層 .50–1.00、d′ 0.6–2.3、虛報 .28；WM 內 .40–.67、虛報 .22。不可說：非母語者 voicing 弱、記憶不保留 voicing。 |

補充書目（§6 未列者）：Klatt, D. H. (1968). Structure of confusions in short-term memory between English consonants. *JASA, 44*(2), 401–407. https://doi.org/10.1121/1.1911094 [abstract]；Hintzman, D. L. (1967). Articulatory coding in short-term memory. *JVLVB, 6*(3), 312–316 [abstract]；Roodenrys, S., Miller, L. M., Guitard, D., & Neath, I. (2025). Similar phonemes create interference in the serial recall task. *Memory, 33*(2), 248–258. https://doi.org/10.1080/09658211.2024.2433049 [全文已讀 2026-11-10；相似＝差 1 個構音特徵（voice/place/manner），36/60 對為 voicing]；Pisoni, D. B. (1973). Auditory and phonetic memory codes in the discrimination of consonants and vowels. *Perception & Psychophysics, 13*(2), 253–260. https://doi.org/10.3758/BF03214136 [abstract；只測 /ba–pa/ 一條 voicing 連續體]。Yang et al. (2026) 的篇名未核對，引用前要補。
