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
- **Cutler, Weber, Smits & Cooper (2004).** Patterns of English phoneme confusions by native and non-native listeners. *JASA, 116*, 3668–3678. doi:10.1121/1.1810292 — 荷蘭 vs 美國聽者。摘要片段：對非母語者 manner 傳得最好，**voicing 傳得最差**。這和本文 §3「voicing 在噪音裡最耐」相反，但和 11-03 非母語受試者的結果一致。引用前要讀全文。
- Flege (1989) *JASA, 86*, 1684–1697 — 中文母語者對英語詞尾 /t/–/d/ 的辨識。

**未確認**：中文母語者英語子音噪音混淆的逐特徵分析（只搜到 ICPhS 2023 的 Sheng et al. 與一張海報）；Shepard 1972 頁碼；Wang & Bilger、Soli & Arabie 的 DOI。

**結論**：「voicing / nasality 耐噪、place 先丟、塌成五群」直接出自 M&N p. 348–352；「差幾個特徵」這個量化是從 M&N 的獨立通道加 Wickelgren（1965, 1966）推出來的，沒有一篇原文逐字這樣說；對非母語者 Cutler et al. (2004) 指出 voicing 反而最弱。
