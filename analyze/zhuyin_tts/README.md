# ㄅㄛ 對 ㄨㄛ：用合成語音與剪接檢查「為什麼寫 ㄅㄛ 不寫 ㄅㄨㄛ」

這個資料夾跟實驗本身無關，是注音音韻問題的副題；放在這裡只是為了能下載到另一台機器執行。
不含音檔、不含受試者資料。合成用 OpenFormosa 的 BlueMagpie-TTS（臺灣口音，內附已授權的聲音）。

| 程式 | 做什麼 | 寫出 |
|---|---|---|
| `synth.py` | 以「這個字唸，X。」合成目標字，每個字多個種子 | `out/*.wav`、`out/log.json` |
| `verify.py` | 語音辨識器核對句尾是不是目標音節 | `out/verified.json` |
| `measure.py` | 切出目標音節，量 0.2-1.5 kHz 能量重心 | `out/measures.json` |
| `figs.py` | ㄅㄛ 對 ㄉㄨㄛ / ㄍㄨㄛ、ㄉㄨㄛ 對 ㄉㄨ、分組長條圖 | `out/figs/figT1-T4` |
| `splice.py` | 在 ㄅ 和 ㄛ 中間剪進 40 / 80 / 120 ms 的 ㄨ（另有插入 ㄛ 的對照） | `out/splice/*.wav`、`splice_log.json` |
| `splice_check.py` | 辨識器聽剪接檔，量重心曲線 | `out/splice/splice_results.json` |
| `figs_splice.py` | 剪接前後對照、ㄍㄜ 對 ㄍㄨㄛ、ㄅㄛ 落在哪一邊 | `out/figs/figS1-S5` |
| `listen.py` | 終端機聽測：隨機播放剪接檔，逐題作答 | `out/splice/listen_responses.csv` |
| `send_back.py` | 把結果小檔收到一個資料夾 | `out/send_back/` |

`run_all.sh` 跑前四支；`run_splice.sh` 跑全部（聽測除外）。

## 執行（Arch Linux）

```bash
sudo pacman -S --needed git uv libsndfile wqy-zenhei
git clone https://github.com/OpenFormosa/BlueMagpie-TTS
uv venv --python 3.12 venv            # 套件要求 Python 3.10-3.12
source venv/bin/activate
uv pip install -e ./BlueMagpie-TTS soundfile faster-whisper pypinyin praat-parselmouth scipy matplotlib
export BLUEMAGPIE_DIR=~/zhuyin_tts/model   # 已下載的模型資料夾；不設則 synth.py 自己從 Hugging Face 下載（約 8 GB）
./run_splice.sh out 8
python listen.py --out out
python send_back.py --out out
```

## 驗證範圍

在 Linux 容器（Python 3.12、CPU）從空資料夾跑通；同一個種子在 CPU 上重跑的音檔相同。
Arch、GPU、實際播放沒有測過（`listen.py` 只用模擬輸入測過）。

## 限制

- 單一合成聲音；量到的是模型怎麼唸，不是人怎麼唸。
- 單獨輸入一個字或注音，模型會唸出不相干的音節，所以「ㄅㄨㄛ」只能用剪接造出來。
- 剪接檔在接點有音高跳動；插入的 ㄨ 取自 ㄉㄨ / ㄍㄨ / ㄈㄨ / ㄆㄨ，不是 ㄅ 後面自然的 ㄨ。
- 以 ㄇ、ㄨ 或母音開頭的字切不出起點，預設清單沒有放。
