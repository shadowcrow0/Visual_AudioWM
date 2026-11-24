# 從頭重來：ㄅㄛ / ㄅㄨㄛ 合成與剪接（Arch Linux）

每一步都可以重複執行：已完成的部分會被跳過或續傳，所以斷線後從第 0 步照順序再跑一次即可。

## 0. 先開 tmux，斷線也不會中斷

```bash
sudo pacman -S --needed tmux
tmux new -A -s zhuyin
```

斷線後重新連上，再打一次 `tmux new -A -s zhuyin` 就回到原本的畫面，程式還在跑。
以下所有指令都在 tmux 裡面下。

## 1. 系統套件

```bash
sudo pacman -S --needed git uv curl libsndfile wqy-zenhei
```

## 2. 取得程式

家目錄本身就是 Backup 的 repo（這台機器的情況）：

```bash
mkdir -p /tmp/old_paste
cd ~/zhuyin_tts 2>/dev/null && mv -f figs.py measure.py run_splice.sh splice.py splice_check.py synth.py verify.py figs_splice.py listen.py send_back.py run_all.sh /tmp/old_paste/ 2>/dev/null
cd ~ && git pull
ls ~/zhuyin_tts
```

`mv` 那行是把之前手動貼上的舊檔移開，否則 `git pull` 會拒絕覆蓋；檔案不存在時會安靜略過。

全新的機器（還沒有 Backup）：

```bash
git clone git@github.com:shadowcrow0/Backup.git ~/Backup
mkdir -p ~/zhuyin_tts && cp ~/Backup/zhuyin_tts/* ~/zhuyin_tts/
```

確認：`ls ~/zhuyin_tts` 應該有
`synth.py verify.py measure.py figs.py splice.py splice_check.py figs_splice.py listen.py send_back.py run_all.sh run_splice.sh`。

## 3. 下載模型（約 8 GB，可續傳）

```bash
mkdir -p ~/zhuyin_tts/model/checkpoints && cd ~/zhuyin_tts/model
for f in config.json tokenizer.json tokenizer_config.json \
         checkpoints/speaker_centroids.pt checkpoints/hung_yi_lee_speaker_centroids.pt \
         audiovae.pth pytorch_model.bin; do
  curl -L --fail -C - -o "$f" "https://huggingface.co/OpenFormosa/BlueMagpie-TTS/resolve/main/$f"
done
ls -l pytorch_model.bin
```

`pytorch_model.bin` 完整大小是 `7800079543`。數字不對就再跑一次同一段（會從斷掉的地方接續）。
已經下載完整時，curl 會對每個檔顯示一行錯誤（HTTP 416，表示沒有東西可續傳），可以忽略。

## 4. Python 環境

```bash
cd ~/zhuyin_tts
[ -d BlueMagpie-TTS ] || git clone https://github.com/OpenFormosa/BlueMagpie-TTS
[ -d venv ] || uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -e ./BlueMagpie-TTS soundfile faster-whisper pypinyin praat-parselmouth scipy matplotlib
python -c "import torch, bluemagpie; print('ok', torch.__version__)"
```

最後一行印出 `ok` 加版本號就表示環境好了（前面會有幾行 FutureWarning，不是錯誤）。
如果 `uv venv` 問要不要取代既有的環境，答 `n`。

## 5. 合成、核對、量測、剪接、畫圖

```bash
cd ~/zhuyin_tts
source venv/bin/activate
export BLUEMAGPIE_DIR=~/zhuyin_tts/model
chmod +x run_all.sh run_splice.sh
./run_splice.sh out 8
```

- 載入模型時有一小段時間不會印東西；之後每合成一個音檔印一行。
- 中斷後重跑同一個指令：已經合成的音檔不會重做。
- 跑完會有 `out/figs/`（九張圖）和 `out/splice/`（剪接音檔）。

只想先試少量：

```bash
python synth.py --out out_test --seeds 2 --items bo_撥 duo_多 guo_鍋 du_督
```

## 6. 聽測（選做，但這是最重要的一筆資料）

```bash
cd ~/zhuyin_tts && source venv/bin/activate
python listen.py --out out
```

每題聽完輸入 1-4 再按 Enter；`r` 再播一次，`q` 結束（已作答的都會存下來）。
畫面上的選項是英文，意思如下：

- 1 = 一個音節，像「撥」（ㄅㄛ）
- 2 = 一個音節，ㄛ 前面多了 ㄨ（ㄅㄨㄛ）
- 3 = 兩個音節（像「不喔」）
- 4 = 其他

沒有聲音：需要 `pw-play`、`paplay`、`aplay`、`ffplay` 其中之一（`sudo pacman -S --needed pipewire` 或 `alsa-utils` 或 `ffmpeg`）。
透過 SSH 連線時聲音不會傳到你這邊，要在那台機器本機上做這一步。

## 7. 收集要傳回的檔案

```bash
cd ~/zhuyin_tts && source venv/bin/activate
python send_back.py --out out
ls out/send_back
```

把 `out/send_back/` 裡的檔案傳回來（十幾個小檔，不含音檔）。

## 出問題時

| 現象 | 原因與處理 |
|---|---|
| `can't open file ... synth.py` | 人不在 `~/zhuyin_tts`。先 `cd ~/zhuyin_tts` |
| `No module named 'bluemagpie'` | 沒有啟用環境。先 `source venv/bin/activate` |
| 只印三行 FutureWarning 就沒動靜 | 正在載入模型，等它印 `model loaded` |
| 出現 `Killed` | 記憶體不夠（模型約佔 8 GB，機器至少要 12 GB）。看 `free -h` |
| `git pull` 說未追蹤的檔案會被覆蓋 | 重跑第 2 步的 `mv` 那行再 `git pull` |
| 圖上中文變方框 | `rm -rf ~/.cache/matplotlib` 後重跑 `python figs.py --out out && python figs_splice.py --out out` |
| 又把模型下載一次 | 忘了 `export BLUEMAGPIE_DIR=~/zhuyin_tts/model` |
