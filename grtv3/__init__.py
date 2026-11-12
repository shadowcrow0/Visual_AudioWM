"""GRTv3_ada 的模組化版本（放棄 PsychoPy Builder 相容；流程、參數、資料欄位與 GRTv3_ada_EN.py 相同）。

    grtv3/config.py       所有常數（試次數、時序、幾何、Psi 設定、聲音）
    grtv3/text_en.py      受試者看到的所有文字
    grtv3/plan.py         純邏輯：試次計畫、版面、計分、休息訊息（不需要 PsychoPy，有測試）
    grtv3/colour.py       顏色 LUT（弧長 dE00 → hex）
    grtv3/masking.py      遮蔽噪音（snr_audio）
    grtv3/calibration.py  兩個 Psi（AGRT）的包裝：提議水準、更新、收尾估計
    grtv3/display.py      PsychoPy：視窗、刺激物件、通用的「時間軸」routine 執行器
    grtv3/session.py      把以上接起來：對話框 → 資料檔 → 校準 → 練習 → 主實驗

入口：python GRTv3_ada_modular.py [--pilot] [--audio-device NAME] [--require-headphones]
"""
__version__ = "0.1"
