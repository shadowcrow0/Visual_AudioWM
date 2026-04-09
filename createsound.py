import subprocess

# 直接播放
subprocess.run(["espeak-ng", "-v", "en", "asha"])

# 文字轉 IPA
result = subprocess.run(
    ["espeak-ng", "--ipa", "-q", "asha"],
    capture_output=True, text=True
)
print(result.stdout)  # → ˈæʃə

# 存成 WAV
subprocess.run(["espeak-ng", "-v", "en", "-w", "asha.wav", "asha"])