import subprocess

# 直接播放
subprocess.run(["espeak-ng", "-v", "en", "apa"])

# 文字轉 IPA
result = subprocess.run(
    ["espeak-ng", "--ipa", "-q", "apa"],
    capture_output=True, text=True
)
print(result.stdout)  # → ˈæpə

# 存成 WAV
subprocess.run(["espeak-ng", "-v", "en", "-w", "out.wav", "apa"])