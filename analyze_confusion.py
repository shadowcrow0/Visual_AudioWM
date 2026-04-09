import pandas as pd
import numpy as np

def analyze_confusion(csv_path, top_n=3):
    """
    分析混淆矩陣，找出每個音最容易與哪些音混淆

    Parameters:
    -----------
    csv_path : str
        混淆矩陣 CSV 檔案路徑
    top_n : int
        顯示前 N 個最常混淆的音

    Returns:
    --------
    dict : 每個音的混淆分析結果
    """
    # 讀取 CSV
    df = pd.read_csv(csv_path, index_col=0)

    # 清理 index（移除 "→" 符號）
    df.index = df.index.str.replace('→', '')

    results = {}

    print("=" * 60)
    print(f"混淆矩陣分析 (Top {top_n} 混淆)")
    print("=" * 60)

    for sound in df.index:
        row = df.loc[sound]
        total = row.sum()
        correct = row[sound]  # 對角線 = 正確次數
        accuracy = correct / total * 100

        # 排除自己，找出最常混淆的音
        confusions = row.drop(sound).sort_values(ascending=False)
        top_confusions = confusions.head(top_n)

        # 過濾掉 0 的項目
        top_confusions = top_confusions[top_confusions > 0]

        results[sound] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'confusions': [(c, int(top_confusions[c]), top_confusions[c]/total*100)
                          for c in top_confusions.index]
        }

        # 印出結果
        print(f"\n[{sound}] 正確率: {accuracy:.1f}% ({correct}/{total})")
        if top_confusions.empty:
            print("   → 無明顯混淆")
        else:
            for conf_sound in top_confusions.index:
                count = top_confusions[conf_sound]
                pct = count / total * 100
                print(f"   → 混淆成 [{conf_sound}]: {int(count)} 次 ({pct:.1f}%)")

    return results


def find_symmetric_confusions(csv_path, threshold=10):
    """
    找出雙向混淆的音對（A 混成 B 且 B 也混成 A）

    Parameters:
    -----------
    csv_path : str
        混淆矩陣 CSV 檔案路徑
    threshold : int
        混淆次數門檻
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.str.replace('→', '')

    sounds = list(df.index)
    pairs = []

    print("\n" + "=" * 60)
    print(f"雙向混淆配對 (門檻 >= {threshold} 次)")
    print("=" * 60)

    for i, s1 in enumerate(sounds):
        for s2 in sounds[i+1:]:
            conf_1to2 = df.loc[s1, s2]  # s1 被聽成 s2
            conf_2to1 = df.loc[s2, s1]  # s2 被聽成 s1

            if conf_1to2 >= threshold and conf_2to1 >= threshold:
                pairs.append((s1, s2, conf_1to2, conf_2to1))
                print(f"[{s1}] ↔ [{s2}]: {s1}→{s2}={int(conf_1to2)}, {s2}→{s1}={int(conf_2to1)}")

    return pairs


if __name__ == "__main__":
    csv_path = "/mnt/c/Users/spt904/Dropbox/Spring2026/table_III_neg6dB.csv"

    # 分析每個音的混淆
    results = analyze_confusion(csv_path, top_n=3)

    # 找出雙向混淆
    pairs = find_symmetric_confusions(csv_path, threshold=10)
