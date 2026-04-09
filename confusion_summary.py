import pandas as pd
import numpy as np

def load_confusion_matrix(csv_path):
    """讀取混淆矩陣"""
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.str.replace('→', '')
    return df

def analyze_all_tables():
    """分析三個 SNR 條件的混淆"""
    tables = {
        '-6dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_III_neg6dB.csv"),
        '0dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_IV_0dB.csv"),
        '+6dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_V_pos6dB.csv"),
    }

    sounds = list(tables['-6dB'].index)

    print("=" * 90)
    print("每個音的混淆分析 (三個 SNR 條件加總)")
    print("=" * 90)

    for sound in sounds:
        print(f"\n{'='*90}")
        print(f"【{sound}】")
        print("-" * 90)

        # 計算每個混淆對象在三個條件下的總次數
        confusion_totals = {}
        for other in sounds:
            if other == sound:
                continue
            total = sum(tables[snr].loc[sound, other] for snr in tables)
            confusion_totals[other] = {
                'total': total,
                '-6dB': int(tables['-6dB'].loc[sound, other]),
                '0dB': int(tables['0dB'].loc[sound, other]),
                '+6dB': int(tables['+6dB'].loc[sound, other]),
            }

        # 排序
        sorted_conf = sorted(confusion_totals.items(), key=lambda x: x[1]['total'], reverse=True)

        # Top 3 最容易混淆
        print(f"\n  🔴 最容易混淆 (Top 3):")
        print(f"  {'對象':<8} {'總計':<8} {'-6dB':<8} {'0dB':<8} {'+6dB':<8}")
        print(f"  {'-'*40}")
        for i, (conf_sound, data) in enumerate(sorted_conf[:3]):
            if data['total'] > 0:
                print(f"  {conf_sound:<8} {data['total']:<8} {data['-6dB']:<8} {data['0dB']:<8} {data['+6dB']:<8}")

        # Top 3 最不容易混淆 (排除全為 0 的)
        print(f"\n  🟢 最不容易混淆 (Bottom 3, 排除 0):")
        print(f"  {'對象':<8} {'總計':<8} {'-6dB':<8} {'0dB':<8} {'+6dB':<8}")
        print(f"  {'-'*40}")

        # 過濾掉 total=0 的，然後取最小的
        non_zero = [(s, d) for s, d in sorted_conf if d['total'] > 0]
        bottom_3 = non_zero[-3:] if len(non_zero) >= 3 else non_zero
        bottom_3.reverse()  # 從最小開始

        for conf_sound, data in bottom_3:
            print(f"  {conf_sound:<8} {data['total']:<8} {data['-6dB']:<8} {data['0dB']:<8} {data['+6dB']:<8}")

        # 正確率
        correct_total = sum(tables[snr].loc[sound, sound] for snr in tables)
        total_responses = sum(tables[snr].loc[sound].sum() for snr in tables)
        print(f"\n  📊 正確率: {correct_total}/{total_responses} ({correct_total/total_responses*100:.1f}%)")


def create_summary_table():
    """建立總結表格"""
    tables = {
        '-6dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_III_neg6dB.csv"),
        '0dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_IV_0dB.csv"),
        '+6dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_V_pos6dB.csv"),
    }

    sounds = list(tables['-6dB'].index)

    print("\n\n")
    print("=" * 100)
    print("簡易總結表 (三個 SNR 加總)")
    print("=" * 100)
    print(f"{'音':<6} {'正確率':<10} {'#1 混淆':<20} {'#2 混淆':<20} {'#3 混淆':<20}")
    print("-" * 100)

    for sound in sounds:
        # 計算正確率
        correct_total = sum(tables[snr].loc[sound, sound] for snr in tables)
        total_responses = sum(tables[snr].loc[sound].sum() for snr in tables)
        acc = correct_total / total_responses * 100

        # 計算混淆
        confusion_totals = {}
        for other in sounds:
            if other == sound:
                continue
            total = sum(tables[snr].loc[sound, other] for snr in tables)
            confusion_totals[other] = total

        sorted_conf = sorted(confusion_totals.items(), key=lambda x: x[1], reverse=True)

        top3 = []
        for conf_sound, count in sorted_conf[:3]:
            if count > 0:
                top3.append(f"{conf_sound}({count})")
            else:
                top3.append("-")

        while len(top3) < 3:
            top3.append("-")

        print(f"{sound:<6} {acc:>6.1f}%   {top3[0]:<20} {top3[1]:<20} {top3[2]:<20}")


def export_confusion_csv(output_path):
    """
    輸出 CSV：
    - 第一欄: 音
    - 第二欄: confused (0=容易混淆, 1=不容易混淆)
    - 第三欄: 混淆對象
    - 第四欄: 總次數
    """
    tables = {
        '-6dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_III_neg6dB.csv"),
        '0dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_IV_0dB.csv"),
        '+6dB': load_confusion_matrix("/mnt/c/Users/spt904/Dropbox/Spring2026/table_V_pos6dB.csv"),
    }

    sounds = list(tables['-6dB'].index)
    rows = []

    for sound in sounds:
        # 計算混淆總次數
        confusion_totals = {}
        for other in sounds:
            if other == sound:
                continue
            total = sum(tables[snr].loc[sound, other] for snr in tables)
            confusion_totals[other] = total

        sorted_conf = sorted(confusion_totals.items(), key=lambda x: x[1], reverse=True)

        # Top 3 容易混淆 (confused=0)
        for conf_sound, count in sorted_conf[:3]:
            if count > 0:
                rows.append([sound, 0, conf_sound, count])

        # Top 3 不容易混淆 (confused=1), 排除 0
        non_zero = [(s, c) for s, c in sorted_conf if c > 0]
        bottom_3 = non_zero[-3:] if len(non_zero) >= 3 else non_zero
        for conf_sound, count in bottom_3:
            rows.append([sound, 1, conf_sound, count])

    # 存成 CSV
    df = pd.DataFrame(rows, columns=['sound', 'confused', 'target', 'count'])
    df.to_csv(output_path, index=False)
    print(f"已存檔: {output_path}")
    return df


if __name__ == "__main__":
    analyze_all_tables()
    create_summary_table()

    # 輸出 CSV
    export_confusion_csv("/home/yyc/symmetry/.venv/vwm_awm/Visual_AudioWM/confusion_analysis.csv")
