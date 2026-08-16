import csv
import os

def fetch_matches_from_csv(filepath="data/matches.csv"):
    """
    从 CSV 文件读取比赛数据。
    CSV 格式：
    match_id,league,home_team,away_team,had_win,had_draw,had_lose
    周一001,英超,曼城,阿森纳,1.85,3.60,4.10
    """
    matches = []
    if not os.path.exists(filepath):
        print(f"文件 {filepath} 不存在")
        return matches

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matches.append({
                "match_id": row["match_id"].strip(),
                "league": row["league"].strip(),
                "home_team": row["home_team"].strip(),
                "away_team": row["away_team"].strip(),
                "had_odds": {
                    "win": float(row["had_win"]),
                    "draw": float(row["had_draw"]),
                    "lose": float(row["had_lose"]),
                }
            })
    return matches

if __name__ == "__main__":
    matches = fetch_matches_from_csv()
    print(f"读取到 {len(matches)} 场比赛")
    for m in matches:
        print(m)
