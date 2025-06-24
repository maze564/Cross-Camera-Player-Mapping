import json
from scipy.spatial.distance import cosine

def load_features(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def match_players(broadcast, tacticam, threshold=0.4):
    mapping = {}
    for tid, tvec in tacticam.items():
        best_match = None
        best_score = float('inf')
        for bid, bvec in broadcast.items():
            score = cosine(tvec, bvec)
            if score < best_score and score < threshold:
                best_score = score
                best_match = bid
        mapping[tid] = best_match if best_match else "no match"
    return mapping

if __name__ == "__main__":
    broadcast = load_features("outputs/broadcast.mp4_features.jsson")
    tacticam = load_features("outputs/tacticam.mp4_features.json")

    result = match_players(broadcast, tacticam)
    print("\nMatched Player IDs from Tacticam to Broadcast:\n")
    for k, v in result.items():
        print(f"Tacticam ID {k} → Broadcast ID {v}")

    with open("outputs/player_id_mapping.json", "w") as f:
        json.dump(result, f, indent=2)
