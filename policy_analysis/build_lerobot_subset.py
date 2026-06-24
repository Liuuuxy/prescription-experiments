"""Build a LeRobot v2.1 subset from an ARBITRARY list of episode indices
(unlike subset_lerobot.py which is prefix-only). Per-episode parquet/mp4/extras
are copied and RE-INDEXED to contiguous episode ids + contiguous global frame
indices, so the result is a standalone, valid LeRobot dataset that openpi's
robocasa data_dirs loader can consume.

  python policy_analysis/build_lerobot_subset.py --src <mg lerobot> \
      --arms weakregion/arms.json --which base+core --dst <out dir>
"""
import argparse
import json
import os
import shutil

import pandas as pd


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--which", required=True,
                    help="'+'-joined arm keys, e.g. base+core or base+random")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src = a.src.rstrip("/")
    assert not os.path.exists(a.dst), f"{a.dst} exists"

    arms = json.load(open(a.arms))
    selected = []
    for key in a.which.split("+"):
        selected += arms[f"{key}_episodes"]
    # keep order stable but de-dup (base disjoint from arms, so no dups expected)
    seen = set(); selected = [e for e in selected if not (e in seen or seen.add(e))]

    info = json.load(open(os.path.join(src, "meta", "info.json")))
    video_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
    csize = info.get("chunks_size", 1000)  # src episodes are in chunk-{e//csize:03d}
    eps_by_idx = {l["episode_index"]: l for l in read_jsonl(os.path.join(src, "meta", "episodes.jsonl"))}
    es_path = os.path.join(src, "meta", "episodes_stats.jsonl")
    es_by_idx = ({l["episode_index"]: l for l in read_jsonl(es_path)} if os.path.exists(es_path) else {})

    os.makedirs(os.path.join(a.dst, "meta"))
    os.makedirs(os.path.join(a.dst, "data", "chunk-000"))
    for vk in video_keys:
        os.makedirs(os.path.join(a.dst, "videos", "chunk-000", vk))

    new_eps, new_es = [], []
    offset = 0
    for j, e in enumerate(selected):
        sc = f"chunk-{e // csize:03d}"  # source chunk for this episode
        df = pd.read_parquet(os.path.join(src, "data", sc, f"episode_{e:06d}.parquet"))
        L = len(df)
        df["episode_index"] = j
        df["index"] = range(offset, offset + L)
        df.to_parquet(os.path.join(a.dst, "data", "chunk-000", f"episode_{j:06d}.parquet"))
        offset += L
        for vk in video_keys:
            shutil.copy2(os.path.join(src, "videos", sc, vk, f"episode_{e:06d}.mp4"),
                         os.path.join(a.dst, "videos", "chunk-000", vk, f"episode_{j:06d}.mp4"))
        se = os.path.join(src, "extras", f"episode_{e:06d}")
        if os.path.isdir(se):
            shutil.copytree(se, os.path.join(a.dst, "extras", f"episode_{j:06d}"))
        epl = dict(eps_by_idx[e]); epl["episode_index"] = j; new_eps.append(epl)
        if e in es_by_idx:
            esl = dict(es_by_idx[e]); esl["episode_index"] = j; new_es.append(esl)
        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{len(selected)} episodes", flush=True)

    write_jsonl(os.path.join(a.dst, "meta", "episodes.jsonl"), new_eps)
    if new_es:
        write_jsonl(os.path.join(a.dst, "meta", "episodes_stats.jsonl"), new_es)
    for f in os.listdir(os.path.join(src, "meta")):
        if f not in ("episodes.jsonl", "episodes_stats.jsonl", "info.json"):
            shutil.copy2(os.path.join(src, "meta", f), os.path.join(a.dst, "meta", f))
    info.update(total_episodes=len(selected), total_frames=offset,
                total_videos=len(selected) * max(len(video_keys), 1), splits={"train": f"0:{len(selected)}"})
    json.dump(info, open(os.path.join(a.dst, "meta", "info.json"), "w"), indent=4)
    print(f"Wrote {a.dst}: {len(selected)} episodes, {offset} frames ({a.which})")


if __name__ == "__main__":
    main()
