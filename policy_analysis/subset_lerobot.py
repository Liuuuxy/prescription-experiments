"""Make an N-episode prefix subset of a LeRobot v2.1 dataset (for the N-sweep).

Keeps episodes 0..N-1: per-episode parquet/mp4/extras are copied, episodes.jsonl
lines truncated, info.json totals updated. Keeps the PARENT's stats.json so all
subsets share identical normalization (deliberate: consistent across the
N=25/50/100 sweep, and prefix subsetting keeps global frame indices contiguous).

Usage: python subset_lerobot.py --src <lerobot_dir> --n 25 [--dst <dir>]
"""
import argparse
import json
import os
import shutil


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
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dst", default=None)
    a = ap.parse_args()
    src = a.src.rstrip("/")
    dst = a.dst or f"{src}_N{a.n}"
    assert not os.path.exists(dst), f"{dst} exists"

    eps = read_jsonl(os.path.join(src, "meta", "episodes.jsonl"))
    assert a.n <= len(eps), f"asked N={a.n} but only {len(eps)} episodes"
    keep = eps[: a.n]
    total_frames = sum(e["length"] for e in keep)

    os.makedirs(os.path.join(dst, "meta"))
    # meta: episodes(.jsonl,_stats.jsonl) truncated; others copied verbatim
    write_jsonl(os.path.join(dst, "meta", "episodes.jsonl"), keep)
    es = read_jsonl(os.path.join(src, "meta", "episodes_stats.jsonl"))
    write_jsonl(os.path.join(dst, "meta", "episodes_stats.jsonl"), es[: a.n])
    for f in os.listdir(os.path.join(src, "meta")):
        if f not in ("episodes.jsonl", "episodes_stats.jsonl", "info.json"):
            shutil.copy2(os.path.join(src, "meta", f), os.path.join(dst, "meta", f))
    info = json.load(open(os.path.join(src, "meta", "info.json")))
    n_video_keys = sum(1 for k in info.get("features", {}) if info["features"][k].get("dtype") == "video")
    info.update(total_episodes=a.n, total_frames=total_frames,
                total_videos=a.n * max(n_video_keys, 1))
    json.dump(info, open(os.path.join(dst, "meta", "info.json"), "w"), indent=4)

    # per-episode payloads (chunk 0 layout; chunks_size=1000 covers N<=1000)
    for i in range(a.n):
        ep = f"episode_{i:06d}"
        d = os.path.join(dst, "data", "chunk-000")
        os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(src, "data", "chunk-000", ep + ".parquet"), d)
        for vk in [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]:
            vd = os.path.join(dst, "videos", "chunk-000", vk)
            os.makedirs(vd, exist_ok=True)
            shutil.copy2(os.path.join(src, "videos", "chunk-000", vk, ep + ".mp4"), vd)
        se = os.path.join(src, "extras", ep)
        if os.path.isdir(se):
            shutil.copytree(se, os.path.join(dst, "extras", ep))
    for f in ("extras/dataset_meta.json",):
        if os.path.exists(os.path.join(src, f)):
            os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True)
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    print(f"Wrote {dst}: {a.n} episodes, {total_frames} frames")


if __name__ == "__main__":
    main()
