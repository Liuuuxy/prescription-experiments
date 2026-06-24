"""Method B: failure-conditioned retrieval scorer (pluggable).

Selects pool demos most VISUALLY similar to the states pi0 actually fails in --
going beyond the per-category heuristic. Pipeline:
  1. failed seeds  := scenes pi0 failed on (from the n=500 student eval)
  2. render each failed scene's agentview frame (gym.make at that seed) -> CLIP embed
  3. embed each pool demo's first agentview frame -> CLIP embed
  4. score(demo) = mean of its top-k cosine sims to the failure set
  5. select top-N -> a core-episode list (feeds build_arms.py --core_file)

Uses the cached CLIP (openai/clip-vit-large-patch14) as the encoder; no FAISS needed.
Run in the robocasa env, MUJOCO_GL=egl.

  MUJOCO_GL=egl TRANSFORMERS_OFFLINE=1 python policy_analysis/score_retrieval.py \
    --student weakregion/pi0_student_n500/weakregion.json --mg_dir <mg lerobot> \
    --n_select 200 --out weakregion/retrieval_core.json
"""
import argparse
import json
import numpy as np
import torch
import gymnasium as gym
import robocasa  # noqa: F401
from transformers import CLIPModel, CLIPProcessor
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAM = "robot0_agentview_left"
CLIP_ID = "openai/clip-vit-base-patch32"


def to_uint8_hwc(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[-1]:  # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if float(a.max()) <= 1.0 else a.astype(np.uint8)
    return a


@torch.no_grad()
def clip_embed(model, proc, imgs, device, bs=64, tag=""):
    out = []
    for i in range(0, len(imgs), bs):
        batch = [to_uint8_hwc(x) for x in imgs[i:i + bs]]
        inp = proc(images=batch, return_tensors="pt").to(device)
        pooled = model.vision_model(pixel_values=inp["pixel_values"]).pooler_output
        e = model.visual_projection(pooled)
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        if tag and (i // bs) % 10 == 0:
            print(f"  {tag}: {min(i+bs,len(imgs))}/{len(imgs)}", flush=True)
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--mg_dir", required=True)
    p.add_argument("--task", default="PickPlaceCounterToSink")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--n_select", type=int, default=200)
    p.add_argument("--topk", type=int, default=5, help="avg over top-k failure neighbours")
    p.add_argument("--out", default="weakregion/retrieval_core.json")
    p.add_argument("-d", "--device", default="cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)

    model = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_ID)

    # 1+2) failure scenes -> render -> embed
    eps = json.load(open(args.student))["episodes"]
    failed = [e["episode"] for e in eps if not e["success"]]
    print(f"pi0 failures: {len(failed)} scenes -> rendering", flush=True)
    fail_frames = []
    for s in failed:
        env = gym.make(f"robocasa/{args.task}", split=args.split, seed=s)
        obs, _ = env.reset()
        fail_frames.append(np.asarray(obs[f"video.{CAM}"]))
        env.close()
    Fe = clip_embed(model, proc, fail_frames, device, tag="failure-embed")

    # 3) pool first-frame embeddings
    ds = LeRobotDataset("x", root=args.mg_dir)
    starts = ds.episode_data_index["from"].tolist()
    n_pool = ds.num_episodes
    print(f"pool: {n_pool} demos -> decoding+embedding first frames", flush=True)
    pool_frames = [ds[starts[e]][f"observation.images.{CAM}"] for e in range(n_pool)]
    Pe = clip_embed(model, proc, pool_frames, device, tag="pool-embed")

    # 4) score = mean top-k cosine sim to failure set
    sims = Pe @ Fe.T  # (n_pool, n_fail), both unit-norm
    score = sims.topk(min(args.topk, Fe.shape[0]), dim=1).values.mean(dim=1).numpy()

    # 5) select top-N
    order = np.argsort(-score)
    selected = sorted(int(i) for i in order[:args.n_select])
    json.dump({"method": "failure_retrieval", "n_select": args.n_select, "topk": args.topk,
               "core_episodes": selected,
               "score_range": [float(score.min()), float(score.max())]},
              open(args.out, "w"), indent=2)
    print(f"\nselected {len(selected)} demos -> {args.out}")
    print(f"score range: [{score.min():.3f}, {score.max():.3f}]")


if __name__ == "__main__":
    main()
