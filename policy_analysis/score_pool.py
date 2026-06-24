"""Pluggable pool scorer with feature-space selection methods (beyond the P(fail) heuristic).

Embeds pi0's failure scenes + every pool demo (DINOv2 or CLIP), caches the embeddings,
then selects pool demos by one of:
  - retrieval_topk   : top-N by mean top-k cosine sim to the failure set (Method B).
  - submod_coverage  : GREEDY FACILITY-LOCATION -- maximize coverage of the failure
                       distribution; each demo's value is its MARGINAL gain, so the set is
                       diverse + relevant by construction (a principled optimization, not a
                       per-category rule). This is the "better core algorithm" candidate.

Embeddings cache to weakregion/emb_<encoder>_{fail,pool}.npy (reused across methods).
Run in the robocasa env, MUJOCO_GL=egl.

  MUJOCO_GL=egl python policy_analysis/score_pool.py --student weakregion/pi0_student_n500/weakregion.json \
    --mg_dir <mg lerobot> --encoder dinov2 --method submod_coverage --n_select 200 \
    --out weakregion/submod_core.json
"""
import argparse
import json
import os
import numpy as np
import torch
import gymnasium as gym
import robocasa  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAM = "robot0_agentview_left"


def to_uint8_hwc(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] < a.shape[-1]:
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8) if float(a.max()) <= 1.0 else a.astype(np.uint8)
    return a


def load_encoder(name, device):
    if name == "dinov2":
        from transformers import AutoModel, AutoImageProcessor
        m = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
        p = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)

        def emb(imgs):
            inp = p(images=[to_uint8_hwc(x) for x in imgs], return_tensors="pt").to(device)
            out = m(**inp).pooler_output  # (B, 768) CLS-pooled
            return out
        return emb
    elif name == "clip":
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def emb(imgs):
            inp = p(images=[to_uint8_hwc(x) for x in imgs], return_tensors="pt").to(device)
            pooled = m.vision_model(pixel_values=inp["pixel_values"]).pooler_output
            return m.visual_projection(pooled)
        return emb
    raise ValueError(name)


@torch.no_grad()
def embed_all(emb_fn, imgs, device, bs=64, tag=""):
    out = []
    for i in range(0, len(imgs), bs):
        e = emb_fn(imgs[i:i + bs])
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        if tag and (i // bs) % 10 == 0:
            print(f"  {tag}: {min(i+bs,len(imgs))}/{len(imgs)}", flush=True)
    return torch.cat(out).numpy()


def greedy_coverage(sim_pf, n_select):
    """Facility-location greedy: maximize sum_f max_{p in S} sim[f,p] (marginal gain)."""
    n_pool, n_fail = sim_pf.shape
    best = np.zeros(n_fail, dtype=np.float32)
    avail = np.ones(n_pool, dtype=bool)
    selected = []
    for _ in range(n_select):
        gain = np.where(avail, np.maximum(sim_pf - best[None, :], 0).sum(axis=1), -1.0)
        p = int(gain.argmax())
        selected.append(p)
        avail[p] = False
        best = np.maximum(best, sim_pf[p])
    return selected, float(best.mean())  # mean coverage achieved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--mg_dir", required=True)
    ap.add_argument("--task", default="PickPlaceCounterToSink")
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--encoder", choices=["dinov2", "clip"], default="dinov2")
    ap.add_argument("--method", choices=["retrieval_topk", "submod_coverage"], default="submod_coverage")
    ap.add_argument("--n_select", type=int, default=200)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("-d", "--device", default="cuda:0")
    a = ap.parse_args()
    device = torch.device(a.device)
    fe_cache = f"weakregion/emb_{a.encoder}_fail.npy"
    pe_cache = f"weakregion/emb_{a.encoder}_pool.npy"

    if os.path.exists(fe_cache) and os.path.exists(pe_cache):
        print("using cached embeddings", flush=True)
        Fe, Pe = np.load(fe_cache), np.load(pe_cache)
    else:
        emb_fn = load_encoder(a.encoder, device)
        eps = json.load(open(a.student))["episodes"]
        failed = [e["episode"] for e in eps if not e["success"]]
        print(f"rendering {len(failed)} failure scenes", flush=True)
        ff = []
        for s in failed:
            env = gym.make(f"robocasa/{a.task}", split=a.split, seed=s)
            obs, _ = env.reset(); ff.append(np.asarray(obs[f"video.{CAM}"])); env.close()
        Fe = embed_all(emb_fn, ff, device, tag="fail-embed")
        ds = LeRobotDataset("x", root=a.mg_dir)
        starts = ds.episode_data_index["from"].tolist()
        print(f"embedding {ds.num_episodes} pool demos", flush=True)
        pf = [ds[starts[e]][f"observation.images.{CAM}"] for e in range(ds.num_episodes)]
        Pe = embed_all(emb_fn, pf, device, tag="pool-embed")
        np.save(fe_cache, Fe); np.save(pe_cache, Pe)

    sim = Pe @ Fe.T  # (n_pool, n_fail)
    if a.method == "submod_coverage":
        selected, cov = greedy_coverage(sim, a.n_select)
        meta = {"coverage_achieved": cov}
    else:
        score = np.sort(sim, axis=1)[:, -a.topk:].mean(axis=1)
        selected = list(np.argsort(-score)[:a.n_select])
        meta = {"score_range": [float(score.min()), float(score.max())]}
    selected = sorted(int(i) for i in selected)
    json.dump({"method": a.method, "encoder": a.encoder, "n_select": len(selected),
               "core_episodes": selected, **meta}, open(a.out, "w"), indent=2)
    print(f"\n{a.method}/{a.encoder}: selected {len(selected)} -> {a.out}  {meta}")


if __name__ == "__main__":
    main()
