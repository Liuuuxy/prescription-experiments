"""Category-free failed-state retrieval (BehaviorRetrieval-style).

Select training demos whose INITIAL SCENE looks like the scenes where the policy actually fails —
no object categories, no estimated failure labels, target = real deployment failures.

  embed_demos : embed the first agentview frame of every candidate demo (DINOv2)  [run anytime]
  retrieve    : embed the logged failed/success scenes, score demos, write arms_failretr.json
                [run after weakregion/failstates logging completes]
Run in the robocasa conda env (torch + torchvision + cv2 + DINOv2 hub).
"""
import argparse, glob, json, os
import numpy as np, cv2, torch

ROOT = "/data/xinyua11/robocasa"
MG = "/data/xinyua11/robocasa_pkg/datasets/v1.0/pretrain/atomic/PickPlaceCounterToSink/20250819/mg/demo/2025-08-20-22-32-27/lerobot"
DEV = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder():
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(DEV)


def prep(imgs):
    x = torch.stack([torch.from_numpy(cv2.resize(im, (224, 224))) for im in imgs]).float() / 255.0
    return (x.permute(0, 3, 1, 2) - MEAN) / STD


@torch.no_grad()
def embed(model, imgs, bs=64):
    out = []
    for s in range(0, len(imgs), bs):
        f = model(prep(imgs[s:s + bs]).to(DEV))
        out.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def demo_frame(e):
    p = f"{MG}/videos/chunk-{e//1000:03d}/observation.images.robot0_agentview_left/episode_{e:06d}.mp4"
    cap = cv2.VideoCapture(p); ok, fr = cap.read(); cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


def cmd_embed_demos(a):
    eps = json.load(open(f"{ROOT}/weakregion/rcless/cand_meta.json"))["episodes"]
    model = load_encoder()
    embs = np.zeros((len(eps), 384), np.float32); bad = 0
    buf, idx = [], []
    for i, e in enumerate(eps):
        fr = demo_frame(e)
        if fr is None:
            bad += 1; fr = np.zeros((256, 256, 3), np.uint8)
        buf.append(fr); idx.append(i)
        if len(buf) == 64 or i == len(eps) - 1:
            embs[idx] = embed(model, buf); buf, idx = [], []
            if (i // 64) % 10 == 0:
                print(f"  embedded {i+1}/{len(eps)}", flush=True)
    np.save(f"{ROOT}/weakregion/failretr_demo_emb.npy", embs)
    print(f"saved demo embeddings {embs.shape}; unreadable frames: {bad}")


def _topk_sim(D, E, k=10):
    S = D @ E.T                                   # [n_demo, n_scene]
    k = min(k, S.shape[1])
    return np.sort(S, 1)[:, -k:].mean(1)


def cmd_retrieve(a):
    recs = json.load(open(f"{ROOT}/weakregion/failstates_eval/weakregion.json"))["episodes"]
    imgs = {}
    for f in glob.glob(f"{ROOT}/weakregion/failstates/ep*.npy"):
        imgs[int(os.path.basename(f)[2:6])] = np.load(f)
    fail = [imgs[r["episode"]] for r in recs if (not r["success"]) and r["episode"] in imgs]
    succ = [imgs[r["episode"]] for r in recs if r["success"] and r["episode"] in imgs]
    print(f"fail scenes {len(fail)} | success scenes {len(succ)}")
    model = load_encoder()
    fe, se = embed(model, fail), embed(model, succ)
    de = np.load(f"{ROOT}/weakregion/failretr_demo_emb.npy")

    fail_s = _topk_sim(de, fe, a.k)
    succ_s = _topk_sim(de, se, a.k)
    contrast = fail_s - succ_s                     # like failures, unlike successes

    # ---- validation (labels NOT used in scoring) ----
    zcat = np.array(json.load(open(f"{ROOT}/weakregion/rcless/cand_meta.json"))["categories"])
    zeps = np.array(json.load(open(f"{ROOT}/weakregion/rcless/cand_meta.json"))["episodes"])
    HARD = set(json.load(open(f"{ROOT}/weakregion/targeted_rebalanced.json"))["new_targeted"])
    is_hard = np.array([c in HARD for c in zcat])
    core = set(json.load(open(f"{ROOT}/weakregion/arms.json"))["core_episodes"])
    def report(name, score):
        sel = np.argsort(-score)[:200]
        ov = len(set(int(zeps[i]) for i in sel) & core)
        print(f"  {name:22s} top200 known-hard%={is_hard[sel].mean()*100:4.1f}%  overlap-core={ov}")
        return sel
    print(f"base rate known-hard% = {is_hard.mean()*100:.1f}%  (random selection baseline)")
    report("fail-sim", fail_s)
    report("contrast(fail-succ)", contrast)
    import collections
    sel = report(f"CHOSEN contrast", contrast)
    print("  top categories:", dict(collections.Counter(zcat[i] for i in sel).most_common(10)))

    chosen = sorted(int(zeps[i]) for i in np.argsort(-contrast)[:200])
    base = json.load(open(f"{ROOT}/weakregion/arms.json"))
    json.dump({"n_core": 200, "n_random": 200, "n_base": len(base["base_episodes"]), "seed": 0,
               "selector": "failretr_dinov2_contrast", "core_episodes": chosen,
               "random_episodes": base["random_episodes"], "base_episodes": base["base_episodes"]},
              open(f"{ROOT}/weakregion/arms_failretr.json", "w"))
    print(f"wrote weakregion/arms_failretr.json ({len(chosen)} demos)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed_demos")
    r = sub.add_parser("retrieve"); r.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    {"embed_demos": cmd_embed_demos, "retrieve": cmd_retrieve}[a.cmd](a)


if __name__ == "__main__":
    main()
