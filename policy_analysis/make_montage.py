"""Assemble per-style / per-layout render thumbnails into a labeled contact sheet,
sorted worst->best by run-adjusted SR. Each cell: the kitchen render + a color
strip (red=below-avg/harder, blue=above-avg/easier) with 'id | SR n'.

Usage: make_montage.py <style|layout>
"""
import sys, glob, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm

kind = sys.argv[1]  # 'style' or 'layout'
FA = "/data/xinyua11/robocasa/weakregion/factor_analysis"
col = "id"
stats = pd.read_csv(FA + f"/perid_{kind}_id_stats.csv").sort_values("sr_adj").reset_index(drop=True)
gm = 0.465
norm = TwoSlopeNorm(vmin=stats.sr_adj.min(), vcenter=gm, vmax=stats.sr_adj.max())
cmap = cm.get_cmap("RdBu")

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

TW, TH, STRIP, PAD, COLS = 264, 198, 26, 6, 7
f = font(15)
rows = int(np.ceil(len(stats) / COLS))
cellW, cellH = TW + PAD, TH + STRIP + PAD
W = COLS * cellW + PAD
H = rows * cellH + PAD + 34
sheet = Image.new("RGB", (W, H), (255, 255, 255))
draw = ImageDraw.Draw(sheet)
title = (f"Kitchen {kind.upper()} catalog — sorted hardest (left/top) to easiest, "
         f"labeled id | run-adjusted SR | n.  Fixed {'layout 11' if kind=='style' else 'style 11'}, object=cup.")
draw.text((PAD, 8), title, fill=(0, 0, 0), font=font(17))

for i, r in stats.iterrows():
    idv = int(r[col]); sr = r.sr_adj; n = int(r.n)
    path = f"{FA}/renders/{kind}s/{kind}_{idv:02d}.png"
    cx = PAD + (i % COLS) * cellW
    cy = 34 + PAD + (i // COLS) * cellH
    rgb = tuple(int(255 * c) for c in cmap(norm(sr))[:3])
    draw.rectangle([cx, cy, cx + TW, cy + STRIP], fill=rgb)
    txt = f"{kind} {idv}  |  SR {sr:.2f}  |  n{n}"
    tcol = (255, 255, 255) if (0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]) < 140 else (0, 0, 0)
    draw.text((cx + 6, cy + 4), txt, fill=tcol, font=f)
    if os.path.exists(path):
        im = Image.open(path).convert("RGB").resize((TW, TH))
        sheet.paste(im, (cx, cy + STRIP))
    else:
        draw.rectangle([cx, cy + STRIP, cx + TW, cy + STRIP + TH], fill=(60, 60, 60))
        draw.text((cx + 8, cy + STRIP + 8), "MISSING", fill=(255, 80, 80), font=f)

out = f"{FA}/plots/montage_{kind}s.png"
sheet.save(out)
print("wrote", out, "| n cells", len(stats))
