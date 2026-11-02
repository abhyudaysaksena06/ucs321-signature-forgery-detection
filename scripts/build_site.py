"""Turn the notebook's saved outputs into site/data.js and site/img. Usage: build_site.py rundir sitedir"""
import io, json, re, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

run, site = Path(sys.argv[1]), Path(sys.argv[2])
R = json.loads((run / "results.json").read_text())
img = site / "img"; img.mkdir(parents=True, exist_ok=True)
stats = pd.read_csv(run / "image_stats.csv")
S = np.load(run / "test_scores.npz")
I = np.load(run / "test_images.npz", allow_pickle=True)
clean, index, enrolled = I["clean"], I["index"], I["enrolled"].item()
pos = {int(i): k for k, i in enumerate(index)}
p, y, q = S["Reference_CNN"], S["y_test"].astype(int), S["test_q"]

def save(array, name):
    Image.fromarray(255 - array).save(img / name, optimize=True)
    return f"img/{name}"

# Ten rounds: five genuine, five forged, spread over the test signers, in a fixed shuffled order.
r = np.random.default_rng(7)
writers = stats["writer"].to_numpy()
rounds = []
for truth in (0, 1):
    pool = np.where(y == truth)[0]
    by_writer = {w: pool[writers[q[pool]] == w] for w in S["test_w"]}
    picks = [r.choice(v) for v in list(by_writer.values())[:5]]
    rounds += [(int(i), truth) for i in picks]
rounds = [rounds[i] for i in r.permutation(len(rounds))]
game = []
for n, (i, truth) in enumerate(rounds):
    w = int(writers[q[i]])
    ref = int(enrolled[w][0])
    game.append({"ref": save(clean[pos[ref]], f"g{n}r.png"), "q": save(clean[pos[int(q[i])]], f"g{n}q.png"),
                 "truth": truth, "p": round(float(p[i]), 3)})

# One signature through the cleaning steps.
w = int(S["test_w"][0])
with zipfile.ZipFile(run / "cedar_signatures.zip") as z:
    name = next(n for n in z.namelist() if n.endswith(f"/Dataset/genuine_sign/original_{w}_1.png"))
    gray = Image.open(io.BytesIO(z.read(name))).convert("L")
gray.thumbnail((520, 300))
a = np.asarray(gray)
h = np.bincount(a.ravel(), minlength=256).astype(float); pr = h / h.sum(); om = np.cumsum(pr); mu = np.cumsum(pr * np.arange(256))
t = int(((mu[-1] * om - mu) ** 2 / (om * (1 - om) + 1e-12)).argmax())
mask = a <= t
ys, xs = np.where(mask)
gray.save(img / "step1.png", optimize=True)
Image.fromarray((~mask * 255).astype(np.uint8)).save(img / "step2.png", optimize=True)
Image.fromarray((~mask * 255).astype(np.uint8)).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).save(img / "step3.png", optimize=True)

bg = np.floor(stats["background"]).astype(int)
bins = list(range(int(bg.min()), int(bg.max()) + 1))
table = [{"name": k, **v} for k, v in R["table"].items()]
ref = R["table"]["Reference CNN"]
data = {
    "n_images": R["n_images"], "n_writers": R["n_writers"], "test_queries": R["test_queries"],
    "control_acc": R["table"]["Raw-image control (invalid)"]["Accuracy"],
    "leak_acc": R["leak_acc"], "single_acc": R["single_acc"],
    "stability_mean": R["stability_mean"], "stability_std": R["stability_std"],
    "ref_beats_80": ref["Accuracy"] >= .8,
    "table": table,
    "background": {"bins": bins,
                   "genuine": [int(((bg == b) & (stats["label"] == 0)).sum()) for b in bins],
                   "forged": [int(((bg == b) & (stats["label"] == 1)).sum()) for b in bins]},
    "history": {"reference": pd.read_csv(run / "history_reference.csv")["val_accuracy"].round(4).tolist(),
                "single": pd.read_csv(run / "history_single.csv")["val_accuracy"].round(4).tolist()},
    "scores": {"p": [round(float(v), 3) for v in p], "y": y.tolist()},
    "clean_demo": [["img/step1.png", "1. Scan", "Grey paper, and the shortcut."],
                   ["img/step2.png", "2. Otsu mask", "Ink only. Paper is pure white."],
                   ["img/step3.png", "3. Crop", "Just the signature, ready to resize."]],
    "game": game,
}
(site / "data.js").write_text("window.DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
print("rounds", len(game), "forged", sum(g["truth"] for g in game), "| data.js KB", (site / "data.js").stat().st_size // 1024)
