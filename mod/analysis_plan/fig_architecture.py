"""Figure 1 for the method-introduction manuscript: pipeline and hybrid GAT architecture schematic."""
import os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from common import OUT
FIG = os.path.join(OUT, "figs_v5"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})
BLUE, GREY, TEAL, RED = "#1f4e79", "#8a93a0", "#2a9d8f", "#c0504d"

def box(ax, x, y, w, h, text, fc="white", ec=BLUE, fs=8.5, lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)
def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.1, color=color))

fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.2); ax.axis("off")
# --- panel A: feature pipeline
ax.text(0.15, 6.0, "A", fontsize=15, fontweight="bold")
box(ax, 0.2, 4.4, 1.7, 1.2, "19-channel EEG\n50-s window\n200 Hz", fc="#f3f5f7", ec=GREY)
box(ax, 2.3, 4.4, 1.9, 1.2, "Pre-processing\n0.5–50 Hz band-pass\nclip + z-score", fc="#f3f5f7", ec=GREY)
box(ax, 4.6, 5.0, 2.2, 0.7, "Welch PSD per channel\n→ node features  19 × 101", fc="#e3ecf5")
box(ax, 4.6, 4.05, 2.2, 0.8, "Connectivity per pair\ncorr · coh · AEC · PLV\n→ edge attributes  171 × 4", fc="#e3ecf5")
box(ax, 7.2, 4.4, 1.9, 1.2, "Brain graph\n19 nodes, 171 edges\n(fully connected)", fc="white", ec=BLUE, bold=True)
box(ax, 9.4, 4.4, 1.45, 1.2, "Flattened\nconnectivity\nvector  1,444", fc="#fbf3dc", ec="#b8860b")
for (a, b) in [((1.9, 5.0), (2.3, 5.0)), ((4.2, 5.0), (4.6, 5.35)), ((4.2, 5.0), (4.6, 4.45)), ((6.8, 5.35), (7.2, 5.1)), ((6.8, 4.45), (7.2, 4.9)), ((6.8, 4.45), (9.4, 4.7))]: arrow(ax, *a, *b)
# head sketch
cx, cy, r = 7.55, 3.55, 0.0
# --- panel B: architecture
ax.text(0.15, 3.45, "B", fontsize=15, fontweight="bold")
box(ax, 0.2, 1.3, 1.5, 1.6, "Graph\n19 × 101 nodes\n342 directed edges\n× 4 attributes", fc="white", ec=BLUE)
xs = 2.0
for i, lab in enumerate(["GATConv 1\n13 heads × 49\nedge_dim 4", "GATConv 2\n13 heads × 49", "GATConv 3\n13 heads × 49", "GATConv 4\n1 head, 49\n(mean)"]):
    box(ax, xs + i * 1.45, 1.55, 1.3, 1.1, lab, fc="#e3ecf5", fs=7.5)
    if i < 3: arrow(ax, xs + i * 1.45 + 1.3, 2.1, xs + (i + 1) * 1.45, 2.1)
arrow(ax, 1.7, 2.1, 2.0, 2.1)
ax.text(xs + 2.9, 1.3, "ELU · dropout 0.25 between layers", ha="center", fontsize=7.5, color=GREY)
box(ax, 7.95, 1.55, 1.3, 1.1, "Global\nmean pool\n→ 49", fc="#e3ecf5", fs=7.5); arrow(ax, 7.65, 2.1, 7.95, 2.1)
box(ax, 2.0, 0.25, 2.75, 0.75, "Flat branch: Linear 1,444 → 64 · ReLU · dropout", fc="#fbf3dc", ec="#b8860b", fs=7.5)
box(ax, 0.2, 0.25, 1.5, 0.75, "Flattened\nconnectivity 1,444", fc="white", ec="#b8860b", fs=7.5); arrow(ax, 1.7, 0.62, 2.0, 0.62)
box(ax, 9.55, 1.1, 1.3, 1.6, "Concat 113\nLinear → 49 · ELU\nLinear → 4\nlog-softmax", fc="white", ec=BLUE, bold=True, fs=7.5)
arrow(ax, 9.25, 2.1, 9.55, 2.3); arrow(ax, 4.75, 0.62, 9.55, 1.5, color="#b8860b")
ax.text(10.2, 0.75, "Seizure · LPD\nGPD · Others", ha="center", fontsize=8, color=BLUE)
# attention inset text
ax.text(5.5, 3.2, r"$\alpha_{ij} = \mathrm{softmax}_j(\mathrm{LeakyReLU}(a^T [W h_i \| W h_j \| W_e e_{ij}])), \quad h_i' = \mathrm{ELU}(\sum_{j \in N(i)} \alpha_{ij} W h_j)$", ha="center", fontsize=9)
ax.text(5.5, 2.85, "edge attributes enter the attention logits (edge_dim = 4), so coupling strength modulates message passing", ha="center", fontsize=7.5, color=GREY)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "F0_architecture.png"), dpi=200, bbox_inches="tight"); print("saved")
