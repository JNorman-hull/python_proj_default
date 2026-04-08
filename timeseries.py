import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FILE_ID = "B79-0704112227"
DATA_PATH = "python_data/labeled_data_with_types.csv"

# Sensor channels (excluding mag), with assigned colours
CHANNELS = {
    "higacc_x_g":   "red",
    "higacc_y_g":   "red",
    "higacc_z_g":   "red",
    "inacc_x_ms":   "blue",
    "inacc_y_ms":   "blue",
    "inacc_z_ms":   "blue",
    "rot_x_degs":   "green",
    "rot_y_degs":   "green",
    "rot_z_degs":   "green",
    "pressure_kpa": "black",
}

df = pd.read_csv(DATA_PATH)
df = df[df["file"] == FILE_ID].sort_values("time_s")

fig, axes = plt.subplots(
    nrows=10, ncols=1,
    figsize=(4/2.54, 6/2.54),   # cm → inches (w, h)
    sharex=True,
)

for ax, (col, colour) in zip(axes, CHANNELS.items()):
    ax.plot(df["time_s"], df[col], color=colour, linewidth=0.6)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(
        left=True, right=False, bottom=True, top=False,
        labelleft=False, labelbottom=False,
        length=2, width=0.5,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

fig.subplots_adjust(hspace=0.15, left=0.04, right=0.98, top=0.98, bottom=0.04)

fig.savefig(f"{FILE_ID}_timeseries.svg", dpi=300, bbox_inches="tight")
fig.savefig(f"{FILE_ID}_timeseries.png", dpi=300, bbox_inches="tight")
plt.show()
