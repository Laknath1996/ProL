import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation




def make_video(arrs, fname, start, frames):

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(context='poster',
            style='ticks',
            font_scale=0.75,
            rc={'axes.grid':True,
                'grid.color':'.9',
                'figure.autolayout': True,
                'grid.linewidth':0.75})

    xar = arrs[0]['x'].reshape(-1)

    preds = []
    errs = []
    for arr in arrs:
        preds.append(np.array(arr['pred'])[:, :, 1])
        errs.append(np.array(arr['errs']))

    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 6)
    axs = []
    axs.append(fig.add_subplot(gs[0, 0:2]))
    axs.append(fig.add_subplot(gs[0, 2:4]))
    axs.append(fig.add_subplot(gs[0, 4:6]))
    axs.append(fig.add_subplot(gs[1, 0:4]))
    axs.append(fig.add_subplot(gs[1, 4:6]))

    # (1) Function space characterization
    scats = []
    for i in range(3):
        scats.append(axs[i].scatter([], [], marker='x', s=10, c='C%d' % i))
        axs[i].set_xlabel('X')

    axs[0].set_ylabel('P(Y =1 | X)')
    axs[0].set_title('Online SGD (Linear)')

    # (4) Errors
    xerr = []
    yerr = [[], [], []]
    for i in range(start, start+frames, 10):
        axs[3].axvline(x=i, color='k', linestyle='--', alpha=0.2)
    scats.append(
        [axs[3].plot([], [], c='C0'),
         axs[3].plot([], [], c='C1'),
         axs[3].plot([], [], c='C2')])
    axs[3].set_ylim(-0.1, 1.1)
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("0-1 Error")
    axs[3].set_title('Instantaneous risk')

    # (5) Task distribution
    positions = [((-2, 0), (1.0, -0.5)),
                 ((-2.0, -0.5), (1.0, 0.0))]

    rect1 = patches.Rectangle((-2, 0), 1.0, 0.5, fc='C0')
    rect2 = patches.Rectangle((1, -0.5), 1.0, 0.5, fc='C0')
    axs[4].add_patch(rect1)
    axs[4].add_patch(rect2)
    axs[4].set_xlabel('X')
    axs[4].set_xlim(-2.1, 2.1)
    axs[4].set_ylim(-1, 1)
    axs[4].set_title('Task distribution')

    axs[1].set_title('Online SGD (MLP)')
    axs[2].set_title('Prospective')
    plt.tight_layout()

    # Horizontal line
    for i in range(3):
        ax = axs[i]
        ax.axhline(y=0.5, color='k')
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-0.1, 1.1)

    def update(frame):
        # First 3 plots
        for i in range(3):
            sc = scats[i]
            sc.set_offsets(np.c_[xar, preds[i][start+frame]])

        # 4th plot - Error plot
        xerr.append(start+frame)
        for i in range(3):
            yerr[i].append(errs[i][start+frame])
            scats[3][i][0].set_data(xerr, yerr[i])

        # 5th plot - task Distribution
        tidx = frame % 20 > 10
        rect1.set_xy(positions[tidx][0])
        rect2.set_xy(positions[tidx][1])


    # ani = FuncAnimation(fig, update, frames=500, repeat=False)
    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    ani.save(fname, writer='ffmpeg', fps=8)


arrs = []
fnames = ["info/linear_ogd.pkl",
          "info/mlp_ogd.pkl",
          "info/prol_ogd.pkl"]

for fn in fnames:
    with open(fn, "rb") as fp:
        arrs.append(pickle.load(fp))

frames = 100
for start in [100, 500, 5000]:
    make_video(arrs, "video/ogd_%d.mp4" % start, start, frames)

