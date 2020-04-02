import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plot_good_trial_participant(participants, data):
    x = np.arange(len(participants))
    fig, ax = plt.subplots()
    rects = ax.bar(x, data, 0.6, label='good trials')
    ax.set_xticks(x)
    ax.set_xticklabels(participants)
    ax.set_xlabel('Participants')
    ax.set_ylabel('Good trials (%)')
    ax.set_title('Good trials (%) left for each participant')
    autolabel(rects, ax)
    plt.show()


def plot_good_trial_word(participant, data):
    new_names = {}
    for i in range(len(participant)):
        new_names[i] = int(participant[i])
    # pandas would automatically convert int to float if nan exists
    df = pd.DataFrame(data).T
    df = df.replace(np.nan, -1)
    df = df.astype(int)
    df = df.replace(-1, "-")
    df = df.rename(columns=new_names)
    display(df)


def plot_error_rate_window():
    pass

# doesn't look good


def failed_plot_good_trial_word(participants, data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    yticks = participants
    for p in range(len(participants)):
        xs = np.arange(len(data[p]))
        ys = data[p].tolist()
        ax.bar(xs, participants[p], zs=ys, zdir='y', alpha=0.8)

    ax.set_xlabel('Words')
    ax.set_ylabel('Participants')
    ax.set_zlabel('Good Trials')
    ax.set_xticks(np.arange(len(data[0])))
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels(yticks)

    plt.show()
