import numpy as np
import pandas as pd
import cv2
import matplotlib as plt
from utils.quaternion import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm
xsense = pd.read_csv("input/Participant_541_Setup_A_Seq_5_Trial_3.xsens.csv").iloc[1:]
xsense_orig = xsense.copy()
xsense = xsense[::10].reset_index(drop=True)

xsense_orientation = xsense.filter(regex='orientation', axis=1).copy()

xsense_position = xsense.filter(regex='position', axis=1).copy()

xsense_position = xsense_position.astype('float')
anatomy_color = {"Head": "blue", 
                 "Neck": "Cyan",
                 "UpperRight": "green",
                 "UpperLeft": "green",
                 "UpperMid": "brown",
                 "Pelvis": "black", 
                 "LowerRight": "orange",
                 "LowerLeft": "orange"
                }
anatomy = {
    "Head": ["Head"], "Neck": ["Neck"], 
    "UpperRight": ["RightShoulder", "RightUpperArm", "RightForearm", "RightHand"],
    "UpperLeft": ["LeftShoulder", "LeftUpperArm", "LeftForearm", "LeftHand"],
    "UpperMid": ["L5", "L3", "T12", "T8"], 
    "Pelvis": ["Pelvis"], 
    "LowerRight": ["RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToe"], 
    "LowerLeft": ["LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToe"]
}

vals = np.reshape(xsense_position.values, (69, -1))
RADIUS = 0.9 # space around the subject
xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]

for i in tqdm(range(1, xsense_position.shape[0])):
    fig = plt.figure()
    ax = Axes3D(fig)
    for part in anatomy:
        qr = np.array([])
        for p in anatomy[part]:
            v = np.array([(xsense_position.iloc[i]["position_{}_x".format(p)]), 
                      (xsense_position.iloc[i]["position_{}_y".format(p)]),
                      (xsense_position.iloc[i]["position_{}_z".format(p)])], dtype=np.float).reshape(-1, 3)
            qr = np.append(qr, torch.from_numpy(v).reshape(-1, 3)).reshape(-1, 3)
            
        x, y, z = qr[:, 0], qr[:, 1], qr[:, 2]
        ax.plot(x, y, z, c=anatomy_color[part])
        ax.scatter3D(x, y, z, c=anatomy_color[part])

#     ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20)
    plt.legend(loc="best")
    plt.show()
    plt.savefig("output/frame{}.png".format(i))
    plt.close()