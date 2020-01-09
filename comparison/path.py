import numpy as np
import cv2
from matplotlib import pyplot as plt

from pitchmap.cache_loader import pickler
from pitchmap.players import structure
import imutils

frames = pickler.unpickle_data("data/cache/baltyk_starogard_1.mp4_path_selector_2.pik")
x = []
y = []
x2 = []
y2 = []
for f in frames.values():
    x.append(f[0].position[0])
    y.append(f[0].position[1])
    if f[1] is not None:
        x2.append(f[1].position[0])
        y2.append(f[1].position[1])

fig, ax = plt.subplots()
pitch_model = cv2.imread('data/pitch_model.jpg')
pitch_model = imutils.resize(pitch_model, width=600)

#ax.imshow(pitch_model)
ax.scatter(x, y, marker='s', label="manual")
ax.scatter(x2, y2, marker='^', label="automatic")
# ax.set_xlim([200, 300])
# ax.set_ylim([250, 400])
plt.legend()
plt.show()
print(frames)