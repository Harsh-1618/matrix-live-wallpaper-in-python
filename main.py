import os
import cv2
import time
import numpy as np
import random 

window_height = 300
window_width = 400

window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

line_pos = 120
color_list = []

chars = [cv2.imread(os.path.join("./chars", char)) for char in os.listdir("chars")]
chars = [cv2.resize(char, (20,20), interpolation=cv2.INTER_CUBIC) for char in chars]
for idx in range(len(chars)):
    chars[idx][:,:,[0,2]] = 0

char_idx = 0
while True:
    for cidx in color_list:
        real_cidx = len(color_list) - cidx
        patch = window[20*(real_cidx):20*(real_cidx+1), line_pos:line_pos+20][:,:,1]
        patch_color = int(255-cidx*40)
        if patch_color < 0:
            patch_color = 0
        window[20*(real_cidx):20*(real_cidx+1), line_pos:line_pos+20][:,:,1] = np.where(patch != 0, patch_color, 0)

    height_left = window_height - 20*(char_idx)

    idx = np.random.randint(0, 10, (1,))[0]
    char = np.copy(chars[idx])

    if height_left > 20:
        window[20*(char_idx):20*(char_idx+1), line_pos:line_pos+char.shape[1]] = char

    char_idx += 1
    color_list.append(char_idx)
    cv2.imshow('matrix', window)
    time.sleep(0.05)

    if cv2.waitKey(1) == ord('q'):
        break
