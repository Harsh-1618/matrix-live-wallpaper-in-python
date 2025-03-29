import os
import cv2
import time
import random 
import numpy as np

class Streak:
    char_folder = "./chars"

    def __init__(self, window_height, streak_pos, char_height, char_width):
        self.char_height = char_height
        self.streak_pos = streak_pos
        self.chars = [cv2.imread(os.path.join(Streak.char_folder, char)) for char in os.listdir(Streak.char_folder)]
        self.chars = [cv2.resize(char, (char_width, char_height), interpolation=cv2.INTER_CUBIC) for char in self.chars]
        for idx in range(len(self.chars)):
            self.chars[idx][self.chars[idx] < 128] = 0
            self.chars[idx][self.chars[idx] > 128] = 255
            self.chars[idx][:,:,[0,2]] = 0
        
        self.num_chars = window_height // char_height
        self.streak = np.zeros((self.num_chars * char_height, char_width, 3), dtype=np.uint8)

        self.alive_chars = 0
        self.dec_color = 10
        self.start_remove_count_down = False
        self.initial_color_val = 255

    def update(self):
        self.streak = np.astype(self.streak, np.int16) # to let negative numbers come in
        mask = self.streak[0:self.alive_chars*self.char_height, :, 1] > 0
        self.streak[0:self.alive_chars*self.char_height, :, 1][mask] -= self.dec_color
        neg_mask = self.streak[0:self.alive_chars*self.char_height, :, 1] < 0
        self.streak[0:self.alive_chars*self.char_height, :, 1][neg_mask] = 0
        self.streak = np.astype(self.streak, np.uint8) # changing back to image type 

        if self.alive_chars < self.num_chars:
            char = np.copy(random.choice(self.chars))
            self.streak[self.alive_chars*self.char_height:(self.alive_chars+1)*self.char_height, :, :] = char

            self.alive_chars += 1
        else:
            self.start_remove_count_down = True

        if self.start_remove_count_down:
            self.initial_color_val -= self.dec_color
            if self.initial_color_val < 0:
                return self.streak_pos, None

        return self.streak_pos, self.streak


window_height = 230
window_width = 1200

window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

streak_list = []
while True:
    print(len(streak_list))
    initiate_streaks = random.randint(0, 2)
    for i in range(initiate_streaks):
        pos = random.randint(0, window_width-20)
        streak_list.append(Streak(window_height, pos, 20, 20))

    for s in streak_list:
        pos, streak = s.update()
        if streak is None:
            streak_list.remove(s)
            continue

        window[0:streak.shape[0], pos:pos+streak.shape[1], :] = streak

    cv2.imshow('matrix', window)
    time.sleep(0.02)
    # time.sleep(1)

    if cv2.waitKey(1) == ord('q'):
        break


######################################

# window_height = 300
# window_width = 400

# window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

# line_pos = 120
# color_list = []

# chars = [cv2.imread(os.path.join("./chars", char)) for char in os.listdir("chars")]
# chars = [cv2.resize(char, (20,20), interpolation=cv2.INTER_CUBIC) for char in chars]
# for idx in range(len(chars)):
#     chars[idx][:,:,[0,2]] = 0

# char_idx = 0
# while True:
#     for cidx in color_list:
#         real_cidx = len(color_list) - cidx
#         patch = window[20*(real_cidx):20*(real_cidx+1), line_pos:line_pos+20][:,:,1]
#         patch_color = int(255-cidx*40)
#         if patch_color < 0:
#             patch_color = 0
#         window[20*(real_cidx):20*(real_cidx+1), line_pos:line_pos+20][:,:,1] = np.where(patch != 0, patch_color, 0)

#     height_left = window_height - 20*(char_idx)

#     idx = np.random.randint(0, 10, (1,))[0]
#     char = np.copy(chars[idx])

#     if height_left > 20:
#         window[20*(char_idx):20*(char_idx+1), line_pos:line_pos+char.shape[1]] = char

#     char_idx += 1
#     color_list.append(char_idx)
#     cv2.imshow('matrix', window)
#     time.sleep(0.05)

#     if cv2.waitKey(1) == ord('q'):
#         break


