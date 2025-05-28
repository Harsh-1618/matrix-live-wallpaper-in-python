import os
import cv2
import time
import random 
import numpy as np
from copy import deepcopy


def character_maker(sizes, char_folders, window_color, character_color):
    character_dict = {}
    for size in sizes:
        char_height, char_width = size
        bw_characters_list = []
        characters_list = []
        for folder in char_folders:
            folder_path = os.path.join("./characters_seperated", folder)
            chars = [cv2.imread(os.path.join(folder_path, char_file), cv2.IMREAD_COLOR) for char_file in os.listdir(folder_path)]
            assert len(chars[0].shape) == 3, "input character does not have 3 channels!"

            if chars[0].shape != (char_height, char_width, 3): # from my experiments it turns out cv2 doesn't check if the resize dim is same as original, so it should skip resize!
                chars = [cv2.resize(char, (char_width, char_height), interpolation=cv2.INTER_NEAREST) for char in chars] # not using INTER_CUBIC as it'll introduce values other than 0 and 255
            
            bw_chars = deepcopy(chars)
            for idx in range(len(chars)):
                character_mask = chars[idx][:,:,0] == 255
                chars[idx][:, :, :] = np.array(window_color)
                chars[idx][character_mask, :] = np.array(character_color)

            bw_characters_list.extend(bw_chars)
            characters_list.extend(chars)
        character_dict[size] = (bw_characters_list, characters_list)
    return character_dict


class Streak:
    def __init__(self,
                window_height,
                streak_x_pos,
                char_height,
                char_width,
                chars,
                window_color,
                character_color):
        self.char_height = char_height
        self.streak_x_pos = streak_x_pos
        self.bw_chars = chars[0]
        self.chars = chars[1]
        self.len_chars = len(self.chars)
        self.streak_y_pos = None
        self.out = None

        self.streak_max_char_len = window_height // char_height
        self.streak_char_len = random.randint(3, self.streak_max_char_len)
        self.bw_streak = np.zeros((self.streak_char_len*self.char_height, char_width, 3), dtype=np.uint8)
        self.streak = np.zeros((self.streak_char_len*self.char_height, char_width, 3), dtype=np.uint8)

        self.updates = 0
        self.dec_color = (((np.array(character_color) - np.array(window_color)) / self.streak_char_len).astype(np.int16)).astype(np.uint8) # you can also try self.streak_char_len-1 but that'll produce too dark color for the last element

        self.remove_x_pos = False

    def update(self):
        if self.updates < self.streak_char_len:
            mask = self.bw_streak[0:self.updates*self.char_height, :, 0] == 255
            self.streak[0:self.updates*self.char_height, :, 0][mask] -= self.dec_color[0]
            self.streak[0:self.updates*self.char_height, :, 1][mask] -= self.dec_color[1]
            self.streak[0:self.updates*self.char_height, :, 2][mask] -= self.dec_color[2]
            # self.streak[mask, :] -= self.dec_color

            rnd_char_idx = np.random.randint(0, self.len_chars)
            bw_char = np.copy(self.bw_chars[rnd_char_idx])
            char = np.copy(self.chars[rnd_char_idx])
            
            self.bw_streak[self.updates*self.char_height:(self.updates+1)*self.char_height, :, :] = bw_char
            self.streak[self.updates*self.char_height:(self.updates+1)*self.char_height, :, :] = char

            self.streak_y_pos = 0
            self.out = self.streak[0:(self.updates+1)*self.char_height, :, :]

        elif self.updates < self.streak_max_char_len:
            self.bw_streak[0:(self.streak_char_len-1)*self.char_height, :, :] = self.bw_streak[self.char_height:, :, :]
            self.streak[0:(self.streak_char_len-1)*self.char_height, :, :] = self.streak[self.char_height:, :, :]

            mask = self.bw_streak[0:(self.streak_char_len-1)*self.char_height, :, 0] == 255
            self.streak[0:(self.streak_char_len-1)*self.char_height, :, 0][mask] -= self.dec_color[0]
            self.streak[0:(self.streak_char_len-1)*self.char_height, :, 1][mask] -= self.dec_color[1]
            self.streak[0:(self.streak_char_len-1)*self.char_height, :, 2][mask] -= self.dec_color[2]
            # self.streak[mask, :] -= self.dec_color

            rnd_char_idx = np.random.randint(0, self.len_chars)
            bw_char = np.copy(self.bw_chars[rnd_char_idx])
            char = np.copy(self.chars[rnd_char_idx])

            self.bw_streak[(self.streak_char_len-1)*self.char_height:, :, :] = bw_char
            self.streak[(self.streak_char_len-1)*self.char_height:, :, :] = char

            self.streak_y_pos += self.char_height
            self.out = self.streak

            self.remove_x_pos = True # we can let other streak share x_pos after this point
            # since now we'll be traversing down.

        else:
            self.bw_streak = self.bw_streak[self.char_height:, :, :]
            self.streak = self.streak[self.char_height:, :, :]

            mask = self.bw_streak[:, :, 0] == 255
            self.streak[:, :, 0][mask] -= self.dec_color[0]
            self.streak[:, :, 1][mask] -= self.dec_color[1]
            self.streak[:, :, 2][mask] -= self.dec_color[2]
            # self.streak[mask, :] -= self.dec_color

            self.streak_y_pos += self.char_height
            self.out = self.streak

            # we need to set the flag here as well since in the case when streak_char_len is 
            # equal to streak_max_char_len, the above elif block will be skipped and hence
            # remove_x_pos will never get to be True and it'll not remove x_pos from the list,
            # accumulating it and basically filling it completely. this is much more prominant in
            # small window size. ex 100,100,20,20 will just go blank after some time due to 
            # x_pos list not clearing. it's a really niche bug which took me like 30 mins to figure out
            # since in higher window sizes it's not a problem, atleast for short time runs of like 5 mins
            self.remove_x_pos = True 
            

        self.updates += 1

        return self.streak_x_pos, self.streak_y_pos, self.out

    def __repr__(self):
        return f"Streak at x position of {self.streak_x_pos}"


def run_matrix_flat(window_height=720,
                window_width=1280,
                window_color=(0,0,0),
                character_color=(0,255,0),
                char_height=20,
                char_width=20,
                max_new_streaks=2,
                spf=5e-2,
                consecutive_streak=False,
                character_folders=("english_lower",)):
    """
    spf: seconds per frame, not exact at all since the computation takes time as well
    consecutive_streak: if False, one column will only contain one streak till it dies,
                        if True, on column can contain more than one streak
    """
    character_dict = character_maker(((char_height,char_width),), character_folders, window_color, character_color)
    window = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    random_positions = (window_width // char_width) - 1

    streak_list = []
    x_pos_list = []
    if consecutive_streak:
        x_pos_del_dict = {}
    while True:
        if (random_positions+1 - len(x_pos_list)) > (max_new_streaks-1):
            initiate_streaks = random.randint(0, max_new_streaks) # [0,max_new_streaks]
        else:
            initiate_streaks = random_positions+1 - len(x_pos_list)

        for i in range(initiate_streaks):
            rand_x_pos = random.randint(0, random_positions) * char_width
            while rand_x_pos in x_pos_list:
                rand_x_pos = random.randint(0, random_positions) * char_width
            x_pos_list.append(rand_x_pos)

            s = Streak(window_height, rand_x_pos, char_height, char_width, character_dict[(char_height,char_width)], window_color, character_color)
            streak_list.append(s)
            if consecutive_streak:
                x_pos_del_dict[s] = True

        for s in streak_list[::-1]: # reverse traverse since we are removing elements while traversing, else it'll create annoying bug
            x_pos, y_pos, streak = s.update()
            if consecutive_streak:
                if x_pos_del_dict[s] and s.remove_x_pos:
                    x_pos_list.remove(x_pos)
                    x_pos_del_dict[s] = False
                if streak.shape[0] == 0:
                    streak_list.remove(s)
                    del x_pos_del_dict[s]   
                    continue
            else:
                if streak.shape[0] == 0:
                    x_pos_list.remove(x_pos)
                    streak_list.remove(s)
                    continue

            window[y_pos:y_pos+streak.shape[0], x_pos:x_pos+streak.shape[1], :] = streak

        cv2.imshow('Matrix', window)
        time.sleep(spf)
        window[:, :, :] = np.array(window_color) # resetting the window

        if cv2.waitKey(1) == ord('q'):
            break

def run_matrix_overlap(window_height=720,
                window_width=1280,
                window_color=(0,0,0),
                character_color=(0,255,0),
                max_new_streaks=2,
                spf=5e-2,
                sizes=((20,20),(30,30),(40,40)),
                character_folders=("english_lower",)):
    """
    spf: seconds per frame, not exact at all since the computation takes time as well
    sizes: tuple of tuple of hight,width of characters that'll be displayed in matrix
    """
    character_dict = character_maker(sizes, character_folders, window_color, character_color)
    window = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    max_x_position = window_width - max(s[1] for s in sizes)

    streak_list = []
    while True:
        initiate_streaks = random.randint(0, max_new_streaks)
        for i in range(initiate_streaks):
            rand_x_pos = random.randint(0, max_x_position)
            rand_hw = random.choice(sizes)

            s = Streak(window_height, rand_x_pos, rand_hw[0], rand_hw[1], character_dict[(rand_hw[0],rand_hw[1])], window_color, character_color)
            streak_list.append(s)

        for s in streak_list[::-1]: # reverse traverse since we are removing elements while traversing, else it'll create annoying bug
            x_pos, y_pos, streak = s.update()
            if streak.shape[0] == 0:
                streak_list.remove(s)
                continue

            # adding the streak instead of overriding so as to overlap
            window[y_pos:y_pos+streak.shape[0], x_pos:x_pos+streak.shape[1], :] += streak

        cv2.imshow('Matrix', window)
        time.sleep(spf)
        window[:, :, :] = np.array(window_color) # resetting the window

        if cv2.waitKey(1) == ord('q'):
            break

def main():
    window_color = (255, 255, 255) # BGR, not RGB!
    character_color = (0, 255, 0) # BGR, not RGB!
    # character_folders = ("english_digit", "english_lower", "english_capital")
    run_matrix_flat(window_color=window_color, character_color=character_color, max_new_streaks=5, spf=5e-2, consecutive_streak=True, character_folders=character_folders)
    run_matrix_overlap(window_color=window_color, character_color=character_color, max_new_streaks=2, spf=5e-2, character_folders=character_folders)

if __name__ == "__main__":
    main()