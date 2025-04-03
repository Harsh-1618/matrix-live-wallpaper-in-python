import os
import cv2
import time
import random 
import numpy as np

class Streak:
    char_folder = "./chars"

    def __init__(self,
                window_height,
                streak_x_pos,
                char_height,
                char_width):
        self.char_height = char_height
        self.streak_x_pos = streak_x_pos
        self.streak_y_pos = None
        self.out = None

        self.chars = [cv2.imread(os.path.join(Streak.char_folder, char)) for char in os.listdir(Streak.char_folder)]
        if self.chars[0].shape != (char_height, char_width): # from my experiments it turns out cv2 doesn't check if the resize dim is same as original, so it should skip resize!
            self.chars = [cv2.resize(char, (char_width, char_height), interpolation=cv2.INTER_CUBIC) for char in self.chars]
        for idx in range(len(self.chars)):
            self.chars[idx][self.chars[idx] < 128] = 0 # to make the image perfect monochromatic
            self.chars[idx][self.chars[idx] > 128] = 255 # to make the image perfect monochromatic
            self.chars[idx][:,:,[0,2]] = 0 # only green channel

        self.streak_max_char_len = window_height // char_height
        self.streak_char_len = random.randint(3, self.streak_max_char_len)
        self.streak = np.zeros((self.streak_char_len*self.char_height, char_width, 3), dtype=np.uint8)

        self.updates = 0
        self.dec_color = 255 // self.streak_char_len # you can also try self.streak_char_len-1 but that'll produce too dark color for the last element

    def update(self):
        if self.updates < self.streak_char_len:
            mask = self.streak[0:self.updates*self.char_height, :, 1] > 0
            self.streak[0:self.updates*self.char_height, :, 1][mask] -= self.dec_color

            char = np.copy(random.choice(self.chars))
            self.streak[self.updates*self.char_height:(self.updates+1)*self.char_height, :, :] = char

            self.streak_y_pos = 0
            self.out = self.streak[0:(self.updates+1)*self.char_height, :, :]

        elif self.updates < self.streak_max_char_len:
            self.streak[0:(self.streak_char_len-1)*self.char_height, :, :] = self.streak[self.char_height:, :, :]
            mask = self.streak[0:(self.streak_char_len-1)*self.char_height, :, 1] > 0
            self.streak[0:(self.streak_char_len-1)*self.char_height, :, 1][mask] -= self.dec_color

            char = np.copy(random.choice(self.chars))
            self.streak[(self.streak_char_len-1)*self.char_height:, :, :] = char

            self.streak_y_pos += self.char_height
            self.out = self.streak

        else:
            self.streak = self.streak[self.char_height:, :, :]
            mask = self.streak[:, :, 1] > 0
            self.streak[:, :, 1][mask] -= self.dec_color

            self.streak_y_pos += self.char_height
            self.out = self.streak

        self.updates += 1

        return self.streak_x_pos, self.streak_y_pos, self.out

    def __repr__(self):
        return f"Streak at x position of {self.streak_x_pos}"


def run_matrix_flat(window_height=500,
                window_width=1200,
                char_height=20,
                char_width=20,
                max_new_streaks=2,
                spf=0.05):
    """
    spf: seconds per frame, not exact at all since the computation takes time as well
    """
    window = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    random_positions = (window_width // char_width) - 1

    streak_list = []
    x_pos_list = []
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

            streak_list.append(Streak(window_height, rand_x_pos, char_height, char_width))

        for s in streak_list[::-1]: # reverse traverse since we are removing elements while traversing, else it'll create annoying bug
            x_pos, y_pos, streak = s.update()
            if streak.shape[0] == 0:
                streak_list.remove(s)
                x_pos_list.remove(x_pos)
                continue

            window[y_pos:y_pos+streak.shape[0], x_pos:x_pos+streak.shape[1], :] = streak

        cv2.imshow('Matrix', window)
        time.sleep(spf)
        window[:, :, :] = 0 # resetting the window

        if cv2.waitKey(1) == ord('q'):
            break

def main():
    run_matrix_flat(max_new_streaks=3)

if __name__ == "__main__":
    main()