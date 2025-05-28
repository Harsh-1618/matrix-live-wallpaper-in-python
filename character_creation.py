"""
Program to extract characters from a block of characters.
Refer to character_images directory to see some examples of acceptable images.
Note that the characters of whatever language you are using should not be 
discontinuous horizontally, as in horizontally flipped 'i', where the dot
will be considered as a seperate character by the program.
However, vertical discontinuous characters are fine.
Also, ENSURE that image is white text (255,255,255) on black (0,0,0) background!!
and capture the image such that the individual characters are 50x40 to
minimize resize destortion.
"""

import os
import cv2
import numpy as np


def row_filtering(character):
    # iterating rows of character to eliminate black space with padding of 2
    row_start = False
    start_row_number = None
    end_row_number = None
    for i in range(len(character)):
        if row_start and (not np.all(character[i, :] == 0)):
            end_row_number = i + 2
            continue

        if not np.all(character[i, :] == 0):
            row_start = True
            start_row_number = i - 2
    return character[start_row_number:end_row_number, :]

def character_seperation(characters_image):
    # iterating columns to seperate characters
    characters = []
    character_start = False
    start_col = None
    for j in range(len(characters_image[0])):
        if character_start and (np.all(characters_image[:, j] == 0)):
            characters.append(row_filtering(characters_image[:, start_col:j+2]))
            character_start = False
            start_col = None
            continue

        if (not character_start) and (not np.all(characters_image[:, j] == 0)):
            character_start = True
            start_col = j - 2 # padding of 2 will be there for all sides
    return characters

def main(characters_path, output_folder_name, resize_height, resize_width):
    characters_image = cv2.imread(characters_path)

    # Making image to contain only pure black and white pixels
    characters_image = cv2.cvtColor(characters_image, cv2.COLOR_BGR2GRAY)
    characters_image[characters_image > 127] = 255
    characters_image[characters_image < 128] = 0

    characters = character_seperation(characters_image)

    print(f"Extracted {len(characters)} characters from the image")

    if not os.path.exists(out_fol:=os.path.join("./characters_seperated", output_folder_name)):
        os.makedirs(out_fol)

    for i, character in enumerate(characters):
        resize_character = cv2.resize(character, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST) # not using INTER_CUBIC as it'll introduce values other than 0 and 255
        cv2.imwrite(os.path.join(out_fol, f"{i}.png"), resize_character)


if __name__ == "__main__":
    character_image = "english_capital.png" # only the name!
    characters_path = os.path.join("./character_images", character_image)
    main(characters_path, character_image.split(".")[0], 50, 40)
