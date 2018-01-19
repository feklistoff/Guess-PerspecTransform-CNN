import random
import numpy as np
import cv2
import glob
from PIL import Image, ImageDraw, ImageFont
import markovify


WORD_TERMINATORS = ['.', '!', '?']
SPLIT_SYMBOLS = [',', ';']
VALID_SYMBOLS = WORD_TERMINATORS + SPLIT_SYMBOLS

# random parameters
def get_txt_params(font_list):
    num_sent = int(random.uniform(5, 10))
    fnt_sz   = int(random.uniform(15, 25))
    fnt      = random.choice(font_list)
    l_space  = random.uniform(1, 2.5)
    paragrph = int(random.uniform(0, 3))
    return num_sent, fnt_sz, fnt, l_space, paragrph


# generate text
def get_text(text_model, font_list):
    num_sent, fnt_sz, fnt, l_space, prgph = get_txt_params(font_list)

    font = ImageFont.truetype(fnt, fnt_sz)
    sent = ''
    for j in range(num_sent):
        tmp = text_model.make_sentence()
        sent += (' ' + tmp)
    return sent, font, fnt_sz, l_space, prgph


def text_wrap(drawer, text, font, page_width):
    text_width = page_width * 0.8
    lines = []
    line = ""
    for word in text.split():
        if word in VALID_SYMBOLS:
            line += word
        elif font.getsize(line + word)[0] <= text_width:
            line += ' ' + word
        else:
            lines.append(line)
            line = ""
    return lines


# perspective transformation pipeline
def rnd_range(range):
    pix = range * 0.1
    return int(random.uniform(-pix, pix))


def transform_params(p_w, p_h):
    s_pts = np.array(((0, 0), (p_w, 0), (p_w, p_h), (0, p_h)), np.float32)
    d_pts = np.array(((rnd_range(p_w), rnd_range(p_h)),
                      (p_w + rnd_range(p_w), rnd_range(p_h)),
                      (p_w + rnd_range(p_w), p_h + rnd_range(p_h)),
                      (rnd_range(p_w), p_h + rnd_range(p_h))), np.float32)
    M = cv2.getPerspectiveTransform(s_pts, d_pts)
    return M, d_pts


def warp(pg, p_w, p_h):
    M, d_pts = transform_params(p_w, p_h)
    return cv2.warpPerspective(np.asarray(pg), M, (p_w, p_h)), d_pts


# function to create dataset
def create_samples(num_samples, data, page_width, page_height, phase):
    with open('./austen-emma.txt') as f:
        text = f.read()
    text_model = markovify.Text(text)
    font_list = glob.glob('./fonts/*.ttf')

    for i in range(num_samples):
        sent, font, fnt_sz, l_space, prgph = get_text(text_model, font_list)

        page = Image.new("RGB", (page_width, page_height), color='white')
        drawer = ImageDraw.Draw(page)
        lines = text_wrap(drawer, sent, font, page_width)

        offset = margin = page_width * 0.1
        for j, line in enumerate(lines):
            if offset + fnt_sz >= page_height:
                break
            drawer.text((margin, offset), line, font=font, fill='black')
            if j > 1 and j % 7 == 0 and prgph > 0:
                offset += 3 * fnt_sz * l_space
                prgph -= 1
            else:
                offset += fnt_sz * l_space

        page_warped, d_pts = warp(page, page_width, page_height)
        page_warped[page_warped < 20] = 255  # remove black tiles

        # save image
        cv2.imwrite('./{}/sample{}.png'.format(phase, i), page_warped)
        # save data for csv log
        data['image'].append('./{}/sample{}.png'.format(phase, i))
        data['pts'].append(np.reshape(d_pts, -1))
