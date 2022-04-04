import os
from PIL import Image

DIR = '16-bit-to-8-bit'

if __name__ == '__main__':
    for ff in os.listdir(DIR):
        if '.png' in ff:
            path = os.path.join(DIR, ff)
            Image.open(path).convert('RGB').save(path.replace('.png', '.jpg'), 'JPEG')
            print(f'Converted {ff}')
