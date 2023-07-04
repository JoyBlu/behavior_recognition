import os
from PIL import Image


if __name__ == '__main__':

    im = Image.open('/media/lan/TOSHIBA EXT/yolo_data/val/images/2.jpg')
    print(im.size)
    im.thumbnail((320,320))
    print(im.size)
