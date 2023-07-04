
import os
import shutil

dataset_path = '/media/lan/TOSHIBA EXT/video_data/'
frames_path = '/media/lan/TOSHIBA EXT/video_frames/'
new_path = '/media/lan/TOSHIBA EXT/yolo_data/val/images/'

if __name__ == '__main__':
    categories = ['awake', 'bite', 'drink', 'eat', 'forage', 'sleep']
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    print(dict_categories)

    count = 0
    for i in range(len(categories)):
        list = os.listdir(frames_path+categories[i])
        for file in list:
            if 'output' in file:
                folder = file.split('.')
                shutil.copyfile(os.path.join(frames_path, categories[i], folder[0],"img_00050.jpg"), os.path.join(new_path, str(count)+".jpg"))
                count += 1
    
    
