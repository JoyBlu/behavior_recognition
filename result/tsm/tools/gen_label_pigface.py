
import os


dataset_path = '/media/lan/TOSHIBA EXT/video_4s/video_frames/'
label_path = '/media/lan/TOSHIBA EXT/video_4s/video_label'

if __name__ == '__main__':
    categories = ['bite', 'chew', 'drink', 'eat', 'movehead', 'sleep']
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    print(dict_categories)

    files_output = ['origin.txt', 'val_videofolder.txt', 'train_videofolder.txt']
    output = []
    for i in range(len(categories)):
        path = dataset_path+categories[i]
        list = os.listdir(path)
        list.sort()
        for file in list:
            filelist = os.listdir(os.path.join(path, file))
            filelist.sort()
            count = 0
            for photo in filelist:
                if photo == 'img_00001.jpg':
                    break
                count += 1
                os.rename(os.path.join(path, file, photo), os.path.join(path, file, 'img_%05d.jpg'%count))
            #output.append('%s %d %d'%(os.path.join(categories[i], file), len(filelist), i))
            output.append('%s %d %d'%(os.path.join(categories[i], file), 100, i))
    
    with open(os.path.join(label_path, files_output[0]),'w') as f:
        f.write('\n'.join(output))
