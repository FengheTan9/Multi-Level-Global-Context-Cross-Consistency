import os
from glob import glob
from sklearn.model_selection import train_test_split

name = 'busi'

root = r'./data/' + name

img_ids = glob(os.path.join(root, 'images', '*.png'))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]


count = 1
for i in [41, 64, 1337]:
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.3, random_state=i)
    filename = root + '/{}_train{}.txt'.format(name, count)
    with open(filename, 'w') as file:
        for i in train_img_ids:
            file.write(i + '\n')

    filename = root + '/{}_val{}.txt'.format(name, count)
    with open(filename, 'w') as file:
        for i in val_img_ids:
            file.writelines(i + '\n')

    print(train_img_ids)
    print(val_img_ids)
    count += 1
