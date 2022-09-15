import os
import glob
# 필요 없는 듯
path="/train"
cat = glob.glob(path+"/BTS 슈가"+'/*')

def rename_cat(files):
        for i,f in enumerate(files):
            os.rename(f, os.path.join(path+"/BTS 슈가", 'cat_' + '{0:03d}.jpg'.format(i)))
        cat = glob.glob(path+"/BTS 슈가"+'/*')
        print("cat {}번째 이미지 성공".format(i+1))

rename_cat(cat)