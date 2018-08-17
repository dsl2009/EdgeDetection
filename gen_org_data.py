import cv2
import numpy as np
import os
import shutil
def qie(image_path,aim_dir):
    print image_path
    img = cv2.imread(image_path)
    w,h,s = img.shape
    img = cv2.resize(img,dsize=(h/1,w/1))
    sp = img.shape
    i = 0
    j = 0
    image_size = 512
    while image_size * (i + 1) < sp[0]:
        j = 0
        while image_size * (j + 1) < sp[1]:
            print i * image_size, image_size * (i + 1), j * image_size, image_size * (j + 1)
            print img[0:image_size, 0:image_size].shape
            shutil.copy('704.json',aim_dir + str(i) + str(j) + '.json')
            cv2.imwrite(aim_dir + str(i) + str(j) + '.png',
                        img[i * image_size:image_size * (i + 1), j * image_size:image_size * (j + 1)])
            j = j + 1
        i = i + 1

def qie_dir(data_dir,aim_dir):
    if not os.path.exists(aim_dir):
        os.mkdir(aim_dir)
    for idx,img_path in enumerate(os.listdir(data_dir)):
        if '-' in img_path:
            qie(os.path.join(data_dir,img_path),aim_dir+img_path.split('.')[0].split('-')[0])




def new_gen_edge(ip,name,out):
    img = cv2.imread(ip,0)

    edges = cv2.Canny(img, 100, 200)

    image, contours, a = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    ns = np.zeros(shape=(512,512,3))


    ns = cv2.imread(name)

    cv2.drawContours(ns, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(out,ns)


def dfs():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/new_tree/pix2pix/'
    aim = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/new_tree/edge/'
    for s in os.listdir(rt):
        new_gen_edge(rt + s, aim + s)



def gen_train_list():
    fs = open('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/htd_train/'+'train_val.txt','w')
    for s in os.listdir('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/htd_train/org/'):
        fs.write('org/'+s+' '+'edge/'+s.replace('.png','_lab_edge.png')+'\n')
        fs.flush()
    fs.close()
def gen_train_gray():
    fs = open('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/'+'train_gray.txt','w')
    for s in os.listdir('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/gray'):
        if os.path.exists('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/'+'org/'+s):
            fs.write('org/'+s+' '+'gray/'+s+'\n')
            fs.flush()
    fs.close()

#qie_dir('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/guoshu/'
#       ,'/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/new_tree/pix2pix/')
#dfs()
def he_bin():
    root = "/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/land/org/"
    names = []
    image = np.ones((512*3,10*512,3))
    for s in range(0,30):
        if s<10:
            names.append(root+'b0'+str(s)+'.jpg')

        else:
            names.append(root + 'b' + str(s) + '.jpg')

    for  ix,img in enumerate(names):
        print ix//10
        image[(ix//10)*512:(ix//10+1)*512,(ix%10)*512:(ix%10+1)*512,:] = cv2.imread(img)
    cv2.imwrite('gen_org.jpg',image)
#new_gen_edge('gen_land.jpg','gen_org.jpg','gen_land_edg.jpg')

def he_bin():
    root = "/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/land/org/"
    names = []
    image = np.ones((512*3,10*512,3))
    for s in range(0,30):
        if s<10:
            names.append(root+'b0'+str(s)+'.jpg')

        else:
            names.append(root + 'b' + str(s) + '.jpg')

    for  ix,img in enumerate(names):
        print ix//10
        image[(ix//10)*512:(ix//10+1)*512,(ix%10)*512:(ix%10+1)*512,:] = cv2.imread(img)
    cv2.imwrite('gen_org.jpg',image)

def he_bin1():
    root = "/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/org/"
    names = []
    image = np.ones((512,3*512,3))
    for ix,s in enumerate([5,90,156]):
        #ix = s - 704
        image[:,ix*512:(ix+1)*512,:] = cv2.imread(root+str(s)+'.jpg')
        #image[(ix//7)*512:(ix//7+1)*512,(ix%7)*512:(ix%7+1)*512,:] = cv2.imread(glob.glob(root+str(s)+'*_2.jpg')[0])
    cv2.imwrite('gen_guo_org.jpg',image)

#he_bin1()
#new_gen_edge('gen_guo_cir.jpg','gen_guo_org.jpg','guo.jpg')
import glob
def he2(image_path):
    rt = '/var/www/html/rcnn/ss7/'
    print image_path
    img = cv2.imread(image_path)
    fin = np.zeros(shape=img.shape)
    w, h, s = img.shape
    img = cv2.resize(img, dsize=(h / 1, w / 1))
    sp = img.shape
    i = 0
    j = 0
    image_size = 512
    while image_size * (i + 1) < sp[0]:
        j = 0
        while image_size * (j + 1) < sp[1]:
            print i * image_size, image_size * (i + 1), j * image_size, image_size * (j + 1)
            print img[0:image_size, 0:image_size].shape
            #shutil.copy('704.json', aim_dir + str(i) + str(j) + '.json')


            fin[i * image_size:image_size * (i + 1), j * image_size:image_size * (j + 1),:] = \
                cv2.imread(glob.glob(rt + str(i) + str(j) + '*_2.jpg')[0])

            j = j + 1
        i = i + 1
    cv2.imwrite('fin.jpg',fin)
he2('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/guoshu/9.jpg')