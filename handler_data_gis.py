import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import h5py
root = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/bench/dataset/'
from PIL import Image
fd = []
import tensorflow as tf
image_size = 352
def native_build( path,dim):
    iv1 = cv2.imread(path)
    iv1 = cv2.cvtColor(iv1, cv2.COLOR_BGR2RGB)
    bc = np.asarray(iv1, dtype=np.float32) / 255

    w, h, c = bc.shape
    if min([w,h]) < image_size:
        return None,1,2
    ofh = int((h-image_size)/2)
    ofw = int((w-image_size)/2)
    bbox_h_start = ofh
    bbox_w_start = ofw

    bbox_h_size = h - bbox_h_start * 2
    bbox_w_size = w - bbox_w_start * 2

    img = bc[bbox_w_start:bbox_w_start + image_size, bbox_h_start:image_size + bbox_h_start]

    image = img - 0.5
    image = image * 2.0

    image = np.reshape(image, (1, image_size, image_size, dim))
    return image,bbox_h_start,bbox_w_start

#train_file = h5py.File('train.hdf5','w')
#images = []
#masks = []
#betas =[]


for s in open(root+'train.txt').readlines():
    fd.append(s.replace('\n',''))
total_len = len(fd)
shuf = np.random.permutation(total_len)


def get_data(index):
    global shuf
    if index>=total_len:
        shuf = np.random.permutation(total_len)
        index = 0
    s = fd[shuf[index]]
    data = sio.loadmat('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/bench/dataset/inst/' + s.replace('\n','') + '.mat')
    val = data['GTinst'][0, 0]
    org_image, h_st, w_st = native_build('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/bench/dataset/img/' + s.replace('\n', '') + '.jpg',3)
    if org_image is None:
        return None,None,None,index+1

    val['Boundaries'][-1, 0].toarray()

    bon = val['Boundaries'][0, 0].toarray()
    cages = np.squeeze(val['Categories'], 1)
    bon[np.where(bon != 0)]
    mask = np.zeros(shape=(image_size, image_size, 20))
    betas = []
    for cls in range(20):
        beta = 1
        if cls in cages:
            all_cage_index = np.where(cls == cages)
            bd = None
            for idx in all_cage_index[0]:
                if bd is None:
                    bd = val['Boundaries'][idx, 0].toarray()
                else:
                    bd = bd + val['Boundaries'][idx, 0].toarray()

            bd = bd[w_st:w_st + image_size, h_st:image_size + h_st]
            beta = 1 - np.where(bd != 0)[0].shape[0] * 1.0 / image_size/image_size
            mask[:, :, cls] = bd
            plt.imshow(bd)
            plt.show()

        else:
            beta = 1
        betas.append(beta)
    mask = np.reshape(mask, (1, image_size, image_size, 20))
    return org_image,mask,min(betas),index+1

#hbd_root = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/HED-BSDS/'

hbd_root = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/space/annotations/'
train_list = open(hbd_root+'trainval.txt').readlines()
images = []
labels = []
for s in train_list:
    dts = s.replace('\n','').split(' ')

    images.append(dts[0])

    labels.append(dts[0].replace('.jpg','segobj.png'))

print len(train_list)

def get_hed(idsx):
    if idsx>len(images)-1:
        idsx = 0
    im = Image.open(images[idsx])
    em = Image.open(labels[idsx])
    org_im = im
    im = im.resize((512, 512))
    #em = em.resize((512, 512))

    im = np.array(im, dtype=np.float32)
    im = im[:, :, [2,1,0]]
    im -= [103.939, 116.779, 123.68]

    # Labels needs to be 1 or 0 (edge pixel or not)
    # or can use regression targets as done by the author
    # https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/src/caffe/layers/image_labelmap_data_layer.cpp#L213

    em = np.array(em.convert('L'), dtype=np.float32)

    if True:
        bin_em = em / 255.0
        if np.sum(bin_em) == 0:
            return get_hed(idsx+1)

    else:
        bin_em = np.zeros_like(em)
        bin_em[np.where(em)] = 1

    # Some edge maps have 3 channels some dont
    bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
    # To fit [batch_size, H, W, 1] output of the network
    bin_em = np.expand_dims(bin_em, 2)
    im = np.expand_dims(im,0)
    bin_em = np.expand_dims(bin_em,0)
    return org_im,im,bin_em,idsx+1,images[idsx]

def get_test_data(path):
    im = Image.open(path)
    im = im.resize((512, 512))
    im = np.array(im, dtype=np.float32)
    im = im[:, :, [2,1,0]]
    im -= [103.939, 116.779, 123.68]
    im = np.expand_dims(im, 0)
    return im






