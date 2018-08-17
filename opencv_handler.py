import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob

def covert_gray():
    ads = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/tea/pix2pix/*.jpg')
    for img in ads:
        ig = cv2.imread(img)
        a = np.sum(ig, axis=2)
        if np.std(a) < 2:
            continue
        ig = cv2.cvtColor(ig,cv2.COLOR_BGR2GRAY)
        if int(img.split('/')[-1].split('.')[0])<1300:
            ret, thresh1 = cv2.threshold(ig, 175, 255, cv2.THRESH_BINARY_INV)
            cv2.imwrite(img.replace('pix2pix','gray'),thresh1)
        else:
            ret, thresh1 = cv2.threshold(ig, 1, 255, cv2.THRESH_BINARY)
            cv2.imwrite(img.replace('pix2pix', 'gray'), thresh1)


def covert_gray_htd():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/htd_train/gray/'
    ads = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/HTD512/*/*_lab.png')
    for img in ads:
        ig = cv2.imread(img)
        a = np.sum(ig,axis=2)
        if np.std(a)<2:
            continue

        ig = cv2.cvtColor(ig,cv2.COLOR_BGR2GRAY)


        ret, thresh1 = cv2.threshold(ig, 1, 255, cv2.THRESH_BINARY)

        name = rt+img.split('/')[-1].replace('_lab','')

        cv2.imwrite(name,thresh1)

covert_gray()