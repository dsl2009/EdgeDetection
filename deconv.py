import cv2
import numpy as np
import os
import shutil

def tt():
    org = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/new_tree/org/813.png')
    img = cv2.imread('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/new_tree/gray/813.png', 0)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(img, 100, 200)
    image, contours, her = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cnt = np.squeeze(cnt, 1)
    print cnt.shape
    print cnt[0]
    print cnt
    cv2.drawContours(org, contours, -1, (0, 0, 255), 3)
    cv2.imwrite('ds.jpg', org)

tt()

def hh():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/xinjiang/vectorTile/21/'
    am = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/xinjiang/rasterTile/21'
    for dr in os.listdir(rt):
        am_dir = os.path.join(am, dr)
        root_dir = os.path.join(rt, dr)
        if os.path.exists(am_dir):
            for img in os.listdir(root_dir):
                to_copy = os.path.join(root_dir, img)
                aim_dir = os.path.join(am_dir, img.replace('.', '_0.'))
                print to_copy
                print aim_dir
                shutil.copy(to_copy, aim_dir)






