import requests
import json
import cv2
import os
import numpy as np
import glob

def down_load():
    for s in open('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/ai.json').readlines():
        data = json.loads(s.replace('\n', ''))
        urls = data['pic_url']
        name = '_'.join(urls.split('/')[-3:]) + '.jpg'
        mask_name = os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/building/gray', name)
        name = os.path.join('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/building/org', name)
        r = requests.get(urls)
        ig = open(name, 'wb')
        ig.write(r.content)
        ig.flush()
        bun = data['boundary']
        mask = np.zeros(shape=(256, 256, 3))

        for b in bun:
            pts = []
            for p in b:
                pts.append([int(p['pix_x']), int(p['pix_y'])])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], color=(255, 255, 255))
        cv2.imwrite(mask_name, mask)


def gen():
    rt = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/building'
    f = open(rt+'/train_gray.txt','w')
    if False:
        for s in glob.glob(rt+'/org/*.jpg'):
            ig = cv2.imread(s)
            cv2.imwrite(s,ig)
    for s in os.listdir(rt+'/gray'):
        content = 'org/'+s+' '+'gray/'+s+'\n'
        f.write(content)
        f.flush()
gen()




