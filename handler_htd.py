import shutil
import os
import glob
a = glob.glob('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/HTD512/*/*.png')
for s in a:
    if s.endswith('edge.png'):
        shutil.copy(s,'/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/htd_train/edge')
    elif s.endswith('lab.png'):
        pass
    else:
        shutil.copy(s, '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/edge_detect/htd_train/org')