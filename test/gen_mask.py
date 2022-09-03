import cv2
import numpy as np
import math

arry = np.zeros((256,256),np.uint8)
center=(128,128)
for k in [0,5,10,15,20]:
    val=int(math.sqrt((256*256*k/100))/2)
    for i in range(128-val,128+val):
        for j in range(128-val,128+val):
            arry[i][j]=255
    # cv2.imshow('{}'.format(k),arry)
    # cv2.waitKey(0)
    cv2.imwrite(f'./test_masks/{k}_mask.png',arry)