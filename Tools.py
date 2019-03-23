import numpy as np
import array


#  Returns a 1536 (horizontal) by 1024 (vertical) matrix of linearly scaled pixel intensities. The linear scaling
#  is determined by the camera settings. Each pixel is a 2 byte unsigned int. This function is based on the loading
#  algorithm provided by Bethge Lab
def get_matrix(filepath):
    with open(filepath, 'rb') as handle:
        s = handle.read()
    arr = array.array('H', s)
    arr.byteswap()
    mat = np.array(arr, dtype=np.float16).reshape(1024, 1536)
    return np.divide(mat, np.amax(mat))

