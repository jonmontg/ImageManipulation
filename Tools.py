import numpy as np
import array
from scipy.ndimage import gaussian_filter


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


#  Returns a 2d numpy array of pixel intensities derived from the given matrix. This edge detector finds the difference
#  between an image and its copy with an applied gaussian filter
def get_edges(matrix):
    return np.subtract(gaussian_filter(matrix), matrix)


#  Returns a 2d numpy array based on the given matrix. The new matrix has 1/2 the width and 1/2 the height of the given
#  matrix. Each block of 4 pixels is compressed to an average in the new matrix.
def convolute_and_comptress(matrix):
    shape = matrix.shape
    new_shape = (int(shape[0]/2), int(shape[1]/2))
    compressed = np.zeros(shape=new_shape)

    for i in range(len(compressed)):
        for j in range(len(compressed[i])):
            compressed[i][j] = (matrix[2*i][2*j] +
                                matrix[2*i+1][2*j] +
                                matrix[2*i+1][2*j+1] +
                                matrix[2*i][2*j+1])/4
    return compressed
