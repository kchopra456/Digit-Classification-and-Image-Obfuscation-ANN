'''

Snippet responsible of conversion of images, available in the pixel matrix format to jpg files.
Load the data from the ex4data1 file and output the 5000 jpg image files.

'''

import scipy.misc
import numpy as np
import scipy.io as scio
from PIL import Image


try:
    data=scio.loadmat("image_dataset.mat")

    X=np.matrix(data['X'])
except:
    print 'Error encountered in Convert_to_images...'
    print 'Required files not available in the directory...(View Code snippet for more information)'
    exit()


def main():
    
    print "Conversion to Image Action begin... (wait)"

    for x in range(X.shape[0]):

        image_array=X[x,:]
        image_array=image_array.reshape((20,20))
      #  image_array=image_array.transpose()
      #  print image_array.shape

        # print "Saving Image "+str(x)+"..."
        scipy.misc.toimage(image_array, cmin=0.0,).save("./Images Dataset/Images/output"+str(x)+".jpg")


    print "Converstion to Image Action complete..."

if __name__ == '__main__':main()
