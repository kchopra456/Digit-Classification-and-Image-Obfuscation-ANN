'''

Snippet responsible for converting an image to Gray Scale.

 User can input any image file.
 Rename it as input in the same directory.
 Take the output file i.e. IMG_GRAY.png

 Paste it in the main directory and run Classify Image.

 It would identify the digit only after the autorun system file was executed once successfully.

'''


from PIL import Image
from PIL import ImageOps
import scipy.misc
import numpy as np
import scipy.io as scio

img = Image.open('input.jpg')


img = img.resize((20,20), Image.ANTIALIAS)
img=img.convert('L',(0.2989, 0.5870, 0.1140, 0))
img = ImageOps.invert(img)
pix_val= list(img.getdata())     # Get the pixel densities from the blurred image.
x=np.matrix(pix_val)
img.save('IMG_GRAY.png')
print 'Conversion Successful...'