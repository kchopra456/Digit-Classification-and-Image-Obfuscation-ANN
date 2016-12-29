'''

Snippet responsible for creating the blurred images, by loading the image dataset.
X- Represents the feature matrix for the blurred out images, that we aim to classify.
Y- Represents the feature matrix for the actual images, the target image; corresponding to the image blurred.
'''



from PIL import Image
import ImageFilter
import numpy as np
import scipy.io as scio


X=np.zeros((1,400))
Y=np.zeros((1,400))

def main():
    global X,Y
    print "Blurring Action begin... (wait)"

    try:
        for rec in range(5000):  # 5000 is the Data Set Size
            image=Image.open("./Images Dataset/Images/output"+str(rec)+".jpg")
            #blur_image=image.filter(ImageFilter.BLUR)
            blur_image=image.filter(ImageFilter.GaussianBlur(radius=2)) # Set the blur radius.
            #Image._show(blur_image)

            pix_val= list(blur_image.getdata())     # Get the pixel densities from the blurred image.
            x=np.matrix(pix_val)
            X=np.append(X,x,axis=0)
            pix_val=list(image.getdata())
            y=np.matrix(pix_val)
            Y=np.append(Y,y,axis=0)
    except:
        print 'Error encountered in Create_blur_images...'
        print 'Required files not available in the directory...(View Code snippet for more information)'
        exit()
    X=X[1:,:]
    Y=Y[1:,:]

    X_max=np.matrix(np.max(X,axis=0))
    X_min=np.matrix(np.min(X,axis=0))

    Y_max=np.matrix(np.max(Y,axis=0))
    Y_min=np.matrix(np.min(Y,axis=0))

    ''' ### Feature normalize the blurred image values ###'''

    # All values will lie between 0 - 1.

    for col in range(X.shape[1]):
        X[:,col]=(X[:,col]-X_min[0,col])*1.0/(X_max[0,col]-X_min[0,col])
        Y[:,col]=(Y[:,col]-Y_min[0,col])*1.0/(Y_max[0,col]-Y_min[0,col])

    ''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''

    scio.savemat("data_blur.mat",{"X":X,"Y":Y})

    print "Blurring Action complete..."

if __name__ == '__main__':main()
