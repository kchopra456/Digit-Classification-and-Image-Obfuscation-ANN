
'''

Snippet for pre-processing of data that was retrieved after BLUR CLASSIFICATION or for image data without blurring.

X- The input features from the pixel densities predicted by the ANN for BLUR DETECTION
X- Or the image pixel densities values directly from the image matrix.

Y- The classification of the digit i.e. 0-9.

'''


import numpy as np
from sklearn.utils import shuffle
import scipy.io as scio
from pylab import scatter
import matplotlib.pyplot as plt


''' ### Visualize the data  ### '''
def visualize(X,rec):

    i=19
    j=19
    X=np.transpose(np.matrix(X[rec,1:]))

    x=X.reshape((20,20))
    x=x.transpose()
    plt.imshow(x,cmap='Greys_r')
    plt.show()

    while i>=0:
        j=19
        while j>=0:
            if x[i,j]!=0:
                scatter(i,j)
            j-=1
        i-=1
    plt.xlim(0,40)
    plt.ylim(0,30)

''' @@@@@@@@@@@@@@@@@@@ '''



def main():

    print "Pre-processing of image data... (wait)"


    ''' ### LOAD THE DATA INTO THE X AND Y MATRICES ####'''
    try:
        print 'Looking for blurred pre-processed data...'
        data=scio.loadmat("data_DR.mat")
    except:
        data=scio.loadmat('image_dataset')
        print ' (Blurred pre-processed data not available.) '
        print ' (Digit classification ANN using input data as the image matrics without blurring...) '

    X=np.matrix(data['X'])
    X=np.append(np.ones((X.shape[0],1)),X,axis=1)   # Adding the bias units (feature/pixel) to the input data (input layer).

    Y=np.matrix(data['y'])
    Ymod=np.zeros((X.shape[0],10))                  # Y modification- represent output as a vector of 10.
                                                    # 0 is given by [1,0,0,0,0,0,0,0,0,0,0]

    ''' Set 1's at the location '''
    for record in range(0,Y.shape[0]):
        Ymod[record,Y[record,0]%10]=1

    ''' ###  Train and Test Set     ### '''

    X,Ymod=shuffle(X,Ymod)

    X_test=np.matrix(X[0:X.shape[0]*0.3,:])
    Y_test=np.matrix(Ymod[0:Ymod.shape[0]*0.3,:])


    X_train=np.matrix(X[X.shape[0]*0.3:,:])
    Y_train=np.matrix(Ymod[Ymod.shape[0]*0.3:,:])
    ''' @@@@@@@@@@@@@@@@@@@@@@@@@@ '''

    print 'Division of data complete ...'

    scio.savemat("Test_set",{"X_test":X_test,"Y_test":Y_test})
    scio.savemat("Train_set",{"X_train":X_train,"Y_train":Y_train})

    print 'Pre-processing complete.'

if __name__ == '__main__':main()