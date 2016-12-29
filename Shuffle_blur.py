
'''


Snippet for pre-processing of data that was retrieved after BLURRING OF THE IMAGES.

X= The blurred out image pixel densities.

Y= The actual image pixel densities.

'''

import numpy as np
from sklearn.utils import shuffle
import scipy.io as scio

def main():

    print "Pre-processing of blurred image data... (wait)"

    try:

        ''' ### LOAD THE DATA INTO THE X AND Y MATRICES ####'''

        data=scio.loadmat("data_blur.mat")

        X=np.matrix(data['X'])
        X=np.append(np.ones((X.shape[0],1)),X,axis=1) #Adding the bias units (feature/pixel) to the input data (input layer).

        Y=np.matrix(data['y'])

    except:
        print 'Error encountered in Shuffle_blur...'
        print 'Required files not available in the directory...(View Code snippet for more information)'
        exit()


    ''' ###  Train and Test Set ### '''

    # Division of the data into train and test sets.

    X,Y=shuffle(X,Y)

    X_test=np.matrix(X[0:X.shape[0]*0.3,:])
    Y_test=np.matrix(Y[0:Y.shape[0]*0.3,:])


    X_train=np.matrix(X[X.shape[0]*0.3:,:])
    Y_train=np.matrix(Y[Y.shape[0]*0.3:,:])
    ''' @@@@@@@@@@@@@@@@@@@@@@@@@@ '''

    print 'Division of data complete ...'

    scio.savemat("Test_set_blur",{"X_test":X_test,"Y_test":Y_test})
    scio.savemat("Train_set_blur",{"X_train":X_train,"Y_train":Y_train})

    print 'Pre-processing complete.'

if __name__ == '__main__':main()
