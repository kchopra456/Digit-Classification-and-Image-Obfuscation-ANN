'''

Snippet responsible for pipe lining data from ANN (Blur Detection) to digit classification ANN.

Here we use the parameter matrices that we achieved after training over the records to detect pixel values.

'''

import scipy.io as scio
import numpy as np




''' ### Regression test a record ### '''

def Rtest(Xvec):
    global theta1,theta2

    A1=np.transpose(Xvec)

    Z2=np.dot(theta1,A1)
    A2=sigmoid(Z2)
    A2=np.append(np.ones((1,A2.shape[1])),A2,axis=0)

    Z3=np.dot(theta2,A2)
    A3=sigmoid(Z3)

    h=np.transpose(A3)
    return h

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''

''' ### SIGMOID FUNCTION ###'''

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''

''' ### SIGMOID GRADIENT ### '''

def siggradient(x):
    tempg=sigmoid(x)
    return np.multiply(tempg,(np.ones(tempg.shape)-tempg))

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''


def main():
    global thetas,theta1,theta2,X,Y

    print 'Activating the pipeline... (wait)'


    try:
        ''' Load the parameters '''

        thetas=scio.loadmat("thetas_blur")
        theta1=thetas["theta1"]
        theta2=thetas["theta2"]

        ''' ### LOAD THE DATA INTO THE X AND Y MATRICES ### '''
        dataX=scio.loadmat('data_blur.mat')
        X=np.matrix(dataX['X'])
        X=np.append(np.ones((X.shape[0],1)),X,axis=1)   # Add the bias units to the input layer for ANN.

        dataY=scio.loadmat('ex4data1')                  # Y is the actual digit values i.e. 0-9.
        Y=np.matrix(dataY['y'])

        ''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''

    except:
        print 'Error encountered in ANN_Pipeline...'
        print 'Required files not available in the directory...(View Code snippet for more information)'
        exit()


    X_save=np.zeros((1,400))                        # Output layer matrix.
    for rec in range (X.shape[0]):
        X_save=np.append(X_save,Rtest(np.matrix(X[rec,:])),axis=0)
        #print 'Working with record ' + str(rec) + '...'

    X_save=X_save[1:,:]
    scio.savemat('data_DR',{'X':X_save,'y':Y})

    print 'Pipeline complete.'
    print 'Now blurred image data avilable for digit classification ANN.'

if __name__ == '__main__':main()
