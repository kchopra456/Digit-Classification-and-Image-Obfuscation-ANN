import numpy as np
import scipy.io as scio
from PIL import Image



''' ### SIGMOID FUNCTION ###'''

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''

''' ### SIGMOID GRADIENT ### '''

def siggradient(x):
    tempg=sigmoid(x)
    return np.multiply(tempg,(np.ones(tempg.shape)-tempg))

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''



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

X=np.zeros((1,400))



def main():
    global theta1,theta2,X_test,Y_test

    ''' ### LOAD THE DATA INTO THE X AND Y MATRICES ### '''
    try:

        img=Image.open('IMG_GRAY.png')
        x=np.matrix(list(img.getdata()))
        X_max=np.matrix(np.max(x))
        X_min=np.matrix(np.min(x))
        x=(x-X_min)*1.0/(X_max-X_min)
        theta=scio.loadmat('thetas_digit')
        theta1=theta['Theta1']
        theta2=theta['Theta2']

    except:
        print 'Error encountered in Digit_Classification_TEST...'
        print 'Required files not available in the directory...(View Code snippet for more information)'
        exit()

    x=np.append(np.ones((1,1)),x,axis=1)
    ret= Rtest(x)
    print ret
    print 'The digit is classified as...'
    print np.where(np.max(ret)==ret)[1]

if __name__ == '__main__':main()


