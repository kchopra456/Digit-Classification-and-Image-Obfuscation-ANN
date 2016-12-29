import numpy as np
import scipy.io as scio


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



''' ### ACCURACY OF THE NETWORK ### '''

def accuracy():

    data=scio.loadmat("Test_set")
    X_test=np.matrix(data['X_test'])
    Y_test=np.matrix(data['Y_test'])
    predict=np.zeros((1,Y_test.shape[1]))
    m=X_test.shape[0]

    for record in range(0,X_test.shape[0]):

        hypothesis=Rtest(np.matrix(X_test[record,:]))
        hypothesis[np.where(hypothesis>0.5)]=1              # Set 1 for the class with highest probability of lying into.
        hypothesis[np.where(hypothesis<=0.5)]=0             # Set 0 for all other classes.
        ''' Only a single class allegiance will be found if trained well i.e. only one value will be above 0.5'''
        predict=np.append(predict,hypothesis,axis=0)
    predict=predict[1:,:]
    return len(np.where(Y_test==predict)[1])*100.0/(m*10)   # Total pixel values is m*10 (# of records * size of a single vector)

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'''


def main():
    global theta1,theta2,X_test,Y_test
    print 'Checking the accuracy of the Digit Classification ANN over the Test Set...'


    ''' ### LOAD THE DATA INTO THE X AND Y MATRICES ### '''
    try:
        data=scio.loadmat('Test_set')

        X_test=data['X_test']
        Y_test=data['Y_test']

        theta=scio.loadmat('thetas_digit')
        theta1=theta['Theta1']
        theta2=theta['Theta2']

    except:
        print 'Error encountered in Digit_Classification_TEST...'
        print 'Required files not available in the directory...(View Code snippet for more information)'
        exit()

    print 'Accuracy of the NN over the train set is....'
    print accuracy()

    print 'Check complete...'




if __name__ == '__main__':main()

