'''

Snippet responsible for training ANN for detection of blurred images actual pixel densities.
 X- The matrix data set for blurred images, i.e. the blurred images pixel densities.
 Y- The matrix data set of the actual image pixel densities, wrt what we will regression test our data.

'''
import numpy as np
import scipy.io as scio
import scipy.optimize as op


''' ### SIGMOID FUNCTION ###'''

def sigmoid(x):
    return 1.0/(1+np.exp(-x))


''' @@@@@@@@@@@@@@@@@@@@@@@ '''

''' ### SIGMOID GRADIENT ### '''

def siggradient(x):
    tempg=sigmoid(x)
    return np.multiply(tempg,(np.ones(tempg.shape)-tempg))


''' @@@@@@@@@@@@@@@@@@@@@@@@ '''

''' ### RANDON INITIALIZE THETA ### '''
def randominit():
    global epsilon,theta2,theta1

    theta1=np.random.rand(theta_shape[0][0],theta_shape[0][1])*2*epsilon-epsilon
    theta2=np.random.rand(theta_shape[1][0],theta_shape[1][1])*2*epsilon-epsilon


''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''

''' ### Gradient calculation ### '''

def costgrad(theta):
    global m,X,Y,lamda,theta1,theta2,theta_shape
    theta1_grad=np.zeros(theta_shape[0])
    theta2_grad=np.zeros(theta_shape[1])

    theta1=np.reshape(theta[0:theta_shape[0][0]*theta_shape[0][1]],theta_shape[0])
    theta2=np.reshape(theta[theta_shape[0][0]*theta_shape[0][1]:] ,theta_shape[1])

    A1=np.transpose(X)

    Z2=np.dot(theta1,A1)
    A2=sigmoid(Z2)
    A2=np.append(np.ones((1,A2.shape[1])),A2,axis=0)

    Z3=np.dot(theta2,A2)
    A3=sigmoid(Z3)

    h=np.transpose(A3)

    Z2mod=np.transpose(Z2)
    Z2mod=np.append(np.ones((Z2mod.shape[0],1)),Z2mod,axis=1)

    A1mod=np.matrix(np.transpose(A1))
    A2mod=np.matrix(np.transpose(A2))

    for record in range(0,X.shape[0]):
        DEL3=h[record,:]-Y[record,:]

        DEL2=np.multiply(np.dot(np.transpose(theta2),np.transpose(DEL3)),np.transpose(siggradient(Z2mod[record,:])))

        DEL2=np.transpose(np.matrix(DEL2[1:]))
        DEL3=np.transpose(np.matrix(DEL3))

        theta1_grad=theta1_grad + np.dot(np.transpose(DEL2),A1mod[record,:])
        theta2_grad=theta2_grad + np.dot(DEL3,A2mod[record,:])

    theta1_grad=theta1_grad/m
    theta2_grad=theta2_grad/m

    theta1_grad[:,1:]=theta1_grad[:,1:] + lamda/m * theta1[:,1:]
    theta2_grad[:,1:]=theta2_grad[:,1:] + lamda/m * theta2[:,1:]

    theta1_gradmod=np.transpose(theta1_grad.reshape(theta1_grad.shape[0]*theta1_grad.shape[1]))
    theta2_gradmod=np.transpose(theta2_grad.reshape(theta2_grad.shape[0]*theta2_grad.shape[1]))

    thetanew= (np.append(theta1_gradmod,theta2_gradmod,axis=0))
    thetanew=thetanew.A1
    return thetanew

''' @@@@@@@@@@@@@@@@@@@@ '''

''' ### COST FUNCTION ### '''

def costcal(theta):
    global m,X,Y,lamda,theta_shape,iteration

    JCost=0     # Initialize the cost of the epoch to be 0.
    ''' Carve out the thetas from the single linear vector theta. '''
    theta1=np.reshape(theta[0:theta_shape[0][0]*theta_shape[0][1]],theta_shape[0])
    theta2=np.reshape(theta[theta_shape[0][0]*theta_shape[0][1]:] ,theta_shape[1])

    A1=np.transpose(np.matrix(X))

    Z2=np.dot(theta1,A1)
    A2=sigmoid(Z2)
    A2=np.append(np.ones((1,A2.shape[1])),A2,axis=0)    # Adding the bias units to the hidden layer.

    Z3=np.dot(theta2,A2)
    A3=sigmoid(Z3)

    h=np.transpose(A3)  # (hypothesis) Output of the NN.

    ''' Get the cost for each record '''
    for rec in range(0,h.shape[0]):
        hvec=h[rec,:]
        yvec=Y[rec,:]

        Jk=np.sum((-np.multiply(yvec,np.log(hvec)) - np.multiply((np.ones(yvec.shape) - yvec) , np.log(np.ones(hvec.shape)-hvec)))*1.0/m)

        JCost=JCost+Jk

    ''' Regularization of the parameters (to avoid over-fitting) '''
    reg=(lamda*1.0/(2*m))*(np.sum(np.power(theta1,2))+np.sum(np.power(theta2,2)))
    JCost=JCost+reg

    print "Iteration "+ str(iteration)+'...'
    print 'Cost '+ str( JCost )
    iteration += 1
    return JCost

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@'''

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

''' ### GET COST OVER THE RECORDS  ###'''



def main():

    global theta1,theta2,iteration,theta_shape,X,Y,iteration,MAX_ITER,m,lamda,epsilon
    print 'Begin training NN for blur images... (wait)'

    ''' ### LOAD THE DATA INTO THE X AND Y MATRICES ####'''

    try:
        data=scio.loadmat("Train_set_blur.mat")

        X=np.matrix(data['X_train'])
        Y=np.matrix(data['Y_train'])

        iteration=0     # The iteration number for the optimization function.
        MAX_ITER=150    # The iterations our optimization function will cover before stopping.

        theta_shape=[(25,401),(400,26)]
        theta1=np.zeros(theta_shape[0])   # Parameter matrix for the first layer of the NN.
        theta2=np.zeros(theta_shape[1])   # Parameter matrix for the second layer of the NN.
    except:
        print 'Error encountered in Detect_Blur...'
        print 'Required files not available in the directory...(View Code snippet for more information)'
        exit()

    m=X.shape[0]    # Number of the records.
    lamda=1         # Learning Rate.
    epsilon=0.12    # Regularization constant.

    ''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ '''


    randominit()                # Randomly initialize the parameters.

    ''' Generate a linear parameter vector to pass to the optimization function. '''
    theta1mod=theta1.reshape(theta_shape[0][0]*theta_shape[0][1])
    theta2mod=theta2.reshape(theta_shape[1][0]*theta_shape[1][1])
    theta=np.transpose((np.append(theta1mod,theta2mod,axis=0)))

    iteration=0

    res=op.fmin_cg(costcal,theta,fprime=costgrad,maxiter=MAX_ITER) # Call for the cost function optimization.
    #op.minimize(costcal,theta,method='CG',jac=True)

    theta=res     # Returned parameter values (Optimized).
    theta1=np.reshape(theta[0:theta_shape[0][0]*theta_shape[0][1]],theta_shape[0])
    theta2=np.reshape(theta[theta_shape[0][0]*theta_shape[0][1]:] ,theta_shape[1])

    ''' Save the learned parameters for testing and classification over the whole data set. '''
    scio.savemat("thetas_blur",{"theta1":theta1,"theta2":theta2})

   # print 'Accuracy of the NN over the train set is....'
   # print accuracy()

    print 'Training NN for blur images complete...'
    print 'Trained parameters saved as theta_blur'

'''@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'''



''' ### ACCURACY OF THE NETWORK ### '''

def accuracy():

    predict=np.zeros((1,Y.shape[1]))

    for record in range(0,X.shape[0]):

        hypothesis=Rtest(np.matrix(X[record,:]))
    #   hypothesis[np.where(hypothesis>0.5)]=1
    #   hypothesis[np.where(hypothesis<=0.5)]=0

        predict=np.append(predict,hypothesis,axis=0)

    predict=predict[1:,:]
    predictflat=predict.reshape((predict.shape[0]*predict.shape[1]))
    Yflat=Y.reshape((Y.shape[0]*Y.shape[1]))

    return len(np.where(Yflat==predictflat)[1])*100.0/(m*400)  # Total pixel values is m*400 (# of records * pixel in one record)

''' @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'''

if __name__=='__main__':main()
