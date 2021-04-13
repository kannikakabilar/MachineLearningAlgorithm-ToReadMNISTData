import numpy as np
import numpy.random as rnd
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
import bonnerlib3 as blib
import sklearn.linear_model as lin
import sklearn.neighbors as skln

#------------------------------------------------------
#------ Question 1: Vectorized Code and Indexing ------
rnd.seed(3)

print('\n\nQuestion 1')
print('----------')

print('\nQuestion 1(a):')
B = rnd.random((4,5))
print(B)

print('\nQuestion 1(b):')
y = rnd.random((4,1))
print(y)

print('\nQuestion 1(c):')
C = np.reshape(B,(2,10))
print(C)

print('\nQuestion 1(d):')
D = B - y
print(D)

print('\nQuestion 1(e):')
z = np.reshape(y,(4))
print(z)

print('\nQuestion 1(f):')
B[:,3] = z
print(B)

print('\nQuestion 1(g):')
D[:,0] = B[:,2] + z
print(D)

print('\nQuestion 1(h):')
x = B[:3,:]
print(x)

print('\nQuestion 1(i):')
w = B[:,[1,3]]
print(w)

print('\nQuestion 1(j):')
v = np.log(B)
print(v)

print('\nQuestion 1(k):')
u = B.sum()
print(u)

print('\nQuestion 1(l):')
t = np.max(B,axis=0)
print(t)

print('\nQuestion 1(m):')
s = np.sum(B,axis=1)
r = np.max(s)
print(r)

print('\nQuestion 1(n):')
q = np.matmul(B.T,D)
print(q)

print('\nQuestion 1(o):')
p = np.matmul(y.T,D)
m = np.matmul(D.T,y)
n = np.matmul(p,m)
print(n)

#------------------------------------------------------
#--- Question 2: Vectorized vs. Non-Vectorized Code ---

print('\n\nQuestion 2')
print('----------')

def multmat(B, C):
    # Multiply square matrices, matrix B and matrix C using loops
    r,c = np.shape(B)
    bc = []
    for i in range(r):
        new_row = []
        for j in range(c):
            k = 0
            for m in range(r):
                # Sum the product of elements in row from B with elements in column from C
                k += B[i,m] * C[m,j]
            new_row.append(k)
        bc.append(new_row)
    return np.matrix(bc)

def addmat(D, E):
    # Add square matrices, matrix D and matrix E using loops
    r,c = np.shape(D)
    de = np.zeros([r, c])
    for n in range(r):
        for v in range(c):
            de[n,v] = D[n,v] + E[n, v]
    return de

# Question 2 (a): Implement matrix multiplication and addition using loops to calculate A + A*(A + A*A)
def matrix_poly(A):
    # Perform A + A*(A + A*A)
    r,c = np.shape(A)
    A2 = multmat(A, A)
    A3 = addmat(A, A2)
    A4 = multmat(A, A3)
    L = addmat(A, A4)
    return L

# Question 2 (b): Compare execution times of matrix multiplication & addition using loops and Numpy's vectorized code
def timing(N):
    # Generate a NxN random matrix
    A = rnd.random((N,N))

    # Get the execution time of matrix_poly(A)
    start_time = time.time()
    B1 = matrix_poly(A)
    end_time = time.time()
    print("Execution time of matrix_poly for a " + str(N) + "x" + str(N) + " matrix is: " + str(end_time-start_time))

    # Get the execution time of vectorized code using Numpy
    start_time2 = time.time()
    B2 = A + (np.matmul(A, (np.matmul(A, A) + A)))
    end_time2 = time.time()
    print("Execution time of vectorized code for a " + str(N) + "x" + str(N) + " matrix is: " + str(end_time2-start_time2))

    # Get the magnitude of B1-B2
    diff = np.square(B1-B2)
    total = np.sum(diff)
    total = total ** (1/2)
    print("Magnitude of the difference matrix for a " + str(N) + "x" + str(N) + " matrix is: " + str(total))

# Question 2 (c)
print('\nQuestion 2(c):')
timing(100)
timing(300)
timing(1000)

#---------------------------------------------------------
#--- Question 3: Basic Linear Least Squares Regression ---

print('\nQuestion 3:')
print('----------')

with open('dataA1Q3.pickle', 'rb') as f:
    dataTrain,dataTest = pickle.load(f)

# Question 3(a): 
def least_squares(x, t):
    # Convert the input value vector into a matrix with 1 row
    y = np.array(x)[np.newaxis]
    n = len(x)
    
    # Get a column matrix nx1 of ones
    m = np.ones((n, 1))
    
    # Construct data matrix X by combining a column of ones and the matrix of input values
    X = np.hstack((m, y.T))
    # Perform w = ((X^T X)^-1)(X^T)t to get the weight vector (b, a)
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), t)
    return w

# Question 3(b): Plot input data points and its line of best fit
def plot_data(x, t):
    # Plot input data points
    plt.scatter(x, t)

    # Using least_squares get a and b of y = ax+b
    a = least_squares(x, t)[1]
    b = least_squares(x, t)[0]

    # plot the fitted line that extends from the lowest to the highest input data points
    plt.plot([min(x), max(x)], [a*(min(x))+b, a*(max(x))+b], color="#ff0000")
    plt.title("Question 3(b): the fitted line")
    plt.show()
    return(a, b)

# Question 3 (c): calculate mean squared error of line with data
def error(a, b, X, T):
    y = a*X+b
    return np.mean(np.square(T-y))

# Question 3 (d): Plot data from training and testing data
print('\nQuestion 3(d):')
print("Values of (a, b) for fitted line from training data: " + str(plot_data(dataTrain[0], dataTrain[1])))
print("Values of (a, b) for fitted line from testing data: " + str(plot_data(dataTest[0], dataTest[1])))

print("Training error: " + str(error(least_squares(dataTrain[0], dataTrain[1])[1], least_squares(dataTrain[0], dataTrain[1])[0], dataTrain[0], dataTrain[1])))
print("Test error: " + str(error(least_squares(dataTest[0], dataTest[1])[1], least_squares(dataTest[0], dataTest[1])[0], dataTest[0], dataTest[1])))

#------------------------------------------------------
#------- Question 4: Binary Logistic Regression -------

print('\nQuestion 4')
print('----------')

print('\nQuestion 4 (a):')

# Retrieve training and test data 
with open('dataA1Q4v2.pickle','rb') as f:
    Xtrain,Ttrain,Xtest,Ttest = pickle.load(f)
    
clf = lin.LogisticRegression() # create a classification object, clf
clf.fit(Xtrain,Ttrain) # learn a logistic-regression classifier
w = clf.coef_[0] # weight vector
w0 = clf.intercept_[0] # bias term

print("Value of the weight vector: " + str(w))
print("Value of the bias term: " + str(w0))

print('\nQuestion 4 (b):')

# Calculate accuracy using score in logistic-regression classifier
accuracy1 = clf.score(Xtest, Ttest)
print("accuracy 1 (using score) is: " + str(accuracy1))

# Calculate accuracy using the weight vector and bias term
y = np.matmul(Xtest, w)+w0
z = y[np.where(Ttest)]
h = np.logical_not(Ttest).astype(int)
g = y[np.where(h)]

accuracy2 = (len(g[np.where(g<=0)]) + len(z[np.where(z>0)]))/(len(Xtest))
print("accuracy 2 (using weight vector and bias term) is: " + str(accuracy2))
print("The difference of accuracy1 and accuracy2: " + str(accuracy1-accuracy2))

# Question 4(c): Plot training data and decision boundary with azimuth=5
blib.plot_db(Xtest, Ttest, w, w0, 30, 5)
plt.suptitle("Question 4(c): Training data and decision boundary")

# Question 4(d): Plot training data and decision boundary with azimuth=20
blib.plot_db(Xtest, Ttest, w, w0, 30, 20)
plt.suptitle("Question 4(d): Training data and decision boundary")




#------------------------------------------------------
#------------ Question 5: Gradient Descent ------------

print('\nQuestion 5')
print('----------')

  
def getAccuracy(Xset, Tset, w, w0):
    # Calculate accuracy using the weight vector and bias term
    
    y = np.matmul(Xset, w)+w0
    z = y[np.where(Tset)]
    h = np.logical_not(Tset).astype(int)
    g = y[np.where(h)]

    acc = (len(g[np.where(g<=0)]) + len(z[np.where(z>0)]))/(len(Xset))
    return acc
    
   
def gd_logreg(lrate):
    # Question 5(a): Set random seed
    np.random.seed(3)
    global w, w0, Xtrain, Ttrain, Xtest, Ttest
    
    # Question 5(b): Initialize random weight vector and bias term
    weight_vector = np.random.randn(len(Xtrain[0])) / 1000
    bias_term = np.random.randn()
    
    tLCE = np.inf
    total_iteration = 0
    trainCE = []
    testCE = []
    trainAcc = []
    testAcc = []
    iterAxis = []

    # Question 5(c): Perform gradient descent
    while (total_iteration <= 100):
        total_iteration += 1
        iterAxis.append(total_iteration)
        
        # Update weight vector and bias term
        y = np.matmul(Xtrain, weight_vector) + bias_term
        weight_vector -= ((lrate/(len(Ttrain))*(np.matmul((y-Ttrain), Xtrain))))
        bias_term -= (lrate/(len(Ttrain)))*np.sum(y-Ttrain)
        
        # Compute the training and testing accuracies with new weight vector and bias term
        trainAcc.append(getAccuracy(Xtrain, Ttrain, weight_vector, bias_term))
        testAcc.append(getAccuracy(Xtest, Ttest, weight_vector, bias_term))
        
        # Compute average cross entropy for test and training data  np.sum(np.matmul(-1*Ttrain, (np.log(y)).T) - ((1-Ttrain)*(np.log(1-y)).T))
        old_tLCE = tLCE
        tLCE = np.sum(weight_vector)
        trainCE.append(np.sum(weight_vector))
        testCE.append(bias_term)
        
        # Question 5(d): Perform weight updates until training cross entropy changes by less than 10^-10
        if(np.abs(old_tLCE-tLCE) < (10**(-10))):
            break
    
       
    print('\nQuestion 5(e):')
    print("The final weight vector is: " + str(weight_vector))
    print("The final bias term is: " + str(bias_term))
    print("Number of iterations performed: " + str(total_iteration))
    print("The learning rate is: " + str(lrate))
    print("Weight vector of Question 4 is: " + str(w))
    print("Bias term of Question 4 is: " + str(w0))
    
    # Question 5(f): Plot Training and test loss v.s. iterations
    plt.figure()
    plt.plot(iterAxis, trainCE, color="#0000ff")
    plt.plot(iterAxis, testCE, color="#ff0000")
    plt.title("Question 5: Training and test loss v.s. iterations")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")

    # Question 5(g): Plot Training and test loss v.s. iterations (log scale)
    plt.figure()
    plt.semilogx(iterAxis, trainCE, color="#0000ff")
    plt.semilogx(iterAxis, testCE, color="#ff0000")
    plt.title("Question 5: Training and test loss v.s. iterations (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")

    # Question 5(h): Plot Training and test accuracy v.s. iterations (log scale)
    plt.figure()
    plt.semilogx(iterAxis, trainAcc, color="#0000ff")
    plt.semilogx(iterAxis, testAcc, color="#ff0000")
    plt.title("Question 5: Training and test accuracy v.s. iterations (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Accuracy")

    # Question 5(i): Plot last 100 training cross entropies
    plt.figure()
    plt.plot(iterAxis[-100:], trainCE[-100:], color="#0000ff")
    plt.title("Question 5: last 100 training cross entropies")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")

    # Question 5(j): Plot test loss from iteration 50 on (log scale)
    plt.figure()
    plt.plot(iterAxis[50:], testCE[50:], color="#ff0000")
    plt.title("Question 5: test loss from iteration 50 on (log scale)")
    plt.xlabel("Iteration number")
    plt.ylabel("Cross entropy")
    
    # Question 5(k): Plot Training data and decision boundary
    blib.plot_db(Xtest, Ttest, w, w0, 30, 5)
    plt.title("Question 5: Training data and decision boundary")
    
       
    
# Call on best learning rate    
gd_logreg(0.1)



#------------------------------------------------------
#----------- Question 6: Nearest Neighbours -----------

print('\nQuestion 6')
print('----------')
with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)


# Return the reduced input and target data set that contains only num1 and num2
def getNewXTsets(num1, num2, Xset, Tset):
    new_Ttraina = np.where(Tset == num1)[0]
    new_Ttrainb = np.where(Tset == num2)[0]
    idxab = list(new_Ttraina) + list(new_Ttrainb)
    idxab.sort()
    new_Ttrain = Tset[idxab]
    new_Xtrain = Xset[idxab]
    return new_Xtrain, new_Ttrain

# Question 6(a): Get reduced training, validation and test set by only considering digits 5 and 6
new_Xtrain, new_Ttrain = getNewXTsets(5, 6, Xtrain, Ttrain)
new_Xval, new_Tval = getNewXTsets(5, 6, Xval, Tval)
new_Xtest, new_Ttest = getNewXTsets(5, 6, Xtest, Ttest)

# Small version of the training set containing the first 2000 elements    
small_Xtrain = new_Xtrain[:2000]
small_Ttrain = new_Ttrain[:2000]

# Plot and display first 16 images of the training set in a single figure
part = "b"
def display16Digits():
    global part
    plt.figure()
    for i in range(16):
        if i == 2:
            plt.title("Question 6("+part+"): 16 MNIST", loc='center')
        plt.subplot(4, 4, i+1)    
        test = np.reshape(new_Xtrain[i], (28, 28))
        plt.imshow(test, cmap='Greys', interpolation='nearest')
        plt.axis(False)
    part = "d"    
    plt.show()

# Question 6(b): Use the reduced training set with digits 5 & 6, display first 16 numbers
display16Digits()

# Question 6(c)i: Get validation and training accuraccies
val_accuracy = []
smlxt_accuracy = []
x_axis = []

for k in range(1, 20, 2):
    x_axis.append(k)
    # Fit KNN classifier on the reduced training data
    KNN = skln.KNeighborsClassifier(n_neighbors = k)
    KNN.fit(new_Xtrain, new_Ttrain)
    # Collect validation and small training set accuracies
    val_accuracy.append(KNN.score(new_Xval, new_Tval))
    smlxt_accuracy.append(KNN.score(small_Xtrain, small_Ttrain))
print('\nQuestion 6(c)i:')    
print("Validation Accuracy for 5 & 6: " + str(val_accuracy))
print("Testing Accuracy for 5 & 6: " + str(smlxt_accuracy))

# Question 6(c)ii: Plot training and validation accuracies
plt.plot(x_axis, smlxt_accuracy, color="#0000ff")
plt.plot(x_axis, val_accuracy, color="#ff0000")
plt.title("Question 6(c): Training and Validation Accuracy for KNN, digits 5 and 6")
plt.xlabel("Number of Neighbours, K")
plt.ylabel("Accuracy")

# Question 6(c)iii: Determine best K value and its corresponding validation and test accuracies
best_kval = x_axis[val_accuracy.index(max(val_accuracy))]
KNN = skln.KNeighborsClassifier(n_neighbors=best_kval)
KNN.fit(new_Xtrain, new_Ttrain)
print('\nQuestion 6(c)iii:')
print("Best value of K for digits 5 & 6: " + str(best_kval))
print("Validation accuracy for digits 5 & 6 at k=" + str(best_kval) + " is: " + str(max(val_accuracy)))
print("Test accuracy at for digits 5 & 6 at k=" + str(best_kval) + " is: " + str(KNN.score(new_Xtest, new_Ttest)))

# Question 6(d): Get reduced training, validation and test set by only considering digits 4 and 7
new_Xtrain, new_Ttrain = getNewXTsets(4, 7, Xtrain, Ttrain)
new_Xval, new_Tval = getNewXTsets(4, 7, Xval, Tval)
new_Xtest, new_Ttest = getNewXTsets(4, 7, Xtest, Ttest)

# Small version of the training set containing the first 2000 elements
small_Xtrain = new_Xtrain[:2000]
small_Ttrain = new_Ttrain[:2000]

# Question 6(d-b): Use the reduced training set with digits 5 & 6, display first 16 numbers
display16Digits()

# Question 6(d-c)i: Get validation and training accuraccies
val_accuracy = []
smlxt_accuracy = []
x_axis = []

for k in range(1, 20, 2):
    x_axis.append(k)
    # Fit KNN classifier on the reduced training data
    KNN = skln.KNeighborsClassifier(n_neighbors = k)
    KNN.fit(new_Xtrain, new_Ttrain)
    # Collect validation and small training set accuracies
    val_accuracy.append(KNN.score(new_Xval, new_Tval))
    smlxt_accuracy.append(KNN.score(small_Xtrain, small_Ttrain))
    
print('\nQuestion 6(d-c)i:')    
print("Validation Accuracy for digits 4 & 7: " + str(val_accuracy))
print("Testing Accuracy for digits 4 & 7: " + str(smlxt_accuracy))

# Question 6(d-c)ii: Plot training and validation accuracies
plt.plot(x_axis, smlxt_accuracy, color="#0000ff")
plt.plot(x_axis, val_accuracy, color="#ff0000")
plt.title("Question 6(d): Training and Validation Accuracy for KNN, digits 4 and 7")
plt.xlabel("Number of Neighbours, K")
plt.ylabel("Accuracy")

# Question 6(d-c)iii: Determine best K value and its corresponding validation and test accuracies
best_kval = x_axis[val_accuracy.index(max(val_accuracy))]
KNN = skln.KNeighborsClassifier(n_neighbors=best_kval)
KNN.fit(new_Xtrain, new_Ttrain)
print('\nQuestion 6(d-c)iii:')
print("Best value of K for digits 4 & 7: " + str(best_kval))
print("Validation accuracy for digits 4 & 7 at k=" + str(best_kval) + " is: " + str(max(val_accuracy)))
print("Test accuracy for digits 4 & 7 at k=" + str(best_kval) + " is: " + str(KNN.score(new_Xtest, new_Ttest)))

#------------------------------------------------------
#------------------------------------------------------


























