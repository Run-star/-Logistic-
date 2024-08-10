import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()



m_train = train_set_y.shape[1] #训统计数量
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T #降低维度，把图片信息录入
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

def sigmoid(z):

    s = 1 / (1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):

    w = np.zeros(shape = (dim,1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return (w , b)

def propagate(w, b, X, Y):

    m = X.shape[1]

    #正向传播
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) #计算成本

    #反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    #保存dw db
    grads = {
                "dw": dw,
                "db": db
             }
    return (grads , cost)

def optimize(w , b , X , Y , num_iterations , learning_rate , print_cost = False):

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        #记录成本
        if i % 100 == 0:
            costs.append(cost)
        #打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i,cost))

    params  = {
                "w" : w,
                "b" : b }
    grads = {
            "dw": dw,
            "db": db }
    return (params , grads , costs)


def predict(w , b , X ):

    m  = X.shape[1] #图片的数量
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    #计预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T , X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0

    return Y_prediction


def model(X_train , Y_train , X_test , Y_test , num_iterations = 2000 , learning_rate = 0.5 , print_cost = False):

    w , b = initialize_with_zeros(X_train.shape[0])

    parameters , grads , costs = optimize(w , b , X_train , Y_train, num_iterations , learning_rate , print_cost)

    #检索参数w和b
    w , b = parameters["w"] , parameters["b"]

    #预测测试/训练集的例子
    Y_prediction_test = predict(w , b, X_test)
    Y_prediction_train = predict(w , b, X_train)

    #打印训练后的准确性
    print("训练集准确性："  , format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100) ,"%")
    print("测试集准确性："  , format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100) ,"%")

    d = {
            "costs" : costs,
            "Y_prediction_test" : Y_prediction_test,
            "Y_prediciton_train" : Y_prediction_train,
            "w" : w,
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations" : num_iterations }
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
# 学习率设置比对
learning_rates = [0.005]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)


for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
