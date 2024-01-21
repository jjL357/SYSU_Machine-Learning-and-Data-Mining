import numpy as np
import matplotlib.pyplot as plt
# 加载训练集和测试集
train_data=np.loadtxt('D:\\d_code\\MLDM\\Lab3\\assignment 3\\dataForTrainingLogistic.txt')
test_data=np.loadtxt('D:\\d_code\\MLDM\\Lab3\\assignment 3\\dataForTestingLogistic.txt')
# 提取特征和标签
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# 计算预测结果
def predict(X, w):
    X = np.column_stack((np.ones(X.shape[0]), X))  # 加偏置项
    scores = np.dot(X, w)
    predicted_labels = np.round(sigmoid(scores))
    return predicted_labels
# 初始化参数
p = X_train.shape[1]
w = np.zeros(p + 1)  # 加截距项
# 超参数
learning_rate = 0.00015
iterations = 1000
# 存储目标函数
objective_values = []
# 训练逻辑回归分类器
for iteration in range(iterations):
    objective = 0
    for i in range(len(X_train)):
        xi = np.insert(X_train[i], 0, 1)  # 增加偏置项
        zi = np.dot(w, xi)
        predicted = sigmoid(zi)
        gradient = xi * (predicted - y_train[i])
        w -= learning_rate * gradient
        objective += y_train[i] * np.log(predicted) + (1 - y_train[i]) * np.log(1 - predicted)
    objective_values.append(objective)
# 统计测试集错分数目
y_pred = predict(X_test, w)
misclassified = np.sum(y_pred != y_test)
print(f"Number of misclassified examples in the testing dataset: {misclassified}")
accuracy = (p-misclassified)/p*100
print(f"Accuracy: {accuracy}")
# 可视化结果
plt.plot(range(iterations), objective_values)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Objective Function Value vs. Iteration')
plt.show()


# 记录随着训练子集增大误差的变化
training_sizes = list(range(10, 401, 10))
train_errors = []
test_errors = []
for size in training_sizes:
    # 随机选择一组数据
    random_indices = np.random.choice(len(X_train), size, replace=False)
    X_subset = X_train[random_indices]
    y_subset = y_train[random_indices]
    # 在随机选择数据上训练
    w_subset = np.zeros(p + 1)
    for iteration in range(iterations):
        objective = 0
        for i in range(size):
            xi = np.insert(X_subset[i], 0, 1)  # 加偏置项
            zi = np.dot(w_subset, xi)
            predicted = sigmoid(zi)
            gradient = xi * (predicted - y_subset[i])
            w_subset -= learning_rate * gradient
            objective += y_subset[i] * np.log(predicted) + (1 - y_subset[i]) * np.log(1 - predicted)
    # 预测测试集
    y_pred_subset = predict(X_test, w_subset)
    # 统计训练集和测试集误差
    train_errors.append(np.sum(predict(X_subset, w_subset) != y_subset) / size)
    test_errors.append(np.sum(y_pred_subset != y_test) / len(y_test))
# 可视化
plt.plot(training_sizes, train_errors, label='Training Error', color='blue')
plt.plot(training_sizes, test_errors, label='Test Error', color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Error')
plt.title('Training and Test Error vs. Training Set Size')
plt.legend()
plt.show()
