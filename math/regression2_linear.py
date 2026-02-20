import numpy as np
import matplotlib.pyplot as plt

# 读入训练数据
train = np.loadtxt("click.csv", delimiter=",", skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]


theta = np.random.rand(3)

# 标准化
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


# 目标函数
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


train_z = standardize(train_x)


# 创建训练数据的矩阵
def to_matrix(x: np.ndarray):
    return np.vstack([np.ones(x.shape[0]), x, x**2]).T


X = to_matrix(train_z)


# 预测函数
def f(x: np.ndarray):
    return np.dot(x, theta)


# 均方误差
def MSE(x: np.ndarray, y: np.ndarray):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)


# 均方误差的历史记录
errors = []


# 学习率
ETA = 1e-3


# 误差的差值
diff = 1

# 更新次数
count = 0

# 重复学习
errors.append(MSE(X, train_y))
while diff > 1e-2:
    # 为了调整训练数据的顺序,准备随时的序列
    p = np.random.permutation(X.shape[0])
    print(p)
    # 随机取出训练数据,使用随机梯度下降法更新参数
    for x, y in zip(X[p, :], train_y[p]):
        theta = theta - ETA * (f(x) - y) * x

    # 更新参数
    # theta = theta - ETA * np.dot(f(X) - train_y, X)

    # 计算与上一次误差的差值
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]

    # 输出日志
    count += 1
    log = "第{}次: theta = {}, 差值 = {:.4f}"
    print(log.format(count, theta, diff))

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, "o")
plt.plot(x, f(to_matrix(x)))
plt.title("拟合训练数据曲线")
plt.show()


# 绘制误差变化图
x = np.arange(len(errors))
plt.plot(x, errors)
plt.title("误差变化图")
plt.show()
