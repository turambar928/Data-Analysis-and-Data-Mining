import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X = iris.data[:100, [0, 2]]  # 只取前100个样本（Setosa 和 Versicolor），并取两个特征
y = iris.target[:100]
y = np.where(y == 0, -1, 1)  # 将标签转化为-1 和 1


# 感知器训练算法
class Perceptron:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])  # 权重向量初始化
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update  # 偏置项更新
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# 训练模型
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# 可视化训练错误
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.title('Perceptron - Misclassifications over Epochs')
plt.show()
