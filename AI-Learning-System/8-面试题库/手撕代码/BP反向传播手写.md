# 手撕代码：BP反向传播

## 面试要求

- 难度：★★★★☆
- 能推导MLP反向传播
- 能写出关键代码

---

## 两层MLP完整BP推导

```python
import numpy as np

class TwoLayerMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.lr = lr
        # 初始化
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        # Layer 1
        self.z1 = X @ self.W1 + self.b1          # (N, H)
        self.a1 = self.sigmoid(self.z1)           # (N, H)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2    # (N, C)
        self.a2 = self.softmax(self.z2)           # (N, C)

        return self.a2

    def backward(self, X, y, y_pred):
        N = X.shape[0]

        # Layer 2 梯度
        dz2 = y_pred - y                          # (N, C), softmax+CE简洁梯度
        dW2 = self.a1.T @ dz2 / N                 # (H, C)
        db2 = dz2.sum(axis=0) / N                # (C,)

        # Layer 1 梯度
        da1 = dz2 @ self.W2.T                     # (N, H)
        dz1 = da1 * self.a1 * (1 - self.a1)      # (N, H), sigmoid导数
        dW1 = X.T @ dz1 / N                       # (D, H)
        db1 = dz1.sum(axis=0) / N                # (H,)

        # 更新
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = -np.log(y_pred[range(len(y)), y]).mean()
        self.backward(X, np.eye(y_pred.shape[1])[y], y_pred)
        return loss
```

---

## 关键公式速记

```
1. Softmax + CE梯度：
   ∂L/∂z = softmax(z) - y_onehot

2. Sigmoid导数：
   σ'(x) = σ(x) * (1 - σ(x))

3. ReLU导数：
   ReLU'(x) = 1 if x > 0 else 0

4. Linear层(B=Wx+b)反向：
   ∂L/∂W = x^T @ ∂L/∂output
   ∂L/∂b = sum(∂L/∂output)
   ∂L/∂x = ∂L/∂output @ W^T
```

---

## 面试口述要点

```
1. Forward Pass：保存中间结果
   - 每一层的输入、输出、激活值

2. Backward Pass：链式法则从后向前
   - 输出层：∂L/∂z = softmax(z) - y
   - 隐藏层：上游梯度 @ W^T * 激活函数导数

3. 参数更新：θ = θ - lr * ∂L/∂θ
```
