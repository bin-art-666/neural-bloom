"""Neural Bloom · 神经花开
示例：从零实现一个 NumPy XOR 神经网络。

运行：
    python -m pip install numpy
    python examples/01-xor_numpy.py
"""

import numpy as np


def sigmoid(z):
    """将数值映射到 (0, 1)，作为二分类概率输出。"""
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob):
    """计算平均二元交叉熵，避免直接计算 log(0)。"""
    eps = 1e-12
    return -np.mean(
        y_true * np.log(y_prob + eps)
        + (1.0 - y_true) * np.log(1.0 - y_prob + eps)
    )


def main():
    # 1. XOR 训练数据：两个输入，对应一个正确输出。
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    Y = np.array([[0], [1], [1], [0]], dtype=float)

    # 2. 初始化参数。固定随机种子，便于复现教学结果。
    rng = np.random.default_rng(42)
    W1 = rng.normal(0.0, 0.5, size=(2, 4))
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0.0, 0.5, size=(4, 1))
    b2 = np.zeros((1, 1))

    learning_rate = 0.5
    epochs = 6001

    # 3. 训练：前向传播 -> 损失 -> 反向传播 -> 参数更新。
    for epoch in range(epochs):
        # 前向传播
        H = np.tanh(X @ W1 + b1)
        P = sigmoid(H @ W2 + b2)

        loss = binary_cross_entropy(Y, P)

        # 反向传播：sigmoid + 二元交叉熵的梯度简化形式
        dZ2 = (P - Y) / len(X)
        dW2 = H.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # tanh 的导数：1 - tanh(z)^2
        dZ1 = (dZ2 @ W2.T) * (1.0 - H**2)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # 梯度下降
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 2000 == 0:
            print(f"Epoch {epoch:4d} | loss = {loss:.6f}")

    # 4. 使用训练后的参数再次前向传播，查看最终预测。
    H = np.tanh(X @ W1 + b1)
    P = sigmoid(H @ W2 + b2)
    predicted = (P >= 0.5).astype(int)

    print("\nFinal predictions:")
    for x, prob, label in zip(X.astype(int), P[:, 0], predicted[:, 0]):
        print(f"{x.tolist()} -> probability={prob:.4f} -> class={label}")

    accuracy = np.mean(predicted == Y)
    print(f"\nTraining-set accuracy: {accuracy:.0%}")
    print("注意：四条 XOR 数据既用于训练也用于此处评价，仅用于演示基本机制。")


if __name__ == "__main__":
    main()
