#coding=gbk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import warnings
warnings.filterwarnings('ignore')

# 1. 生成模拟数据
np.random.seed(42)
n_samples, n_features = 200, 1000  # 设定样本数和特征数
# 生成稀疏的真实系数
beta_true = np.zeros(n_features)
beta_true[:50] = np.random.randn(50) * 2  # 只有前50个特征非零
X = np.random.randn(n_samples, n_features)
y = X @ beta_true + np.random.randn(n_samples) * 0.5  # 加入噪声
lambda_ = 0.1  # 正则化系数

# 2. 使用高精度坐标下降法获取“参考最优解”
# 设置非常小的容忍度和最大迭代次数以获得高精度解
lasso_high_precision = Lasso(alpha=lambda_, tol=1e-12, max_iter=1000, fit_intercept=False)
lasso_high_precision.fit(X, y)
beta_opt = lasso_high_precision.coef_

# 3. 定义各算法的目标函数值计算
def objective(beta, X, y, lambda_):
    n = len(y)
    return (1/(2*n)) * np.sum((y - X @ beta)**2) + lambda_ * np.sum(np.abs(beta))

# 4. 实现各算法
def coordinate_descent_lasso(X, y, lambda_, max_iter=100, tol=1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    obj_history = []
    for k in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            X_j = X[:, j]
            y_pred = X @ beta
            r = y - y_pred + X_j * beta[j]  # 计算残差时暂时排除第j个特征的影响
            # 软阈值更新规则
            rho_j = X_j @ r / n
            z_j = np.sum(X_j**2) / n
            beta[j] = np.sign(rho_j) * max(abs(rho_j) - lambda_, 0) / (z_j + 1e-8)
        obj_history.append(objective(beta, X, y, lambda_))
        if np.linalg.norm(beta - beta_old) < tol:
            break
    return beta, obj_history, k+1

def gradient_descent_lasso_smooth(X, y, lambda_, max_iter=1000, epsilon=1e-4):
    """使用Huber函数平滑L1范数的梯度下降法[citation:4]"""
    n, p = X.shape
    beta = np.zeros(p)
    obj_history = []
    # 计算Lipschitz常数
    L = np.linalg.eigvalsh(X.T @ X).max() / n
    step_size = 1 / (L + lambda_/epsilon)  # 根据平滑参数调整步长
    for k in range(max_iter):
        # Huber平滑后的梯度
        grad = (-X.T @ (y - X @ beta)) / n
        # 对L1范数的平滑近似 (Huber函数在epsilon邻域内的导数)
        grad_l1 = lambda_ * np.where(np.abs(beta) > epsilon, np.sign(beta), beta/epsilon)
        beta = beta - step_size * (grad + grad_l1)
        obj_history.append(objective(beta, X, y, lambda_))
    return beta, obj_history, max_iter

def admm_lasso(X, y, lambda_, rho=1.0, max_iter=1000):
    """ADMM算法实现[citation:6][citation:7]"""
    n, p = X.shape
    beta = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    obj_history = []
    XTX = X.T @ X
    I = np.eye(p)
    # 预先计算矩阵分解以提高效率
    M = np.linalg.inv(XTX / n + rho * I)
    for k in range(max_iter):
        # x-update
        beta = M @ (X.T @ y / n + rho * (z - u))
        # z-update (软阈值操作)
        z_old = z
        z = np.sign(beta + u) * np.maximum(np.abs(beta + u) - lambda_/rho, 0)
        # u-update
        u = u + beta - z
        obj_history.append(objective(beta, X, y, lambda_))
    return beta, obj_history, max_iter

# 5. 运行所有算法并记录结果
algorithms = {
    'Coordinate Descent': coordinate_descent_lasso,
    'Gradient Descent (Smooth)': lambda X, y, lambda_: gradient_descent_lasso_smooth(X, y, lambda_, max_iter=1000),
    'ADMM (rho=0.1)': lambda X, y, lambda_: admm_lasso(X, y, lambda_, rho=0.1, max_iter=1000),
    'ADMM (rho=1.0)': lambda X, y, lambda_: admm_lasso(X, y, lambda_, rho=1.0, max_iter=1000),
    'ADMM (rho=10.0)': lambda X, y, lambda_: admm_lasso(X, y, lambda_, rho=10.0, max_iter=1000)
}

results = {}
for name, algo in algorithms.items():
    beta_est, obj_history, iters = algo(X, y, lambda_)
    # 计算距离最优点的距离：目标函数值差 (f(x^k) - f^*)
    f_opt = objective(beta_opt, X, y, lambda_)
    dist_history = [f - f_opt for f in obj_history]
    results[name] = {
        'beta': beta_est,
        'dist_history': dist_history,
        'iterations': iters
    }
    print(f"{name}: 最终目标值 = {obj_history[-1]:.6f}, 距离最优点 = {dist_history[-1]:.6e}")

# 6. 绘制对比图
plt.figure(figsize=(12, 8))
colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))

for idx, (name, result) in enumerate(results.items()):
    dist_hist = result['dist_history']
    iterations = range(1, len(dist_hist) + 1)
    plt.semilogy(iterations, dist_hist, color=colors[idx], linewidth=2.5, label=name)

plt.xlabel('Iteration (k)', fontsize=14, fontweight='bold')
plt.ylabel('Distance to Optimum: f(x^k) - f*', fontsize=14, fontweight='bold')
plt.title('Comparison of LASSO Optimization Algorithms', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend(fontsize=12, framealpha=0.9)
plt.xlim(1, max(len(r['dist_history']) for r in results.values()))
plt.ylim(1e-10, 1e2)  # 调整y轴范围以更好显示差异
plt.tight_layout()
plt.show()

