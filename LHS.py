"""为 GPR.py、GPR_SIMPLE.py 生成三维初始采样点。

逻辑：每轴分层随机采样 → Beta 变换使点偏向中心 → 映射到三轴范围。
n 控制点数，x_range/y_range/z_range 控制边界；alpha 控制 XY 的中心集中度，
beta_z 控制 Z 的集中度。值越大越集中，1 对应均匀分布。
get_initial_points() 固定用 alpha=2、beta_z=3，并将坐标保留两位小数；
plot=True 可查看分布。随机数与 GPR 共用，种子由优化入口设置。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


def lhs_centered_zconcentrated(n, alpha=2, beta_z=3, *, rng=None):
    """每轴各取 n 个分层随机数，再打乱顺序并按 Beta 分布向中心集中。"""
    rng = np.random if rng is None else rng
    intervals = np.arange(n) / n
    u = intervals + rng.random(n) / n
    v = intervals + rng.random(n) / n
    w = intervals + rng.random(n) / n
    rng.shuffle(u)
    rng.shuffle(v)
    rng.shuffle(w)
    return beta.ppf(u, alpha, alpha), beta.ppf(v, alpha, alpha), beta.ppf(w, beta_z, beta_z)


def generate_lhs_points(n=30, x_range=(0, 10), y_range=(0, 10), z_range=(0, 10),
                        alpha=2, beta_z=3, *, rng=None):
    """将 [0,1] 内的三轴采样数映射到给定范围，返回 N×3 坐标。"""
    u, v, w = lhs_centered_zconcentrated(n, alpha, beta_z, rng=rng)
    x = x_range[0] + u * (x_range[1] - x_range[0])
    y = y_range[0] + v * (y_range[1] - y_range[0])
    z = z_range[0] + w * (z_range[1] - z_range[0])
    return np.column_stack((x, y, z))


def plot_3d_distribution(points):
    """显示三维散点及 XY、XZ、YZ 投影，便于查看初始点分布。"""
    fig = plt.figure(figsize=(12, 9))
    x, y, z = points.T
    ax = fig.add_subplot(221, projection="3d")
    scatter = ax.scatter(x, y, z, c=z, cmap="viridis")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", title="Initial LHS points")
    fig.colorbar(scatter, ax=ax, label="Z")
    for index, (left, right, labels) in enumerate(
            [(x, y, "XY"), (x, z, "XZ"), (y, z, "YZ")], start=2):
        ax = fig.add_subplot(2, 2, index)
        ax.scatter(left, right, c=z, cmap="viridis")
        ax.set(xlabel=labels[0], ylabel=labels[1], title=labels + " projection")
        ax.grid(True)
    fig.tight_layout()
    plt.show()


def get_initial_points(n=30, x_range=(-150, 150), y_range=(-150, 150),
                       z_range=(-180, 180), plot=False, *, rng=None):
    """按点数和三轴范围生成初始坐标；plot 决定是否显示图片。"""
    points = generate_lhs_points(n, x_range, y_range, z_range, alpha=2, beta_z=3, rng=rng)
    points = np.around(points, decimals=2)
    if plot:
        plot_3d_distribution(points)
    return points


if __name__ == "__main__":
    get_initial_points(plot=True)
