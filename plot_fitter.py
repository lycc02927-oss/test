"""画 ED 模型切片和同一设定位置的单发抖动。直接运行可预览模型 3 的 RBF 图。

from plot_fitter import plot_fitter_model
fig, data = plot_fitter_model(data_model=3, fit_method="rbf", save_path="lt3.png")
fig, data = plot_fitter_model(formula_model="formula_truncated_peak", setpoint=(130, 0, 0))
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import minimize

from fitter import EDFitter, EDPFModel, build_formula_ed_model
from measurement_data import MODEL_DATA, MODEL_NAMES


def _values(model, xyz):
    """数据拟合器可批量预测，公式模型沿用自身的单点 predict。"""
    if isinstance(model, EDFitter):
        return np.concatenate([model.model.predict(xyz[i:i + 512])
                               for i in range(0, len(xyz), 512)])
    return np.array([model.predict(p) for p in xyz])


def _find_peak(model, bounds, anchors):
    """在三维边界内找静态峰：原始点/公式中心、规则网格以及高值区的局部精修。"""
    lo, hi = bounds.T
    axis = [np.linspace(a, b, 19) for a, b in bounds]
    grid = np.stack(np.meshgrid(*axis, indexing="ij"), axis=-1).reshape(-1, 3)
    xyz = np.vstack((np.clip(anchors, lo, hi), grid))
    ed = _values(model, xyz)
    best = int(np.argmax(ed))
    peak, value = xyz[best].copy(), float(ed[best])
    scale = max(float(np.ptp(ed)), 1e-12)

    # 在单位立方体内精修；起点稍微分散，避免多峰情况下总在同一小片区域启动。
    starts = []
    for index in np.argsort(-ed, kind="stable"):
        u = (xyz[index] - lo) / (hi - lo)
        if all(np.linalg.norm(u - old) > 0.12 for old in starts):
            starts.append(u)
        if len(starts) == 8:
            break
    for u in starts:
        fit = minimize(lambda t: -model.predict(lo + t * (hi - lo)) / scale, u,
                       method="L-BFGS-B", bounds=[(0, 1)] * 3)
        point = lo + np.clip(fit.x, 0, 1) * (hi - lo)
        prediction = model.predict(point)
        if prediction > value:
            peak, value = point, float(prediction)
    return peak, value


class _ShotProbe:
    """记录 run_pipeline 真正传给空间模型的位置和 ED，同时保留模型自身的限幅。"""
    def __init__(self, model):
        self.model = model
        self.ed_limits = getattr(model, "ed_limits", None)

    def predict(self, xyz):
        self.xyz = np.array(xyz, copy=True)
        self.ed = self.model.predict(xyz)
        return self.ed


def _xyz_text(xyz):
    return "(" + ", ".join(f"{v:.2f}" for v in xyz) + ")"


def plot_fitter_model(data_model=3, fit_method="rbf", *, formula_model=None, peaks=None,
                      fit_parameters=None, bounds=((-300, 300),) * 3, setpoint=None,
                      shots=100, seed=42, position_half_width=10.0,
                      ed_time_parameters=None, pf_time_parameters=None,
                      save_path=None, show=True):
    """返回 (Figure, data)。bounds 按 XYZ 排列，单位 µm；默认在静态峰位重复量测。

    数据模式从 MODEL_DATA 读取，fit_parameters 与 EDFitter 一致。传 formula_model
    时使用公式模型及其 peaks。data 包含每一发的真实取样坐标、位置抖动后的空间 ED、
    最终单发 ED、PF，以及峰位、峰值和观测最高发次（从 1 开始）。
    """
    if shots < 1:
        raise ValueError("shots 至少为 1")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds 需要给出 X、Y、Z 的三个递增区间")
    if formula_model is None:
        model = EDFitter(data_model, fit_method, fit_parameters)
        anchors = MODEL_DATA[data_model][:, :3]
        title = MODEL_NAMES[data_model]
        method = f"拟合方法  {fit_method}"
    else:
        model = build_formula_ed_model(formula_model, peaks=peaks)
        anchors = model.centers
        title = formula_model
        method = "生成方法  解析公式"

    peak, peak_ed = _find_peak(model, bounds, anchors)
    center = peak.copy() if setpoint is None else np.asarray(setpoint, dtype=float)
    center_ed = model.predict(center)
    probe = _ShotProbe(model)
    sensor = EDPFModel(probe, seed=seed, position_half_width=position_half_width,
                       ed_time_parameters=ed_time_parameters, pf_time_parameters=pf_time_parameters)
    positions, spatial_ed, measured_ed, pf = [], [], [], []
    for _ in range(shots):
        ed, power = sensor.measure(center)
        positions.append(probe.xyz)
        spatial_ed.append(probe.ed)
        measured_ed.append(ed)
        pf.append(power)
    positions = np.asarray(positions)
    spatial_ed, measured_ed = np.asarray(spatial_ed), np.asarray(measured_ed)
    best = int(np.argmax(measured_ed))
    data = dict(peak_xyz=peak, peak_ed=peak_ed, setpoint=center, setpoint_ed=center_ed,
                sample_xyz=positions, spatial_ed=spatial_ed, ed=measured_ed, pf=np.array(pf),
                best_shot=best + 1, best_shot_xyz=positions[best], best_shot_ed=measured_ed[best])

    # 三张切片均穿过峰位，并共用一种颜色刻度。将峰坐标放进网格，保留窄峰。
    planes = []
    for i, j, fixed in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        a = np.unique(np.append(np.linspace(*bounds[i], 81), peak[i]))
        b = np.unique(np.append(np.linspace(*bounds[j], 81), peak[j]))
        xx, yy = np.meshgrid(a, b)
        xyz = np.tile(peak, (xx.size, 1))
        xyz[:, i], xyz[:, j] = xx.ravel(), yy.ravel()
        z = _values(model, xyz).reshape(xx.shape)
        planes.append((i, j, fixed, a, b, z))
    norm = Normalize(min(p[-1].min() for p in planes), max(p[-1].max() for p in planes))
    style = {"font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                                 "SimHei", "PingFang SC", "DejaVu Sans"], "axes.unicode_minus": False}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13.4, 8.0), layout="constrained")
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .046], height_ratios=[1.3, .86])
        fig.suptitle(f"{title}   |   ED 空间分布与单发抖动", fontsize=19, fontweight="bold")
        for col, (i, j, fixed, a, b, z) in enumerate(planes):
            ax = fig.add_subplot(gs[0, col])
            im = ax.pcolormesh(a, b, z, shading="auto", cmap="viridis", norm=norm)
            ax.scatter(peak[i], peak[j], marker="*", s=155, c="#ffcd55", edgecolors="#24292d")
            ax.set(xlim=bounds[i], ylim=bounds[j], aspect="equal",
                   xlabel=f"{'XYZ'[i]} (µm)", ylabel=f"{'XYZ'[j]} (µm)",
                   title=f"{'XYZ'[i]}{'XYZ'[j]} 切片  ·  {'XYZ'[fixed]} = {peak[fixed]:.2f} µm")
        fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]), label="静态 ED")
        ax = fig.add_subplot(gs[1, :2])
        n = np.arange(1, shots + 1)
        ax.axhline(center_ed, ls="--", lw=1.2, color="#56626d", label="设定点静态 ED")
        ax.vlines(n, spatial_ed, measured_ed, color="#ac65a0", lw=1)
        ax.scatter(n, spatial_ed, facecolors="none", edgecolors="#28758a", s=28, label="仅位置抖动")
        ax.scatter(n, measured_ed, color="#263c78", s=9, label="最终单发 ED")
        ax.scatter(best + 1, measured_ed[best], marker="D", c="#f3824b", edgecolors="#422c25",
                   s=48, zorder=4, label="本组最高单发")
        ax.set(title="固定设定点重复量测", xlabel="发次", ylabel="ED")
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2, fontsize=9)
        info = fig.add_subplot(gs[1, 2:])
        info.axis("off")
        block = (f"{method}  ·  {shots} 发  ·  seed {seed}\n"
                 f"★  静态数值峰   ED = {peak_ed:.7g}\n"
                 f"     XYZ  {_xyz_text(peak)} µm\n"
                 f"◆  最高单发 #{best + 1}   ED = {measured_ed[best]:.7g}\n"
                 f"     取样 XYZ  {_xyz_text(positions[best])} µm\n"
                 f"设定 XYZ  {_xyz_text(center)} µm\n"
                 f"各轴位置抖动 ±{position_half_width:g} µm    单发标准差 {np.std(measured_ed):.3g}")
        info.text(.01, .98, block, va="top", linespacing=1.65, fontsize=10.5, color="#23374a")
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
    return fig, data


if __name__ == "__main__":
    plot_fitter_model(save_path="fitter_model.png")
