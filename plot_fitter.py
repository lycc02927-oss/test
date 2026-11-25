"""画 ED 模型的三维分布、过峰切片及固定设定位置的单发抖动。

from plot_fitter import plot_fitter_model
fig, data = plot_fitter_model(data_model=3, fit_method="rbf", save_path="lt3.png")
fig, data = plot_fitter_model(formula_model="formula_truncated_peak", setpoint=(130, 0, 0))
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from scipy.optimize import minimize

from fitter import EDFitter, EDPFModel, build_formula_ed_model
from measurement_data import MODEL_DATA, MODEL_NAMES


def _values(model, xyz):
    """数据拟合器可批量预测，公式模型沿用自身的单点 predict。"""
    if isinstance(model, EDFitter):
        return np.concatenate([model.model.predict(xyz[i:i + 512])
                               for i in range(0, len(xyz), 512)])
    return np.array([model.predict(p) for p in xyz])


def _refine_extreme(model, bounds, xyz, ed, sign):
    """以分散的高值或低值点为起点做有界局部精修。sign=-1 找最大值，+1 找最小值。"""
    lo, hi = bounds.T
    order = np.argsort(sign * ed, kind="stable")
    location = xyz[order[0]].copy()
    value = float(ed[order[0]])
    scale = max(float(np.ptp(ed)), 1e-12)

    # 在单位立方体内精修，分散起点以覆盖多个高值区或低值区。
    starts = []
    for index in order:
        u = (xyz[index] - lo) / (hi - lo)
        if all(np.linalg.norm(u - old) > 0.12 for old in starts):
            starts.append(u)
        if len(starts) == 8:
            break
    for u in starts:
        fit = minimize(lambda t: sign * model.predict(lo + t * (hi - lo)) / scale, u,
                       method="L-BFGS-B", bounds=[(0, 1)] * 3)
        point = lo + np.clip(fit.x, 0, 1) * (hi - lo)
        prediction = model.predict(point)
        if sign * prediction < sign * value:
            location, value = point, float(prediction)
    return location, value


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


def _draw_heat_slices(ax, planes, peak, cmap, norm):
    """把同一组二维 ED 数据放回三维空间，每张热力面穿过静态峰位。"""
    for i, j, fixed, a, b, ed in planes:
        xyz = [None] * 3
        xyz[i], xyz[j] = np.meshgrid(a, b)
        xyz[fixed] = np.full_like(xyz[i], peak[fixed])
        # 每个网格面的颜色取四角 ED 的均值。关闭光照，避免朝向改变颜色含义。
        face_ed = (ed[:-1, :-1] + ed[1:, :-1] + ed[:-1, 1:] + ed[1:, 1:]) / 4
        colors = cmap(norm(face_ed))

        # 沿交线拆面，让 Matplotlib 旋转视角时能按前后顺序绘制各块。
        col = np.searchsorted(a, peak[i])
        row = np.searchsorted(b, peak[j])
        for y0, y1 in ((0, row), (row, len(b) - 1)):
            for x0, x1 in ((0, col), (col, len(a) - 1)):
                if x1 > x0 and y1 > y0:
                    patch = tuple(coord[y0:y1 + 1, x0:x1 + 1] for coord in xyz)
                    ax.plot_surface(*patch, facecolors=colors[y0:y1, x0:x1],
                                    rstride=1, cstride=1, shade=False,
                                    linewidth=0, antialiased=False)


def plot_fitter_model(data_model=3, fit_method="rbf", *, formula_model=None, peaks=None,
                      fit_parameters=None, bounds=((-300, 300),) * 3, setpoint=None,
                      shots=100, seed=42, position_half_width=10.0,
                      ed_time_parameters=None, pf_time_parameters=None,
                      save_path=None, show=True):
    """返回 (Figure, data)。bounds 按 XYZ 排列，单位 µm；默认在静态峰位重复量测。

    数据模式从 MODEL_DATA 读取，fit_parameters 与 EDFitter 一致。传 formula_model
    时使用公式模型及其 peaks。data 包含每一发的真实取样坐标、位置抖动后的空间 ED、
    最终单发 ED、PF，以及静态数值极值 peak_ed/peak_xyz 和 min_ed/min_xyz、
    观测最高发次（从 1 开始）。同值有多处时标记其中一个位置。
    三维热力图显示过峰位的 XY、XZ、YZ 三张切面，颜色与上方切片共用静态 ED 色标。
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

    # 两个极值共用同一次三维网格预测；包含边界、八个角及模型原始点。
    axes = [np.linspace(a, b, 19) for a, b in bounds]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    candidates = np.vstack((np.clip(anchors, *bounds.T), grid.reshape(-1, 3)))
    predictions = _values(model, candidates)
    peak, peak_ed = _refine_extreme(model, bounds, candidates, predictions, -1)
    low, low_ed = _refine_extreme(model, bounds, candidates, predictions, 1)
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
    data = dict(peak_xyz=peak, peak_ed=peak_ed, min_xyz=low, min_ed=low_ed,
                setpoint=center, setpoint_ed=center_ed,
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
    norm = Normalize(min(low_ed, *(p[-1].min() for p in planes)),
                     max(peak_ed, *(p[-1].max() for p in planes)))
    # 将实际字体名存进图中，调用方旋转三维图或再次保存时仍能显示中文。
    font = FontProperties(family=["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                                  "SimHei", "PingFang SC", "DejaVu Sans"]).get_name()
    style = {"font.family": font, "axes.unicode_minus": False}
    with plt.rc_context(style):
        cmap = plt.get_cmap("plasma")
        fig = plt.figure(figsize=(13.4, 9.8), layout="constrained")
        rows = fig.add_gridspec(2, 1, height_ratios=[.98, 1.34])
        top = rows[0].subgridspec(1, 4, width_ratios=[1, 1, 1, .046])
        bottom = rows[1].subgridspec(2, 2, width_ratios=[.92, 1.18], height_ratios=[3.1, .85])
        fig.suptitle(f"{title}   |   ED 空间分布与单发抖动", fontsize=19, fontweight="bold")
        for col, (i, j, fixed, a, b, z) in enumerate(planes):
            ax = fig.add_subplot(top[0, col])
            im = ax.pcolormesh(a, b, z, shading="auto", cmap=cmap, norm=norm)
            ax.scatter(peak[i], peak[j], marker="*", s=155, c="#ffcd55", edgecolors="#24292d")
            ax.set(xlim=bounds[i], ylim=bounds[j], aspect="equal",
                   xlabel=f"{'XYZ'[i]} (µm)", ylabel=f"{'XYZ'[j]} (µm)",
                   title=f"{'XYZ'[i]}{'XYZ'[j]} 切片  ·  {'XYZ'[fixed]} = {peak[fixed]:.2f} µm")
        fig.colorbar(im, cax=fig.add_subplot(top[0, 3]), label="静态 ED")

        ax3d = fig.add_subplot(bottom[0, 0], projection="3d")
        _draw_heat_slices(ax3d, planes, peak, cmap, norm)
        # 用顶层标记标出整个搜索范围的极值，即使极小点在切面后面也能看到。
        ax3d.plot(*low, marker="v", ms=10, color="#66ccff", mec="#13283f", zorder=100)
        ax3d.plot(*peak, marker="*", ms=15, color="#ffcd55", mec="#24292d", zorder=101)
        ax3d.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2],
                 xlabel="X (µm)", ylabel="Y (µm)", zlabel="Z (µm)", title="三维 ED 热力图 · 过峰正交切片")
        ax3d.set_box_aspect(np.ptp(bounds, axis=1), zoom=.9)
        for axis, (start, stop) in zip((ax3d.xaxis, ax3d.yaxis, ax3d.zaxis), bounds):
            axis.set_ticks(np.linspace(start, stop, 3))
            axis.set_tick_params(labelsize=8)
        ax3d.view_init(elev=23, azim=-53)
        extrema = fig.add_subplot(bottom[1, 0])
        extrema.axis("off")
        for y, marker, color, label, val, point in [
            (.74, "*", "#ffcd55", "最大", peak_ed, peak),
            (.32, "v", "#66ccff", "最小", low_ed, low),
        ]:
            extrema.scatter(.04, y, marker=marker, s=105, c=color, edgecolors="#24292d",
                            transform=extrema.transAxes)
            extrema.text(.09, y, f"{label} ED  {val:.7g}     XYZ  {_xyz_text(point)} µm",
                         va="center", transform=extrema.transAxes)

        ax = fig.add_subplot(bottom[0, 1])
        n = np.arange(1, shots + 1)
        ax.axhline(center_ed, ls="--", lw=1.2, color="#56626d", label="设定点静态 ED")
        ax.vlines(n, spatial_ed, measured_ed, color="#ac65a0", lw=1)
        ax.scatter(n, spatial_ed, facecolors="none", edgecolors="#28758a", s=28, label="仅位置抖动")
        ax.scatter(n, measured_ed, color="#263c78", s=9, label="最终单发 ED")
        ax.scatter(best + 1, measured_ed[best], marker="D", c="#f3824b", edgecolors="#422c25",
                   s=48, zorder=4, label="本组最高单发")
        ax.set(title="固定设定点重复量测", xlabel="发次", ylabel="ED")
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="best", ncol=2, fontsize=8.5)
        info = fig.add_subplot(bottom[1, 1])
        info.axis("off")
        block = (f"{method}  ·  {shots} 发  ·  seed {seed}  ·  单发标准差 {np.std(measured_ed):.3g}\n"
                 f"设定 XYZ  {_xyz_text(center)} µm    各轴抖动 ±{position_half_width:g} µm\n"
                 f"◆  最高单发 #{best + 1}   ED {measured_ed[best]:.7g}    取样 XYZ {_xyz_text(positions[best])} µm")
        info.text(.01, .90, block, va="top", linespacing=1.65, fontsize=10, color="#23374a")
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160, bbox_inches="tight")
        if show:
            plt.show()
    return fig, data


if __name__ == "__main__":
    plot_fitter_model(save_path="fitter_model.png")
