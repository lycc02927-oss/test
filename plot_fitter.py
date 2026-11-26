"""画 ED 模型的三维分布、过峰切片及固定设定位置的单发抖动。

直接运行 python plot_fitter.py 画默认高斯公式。指定模型示例：
python plot_fitter.py --formula-model formula_multi_gaussian
python plot_fitter.py --data-model 2 --fit-method rbf

from plot_fitter import plot_fitter_model
fig, data = plot_fitter_model(data_model=2, fit_method="rbf", save_path="lt_system.png")
fig, data = plot_fitter_model(formula_model="formula_truncated_peak", setpoint=(130, 0, 0))
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.optimize import minimize

from fit_models import MODEL_TYPES
from fitter import EDFitter, EDPFModel, FORMULA_CONFIGS, build_formula_ed_model
from measurement_data import MODEL_DATA, MODEL_NAMES


# 直接运行本文件时使用的公式。峰参数默认从 fitter.py 读取。
DEFAULT_FORMULA = "formula_gaussian"


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


def _plot_formula_surface(ax, plane, peak, peak_ed):
    """固定 Z 在最高点处，直接用同一张 XY 切片的无抖动 ED 作曲面高度。"""
    _, _, _, xs, ys, ed = plane
    xx, yy = np.meshgrid(xs, ys)
    ax.plot_surface(xx, yy, ed, rstride=1, cstride=1, color="#4e9caa",
                    linewidth=0, antialiased=True)
    ax.scatter(peak[0], peak[1], peak_ed, marker="*", s=135,
               c="#ffcd55", edgecolors="#24292d", zorder=4)
    ax.set(xlabel="X (µm)", ylabel="Y (µm)", zlabel="静态 ED",
           xlim=(xs[0], xs[-1]), ylim=(ys[0], ys[-1]),
           title=f"公式曲面  ·  固定 Z = {peak[2]:.2f} µm")
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), .65 * max(np.ptp(xs), np.ptp(ys))), zoom=.9)
    ax.view_init(elev=27, azim=-55)


def plot_fitter_model(data_model=2, fit_method="rbf", *, formula_model=None, peaks=None,
                      fit_parameters=None, bounds=((-300, 300),) * 3, setpoint=None,
                      shots=100, seed=42, position_half_width=10.0,
                      ed_time_parameters=None, pf_time_parameters=None,
                      save_path=None, show=True):
    """返回 (Figure, data)。bounds 按 XYZ 排列，单位 µm；默认在静态峰位重复量测。

    数据模式从 MODEL_DATA 读取，fit_parameters 与 EDFitter 一致。传 formula_model
    时使用公式模型及其 peaks，并额外以 ED 为高度绘制穿过最高点的 XY 曲面。
    data 包含每一发的真实取样坐标、位置抖动后的空间 ED、
    最终单发 ED、PF，以及静态数值极值 peak_ed/peak_xyz 和 min_ed/min_xyz、
    观测最高发次（从 1 开始）。同值有多处时标记其中一个位置。
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
    slice_axes = []
    for axis in range(3):
        a = np.append(np.linspace(*bounds[axis], 81), peak[axis])
        if formula_model is not None:
            # 同时采到各峰中心及峰宽内的若干点，缩小 sigma 后仍能看清峰形。
            offsets = model.sigmas[:, axis, None] * np.linspace(-2, 2, 9)
            local = model.centers[:, axis, None] + offsets
            a = np.append(a, np.clip(local, *bounds[axis]))
        slice_axes.append(np.unique(a))
    planes = []
    for i, j, fixed in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        a, b = slice_axes[i], slice_axes[j]
        xx, yy = np.meshgrid(a, b)
        xyz = np.tile(peak, (xx.size, 1))
        xyz[:, i], xyz[:, j] = xx.ravel(), yy.ravel()
        z = _values(model, xyz).reshape(xx.shape)
        planes.append((i, j, fixed, a, b, z))
    norm = Normalize(min(low_ed, *(p[-1].min() for p in planes)),
                     max(peak_ed, *(p[-1].max() for p in planes)))
    style = {"font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
                                 "SimHei", "PingFang SC", "DejaVu Sans"], "axes.unicode_minus": False}
    with plt.rc_context(style):
        if formula_model is not None:
            fig = plt.figure(figsize=(13.4, 12.2), layout="constrained")
            rows = fig.add_gridspec(3, 1, height_ratios=[1.25, 1.04, .65])
            views = rows[0].subgridspec(2, 2, height_ratios=[4, .85])
            surface = fig.add_subplot(views[:, 0], projection="3d", computed_zorder=False)
            _plot_formula_surface(surface, planes[0], peak, peak_ed)
            volume_cell, extrema_cell = views[0, 1], views[1, 1]
            slices_cell = rows[1]
            shot_row = rows[2].subgridspec(1, 2, width_ratios=[1.25, 1])
            shots_cell, info_cell = shot_row[0, 0], shot_row[0, 1]
        else:
            fig = plt.figure(figsize=(13.4, 9.8), layout="constrained")
            rows = fig.add_gridspec(2, 1, height_ratios=[.98, 1.34])
            slices_cell = rows[0]
            bottom = rows[1].subgridspec(2, 2, width_ratios=[.92, 1.18], height_ratios=[3.1, .85])
            volume_cell, extrema_cell = bottom[0, 0], bottom[1, 0]
            shots_cell, info_cell = bottom[0, 1], bottom[1, 1]
        top = slices_cell.subgridspec(1, 4, width_ratios=[1, 1, 1, .046])
        fig.suptitle(f"{title}   |   ED 空间分布与单发抖动", fontsize=19, fontweight="bold")
        for col, (i, j, fixed, a, b, z) in enumerate(planes):
            ax = fig.add_subplot(top[0, col])
            im = ax.pcolormesh(a, b, z, shading="auto", cmap="viridis", norm=norm)
            ax.scatter(peak[i], peak[j], marker="*", s=155, c="#ffcd55", edgecolors="#24292d")
            ax.set(xlim=bounds[i], ylim=bounds[j], aspect="equal",
                   xlabel=f"{'XYZ'[i]} (µm)", ylabel=f"{'XYZ'[j]} (µm)",
                   title=f"{'XYZ'[i]}{'XYZ'[j]} 切片  ·  {'XYZ'[fixed]} = {peak[fixed]:.2f} µm")
        fig.colorbar(im, cax=fig.add_subplot(top[0, 3]), label="静态 ED")

        ax3d = fig.add_subplot(volume_cell, projection="3d", computed_zorder=False)
        # 三维图只抽稀显示规则网格；极值仍使用全部候选点和局部精修结果。
        volume_ed = predictions[len(anchors):].reshape(grid.shape[:-1])
        visible = grid[::3, ::3, ::3].reshape(-1, 3)
        colors = volume_ed[::3, ::3, ::3].ravel()
        ax3d.scatter(*visible.T, c=colors, cmap="viridis", norm=norm, alpha=.48,
                     s=12, linewidths=0, depthshade=False, zorder=1)
        if formula_model is not None and len(anchors) > 1:
            in_bounds = np.all((anchors >= bounds[:, 0]) & (anchors <= bounds[:, 1]), axis=1)
            secondary = anchors[in_bounds & ~np.all(np.isclose(anchors, peak), axis=1)]
            if len(secondary):
                ax3d.scatter(*secondary.T, marker="D", s=50, facecolors="white",
                             edgecolors="#285362", depthshade=False, zorder=2, label="其他设定峰中心")
                ax3d.legend(loc="upper left", frameon=False, fontsize=8)
        ax3d.scatter(*low, marker="v", s=110, c="#66ccff", edgecolors="#13283f", zorder=3)
        ax3d.scatter(*peak, marker="*", s=175, c="#ffcd55", edgecolors="#24292d", zorder=4)
        ax3d.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2],
                 xlabel="X (µm)", ylabel="Y (µm)", zlabel="Z (µm)", title="三维静态 ED 分布")
        ax3d.set_box_aspect(np.ptp(bounds, axis=1), zoom=.9)
        for axis, (start, stop) in zip((ax3d.xaxis, ax3d.yaxis, ax3d.zaxis), bounds):
            axis.set_ticks(np.linspace(start, stop, 3))
            axis.set_tick_params(labelsize=8)
        ax3d.view_init(elev=23, azim=-53)
        extrema = fig.add_subplot(extrema_cell)
        extrema.axis("off")
        for y, marker, color, label, val, point in [
            (.74, "*", "#ffcd55", "最大", peak_ed, peak),
            (.32, "v", "#66ccff", "最小", low_ed, low),
        ]:
            extrema.scatter(.04, y, marker=marker, s=105, c=color, edgecolors="#24292d",
                            transform=extrema.transAxes)
            extrema.text(.09, y, f"{label} ED  {val:.7g}     XYZ  {_xyz_text(point)} µm",
                         va="center", transform=extrema.transAxes)

        ax = fig.add_subplot(shots_cell)
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
        info = fig.add_subplot(info_cell)
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
    import argparse

    parser = argparse.ArgumentParser(description="绘制 fitter 的 ED 空间分布、三维图和抖动。")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--formula-model", choices=FORMULA_CONFIGS, help="公式模型名")
    source.add_argument("--data-model", type=int, choices=MODEL_NAMES, help="使用拟合数据的编号")
    parser.add_argument("--fit-method", choices=MODEL_TYPES, default="rbf", help="数据拟合方法")
    parser.add_argument("--out", type=Path, default="fitter_model.png", help="输出图路径")
    parser.add_argument("--no-show", action="store_true", help="只保存图片")
    args = parser.parse_args()
    if args.data_model is None:
        options = dict(formula_model=args.formula_model or DEFAULT_FORMULA)
    else:
        options = dict(data_model=args.data_model, fit_method=args.fit_method)
    plot_fitter_model(**options, save_path=args.out, show=not args.no_show)
    print(f"图片已保存到 {Path(args.out).resolve()}")
