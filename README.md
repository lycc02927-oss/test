# Plasma BO Search

新版三维贝叶斯寻优：Matérn 5/2 + 可选 WhiteKernel、标准化 GPR、负 EI 与 L-BFGS-B。
使用 `fitter.py` 的单发 ED/PF 量测，空间模型可选择试验数据拟合或数学公式。

## 运行

在 Linux 下使用 Python 3.9 或更新版本：

```bash
python3 -m pip install -r requirements.txt
python3 search.py
```

默认：数据集 3（LT 精细仿真）、RBF 拟合、最大化 ED、种子 42、40 个初始点、最多 40 轮寻优。
提前停止仍由 `stop_no_improve` 决定，坐标范围为三轴 `[-300, 300]`。

```bash
# 最大化 ED/PF
python3 search.py --target ed_pf_ratio

# 使用数学公式
python3 search.py --source formula --formula-model formula_offset_gaussian

# 无图形界面的 Linux
MPLBACKEND=Agg python3 search.py --no-plot

# 短流程运行，并启用 WhiteKernel
python3 search.py --max-iter 5 --init-points 10 --sigma3-rel 0.03 --no-plot
```

## 修改模型和参数

常用配置集中在 `search.py` 顶部，也可以用 `set_ed_model()` 或 `run_search()` 传参。

| 配置 | 可选值或作用 |
| --- | --- |
| `TARGET` | `ed` 最大化 ED；`ed_pf_ratio` 最大化 ED/PF |
| `ED_SOURCE` | `fitted` 数据拟合；`formula` 数学公式 |
| `DATA_MODEL` | 1=P3 试验；2=LT 仿真；3=LT 精细仿真；4=中心高斯数据；5=偏置高斯数据；6=多峰高斯数据 |
| `FIT_METHOD` | `rbf` / `manual_gpr` / `linear` / `polynomial` / `svm` |
| `FORMULA_MODEL` | `formula_gaussian` / `formula_offset_gaussian` / `formula_multi_gaussian` |
| `RANDOM_SEED` | 控制 LHS、GP 超参数重启、EI 选点和量测随机流；默认 42 |
| `BOUNDS` | X/Y/Z 三轴的中心设定值范围 |
| `BO_PARAMETERS` | 初始点数、迭代次数、xi、噪声假设、探索比例、提前停止等 |

`FIT_METHOD` 用于建立作为量测来源的 ED 空间模型。寻优代理始终使用本次提供的 `GPR.py`，两者作用不同。
公式的幅值、中心和宽度在 `formula_models.py` 的 `FORMULA_CONFIGS` 中修改。
公式 ED 与拟合 ED 都通过同一个 `EDPFModel` 生成单发 ED/PF，ED 能量尺度不取决于 `PF_BASELINE`。

```python
from search import run_search

history = run_search(
    source="fitted", data_model=3, fit_method="rbf",
    target="ed_pf_ratio", seed=42, show=True,
    max_iter=40, init_points=40,
)
```

每次 `run_search()` 都重建量测模型并重置随机流，相同配置和软件环境下可以复现。
直接调用 `bayesian_optimization()` 只重置寻优随机数；若复用已有量测模型，需要先重新调用 `set_ed_model()`。

## ED/PF 和抖动

完整沿用原 `fitter.py`：每次量测先决定是否触发同步扰动，再生成本轮数据。
默认位置在中心的三轴 ±10 范围内采样；未触发时 PF=6，只有 ED 受位置影响。
默认 10% 概率触发；触发后 PF 在基值 ±5% 范围内变化，ED 跟随 PF 相对变化的 50%，另加 ±0.5% 小扰动。
参数常量仍放在 `fitter.py` 顶部；也可通过 `run_search(sampling_parameters=...)` 单次覆盖。

初始采样及每轮寻优都会打印 `ED`、`PF`、`ED/PF` 和目标值 `V`。
打印与记录复用已经获取的那一发，最终结果也从历史记录读取，不重新量测最佳坐标。

`sigma3_rel` 是寻优器的噪声假设，与上述模型生成的抖动分别设置。
`sigma3_rel=0` 关闭 WhiteKernel，并不会关闭模型量测抖动；大于 0 时，`3σ = sigma3_rel × 初始目标值范围`。
WhiteKernel 的噪声方差在标准化空间拟合，EI 使用去掉白噪声项后的后验标准差。

## 结果和绘图

`history['X']`、`history['y']`、`history['ed']`、`history['pf']`、`history['ed_pf_ratio']` 长度一致，含初始采样和 BO 迭代。
`best_value` / `best_x` / `time` 长度为 `1 + 实际 BO 轮数`；
`distance_to_nearest` / `noise_level` / `sigma_abs_used` / `xi_used` 每轮一个值。
`sigma_abs_used` 记录该轮实际用于稳健阈值的绝对标准差。

绘图沿用本次提供的 `plot_optimization_results()` 四宫格：

1. 每轮改进量与 xi 调度。
2. 包含初始扫描的最近样本距离。
3. 三维样本分布与最佳点。
4. 全部评估点与累计最优值。

无界面时可以保存图：

```python
from search import run_search, plot_optimization_results

history = run_search(show=False)
figure = plot_optimization_results(history, show=False)
figure.savefig("optimization.png", dpi=160)
```

## 与本次原始算法的调整

- 保留后半程搜索边界收缩、EI 平坦时局部精修和最小步距兜底。
- `xi_final > 0` 时仍为指数衰减；`xi_final=0` 时改为线性降到 0，避免原来的零比值使第二轮立即归零。
- EI 平坦判断的阈值换算到标准化空间，避免与标准化 EI 混用量纲。
- 增加同发 ED/PF 历史与打印、固定种子、命令行及无界面绘图支持。

`fitter.py`、`formula_models.py`、`measurement_data.py`、`fit_models/`、`LHS.py`、`use_fitter.py`、`time_jitter.py`
来源于 [lycc02927-oss/op](https://github.com/lycc02927-oss/op) 提交
`302944327172311299c5fbda6a94e651520c01bd`，保持原内容；旧寻优器和批量实验入口未复制。
`use_fitter.py` 保留原设备 Mock 接口，`from fitter import run_pipeline` 仍然有效。

## 验证

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python3 -m unittest discover -s tests -v
```

检查核/LML 梯度、后验方差、两种模型来源和两种目标的同发对应关系、固定种子复现、xi 调度与绘图。
