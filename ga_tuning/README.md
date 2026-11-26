# 用遗传算法选择 BO 参数

在仓库根目录运行以下命令：

```bash
python -m ga_tuning.run
```

程序依赖 NumPy、SciPy、Matplotlib 和 openpyxl。结果保存在 `ga_results/运行时间/`。只有 standard 一套方案。如果需要指定并行数和新输出目录，可以运行 `python -m ga_tuning.run --workers 8 --out ga_run_01`。种群、代数和种子集中在 [design.py](design.py) 的 `Standard`。

| 设置 | standard |
|---|---|
| 种群、代数 | 20 个个体，15 代，另测一次第 0 代默认基线 |
| 模型 | 默认选择 2 LT 整机仿真、3 中心高斯、4 偏置高斯三份数据做 RBF 拟合。可自由组合，所选工况等权 |
| 训练 | 每个个体在每个选定工况上跑种子 42、43、44，共 3 × 工况数场 |
| 独立验证 | 默认参数与 GA 冠军分别在每个选定工况上跑种子 1001–1010，各 10 × 工况数场 |
| 每场预算 | 初始采样与后续 BO 采样合计最多 90 个点 |

## 在 run.py 顶部选模型

编辑 [run.py](run.py) 的 `MODELS`。每行左侧写一个不同的工况名，右侧选一份数据或一个公式。保留的条目一起参与训练和验证。同一模型也可以用不同拟合方法或不同峰宽做两种工况。以下示例同时评估 LT 数据和一个自定义双峰。

```python
MODELS = {
    "LT 整机 RBF": fitted(2, "rbf"),
    "自定义双峰": formula("formula_multi_gaussian",
        peak(peak_ed=0.70, center=(-80, 0, 30), sigma=(50, 60, 40)),
        peak(peak_ed=1.00, center=(110, 40, -20), sigma=(70, 50, 60))),
}
```

| 模型 | 填写方式 | 评分参考 |
|---|---|---|
| 数据拟合 | `fitted(编号, 方法)`。编号 1–5 依次是 P3 试验、LT 整机仿真、中心高斯、偏置高斯、多峰高斯。方法可填 `rbf`、`manual_gpr`、`linear`、`polynomial`、`svm` | 原始数据表中最大的 ED 及其坐标。不同拟合方法可共用这个参考 |
| 高斯和偏置高斯公式 | `formula("formula_gaussian", peak(...))` 或 `formula("formula_offset_gaussian", peak(...))` | 所填峰的 `peak_ed` 和 `center` |
| 多峰高斯公式 | `formula("formula_multi_gaussian", peak(...), peak(...))`。每增删一个 `peak` 就增减一座峰 | 所有峰中最大的 `peak_ed` 及对应的 `center` |
| 渐变平台、渐变截断峰 | `formula("formula_plateau", peak(...))`、`formula("formula_truncated_peak", peak(...))`。各填一座峰，底值为 0.2 | 所填峰的 `peak_ed` 和 `center` |

`peak_ed` 是无抖动的设计峰值，`center` 是峰中心，`sigma` 控制三个轴方向的宽度。中心和宽度按 XYZ 顺序填写，单位 µm。中心应在本次 BO 搜索范围内（默认各轴 −300 到 300 µm）。省略全部 `peak(...)` 时沿用 fitter 的默认峰参数。每场 BO 都经过同一条带抖动的 ED 量测路径。

## 九个参数的范围与默认基线

| 参数 | GA 可选值 | 当前 BO 默认值 |
|---|---|---:|
| sigma3_rel | 0–1，步长 0.01 | 0 |
| n_restarts | 1、10、20、…、100 | 10 |
| n_restarts_ei | 1、10、20、…、100 | 20 |
| xi_init | 0.005–0.100，步长 0.001 | 0.010 |
| xi_final | 0–0.010，步长 0.001 | 0 |
| global_prob | 0–1，步长 0.1 | 0.40 |
| ls_lo | 1–100，步长 1 µm | 40 |
| ls_hi | 100–300，步长 1 µm | 2000 |
| init_points | 10–50，步长 1 点 | 40 |

## 怎样评分

同一套参数在一个模型、一个种子上完成一次 BO。每场记录两项成本：

1. **基础 loss** 为 `(3 × ED相对偏差/0.10 + 5 × 总点数/90 + 2 × sum_step_max/54000) / 10`。ED 相对偏差取历史最高单发 ED 相对设计峰值的绝对偏差。每段按三轴位移绝对值中的最大值计路程并求和，包括最后回到最佳点的一段。
2. **30 轮失败标记**记录一次 BO 在预算内是否曾满足“连续 30 轮没有显著改善”。满足记 0，全程未满足记 1。沿用 BO 已记录的噪声和显著改善判据。

每代给每个个体打分时，先对同一工况的重复试验取平均，再对本次选定的工况等权平均。**个体综合 loss = 该个体平均基础 loss + 该个体的 30 轮失败率 / 6**。失败率按 0–1 的比例代入，GA 只按综合 loss 升序排名。例：A 的平均基础 loss 为 0.40，9 场里失败 3 场，得分为 `0.40 + (3/9)/6 = 0.45556`。B 的平均基础 loss 为 0.47，零失败，得分为 0.47，A 排第一。

整代种群也按同一公式汇总：`population_mean_loss = population_mean_base_loss + population_failure_rate / 6`。这等于当代所有个体综合 loss 的平均值。

BO 提前停止仍要求同一轮既满足 30 轮判据，又满足 ED 偏差 ≤10%。若已经出现连续 30 轮无显著改善、ED 偏差仍超限，BO 继续采样，失败标记为 0，ED 偏差和额外采样计入基础 loss。`success`、`hits` 单独报告联合停机情况。

## 怎样看曲线和日志

第 0 个点只评估当前 BO 默认参数。基线和每个候选都直接调用 [search.run_search](../search.py)，默认 GP 和 EI 重启次数也从 `search.BO_PARAMETERS` 读取。基线和 GA 个体使用同一批训练种子、相同预算与停止条件。默认的长度尺度上限 2000 可超出 GA 的候选范围 100–300，基线保留真实取值。

每个新个体完成当前组合的全部训练场次后，终端立即打印代数、个体编号、综合 loss、基础 loss 和失败率。精英和已测过的染色体复用成绩。每代成员的编号、名次、平均基础 loss、失败率与综合 loss 都保存到工作簿。独立验证单独打印。

| 输出 | 看什么 |
|---|---|
| `ga_progress.png` | 点 0 为默认 BO；实线为各代最低训练 loss，点线为当代种群平均 loss，水平虚线为默认基线。越低越好 |
| `generations.csv` / Excel「代际进展」 | `mean_loss` 是当代冠军跨工况平均 loss；`improvement` 是基线减去冠军，正值代表提升 |
| Excel「配方总评」 | 比较 `mean_base_loss`、`failure_rate` 和 `mean_loss`。分别筛选 train 与 validation |
| Excel「逐次试验」 | `patience_round` 是首次满足 30 轮判据的 BO 轮数。空值表示全程未达到。每场还可核对 `base_loss`、`patience_failed`、`loss_check` |
| Excel「模型参考」「公式峰参数」 | 每个工况的来源、拟合方法、设计峰值和位置；选用公式时，逐峰记录峰高及三个轴的中心和宽度 |
| Excel「采样轨迹」「入选配方」 | 查看逐点 XYZ、ED、PF 和训练冠军、默认基线的九项参数 |

以下为上一版 50/50 评分的历史示例：[查看历史曲线](results/ga_progress_example.png) · [下载这次的 Excel](results/GA_progress_example.xlsx)。示例用 4 个个体、3 代，每个个体同样跑 9 场训练；独立验证用种子 1001、1002，合计 111 场 BO。第 0–3 代曲线上的 loss 依次是 0.553162、0.187574、0.177799、0.174484。

默认基线独立于遗传种群，第一代的最佳个体可以比默认参数差。从第 1 代起保留精英，最佳训练 loss 持平或下降。独立验证用于检验换种子后的表现。

## 改参数时看哪个文件

| 文件 | 作用 |
|---|---|
| [encoding.py](encoding.py) | 九项参数的取值档位、二进制编码及交叉变异后的越界修复 |
| [genetic.py](genetic.py) | 排名轮盘赌、两点交叉、逐位变异、精英保留。每连续两代未提升时增加选择指数、交叉率和变异率 |
| [simulation.py](simulation.py) | 读取默认 BO 参数，并用现有 fitter 和 BO 执行一场试验 |
| [objective.py](objective.py) | 计算基础 loss、30 轮失败项和最终评分 |
| [run.py](run.py) / [report.py](report.py) | 运行、打印、生成曲线和汇总；[excel.py](excel.py) 导出工作簿 |

历史四工况试验和原工作簿使用当时的编号、评分及方案，见[历史结果](RESULTS.md)。
