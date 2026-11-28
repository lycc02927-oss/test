# 用遗传算法选择三维寻优参数

内层沿用 `search.run_search` 的三维 GPR + EI。外层每条染色体给出一套 BO 参数，通过多模型、多种子的仿真筛选配置。六种默认数据分别是 P3、LT、LT 精细、中心高斯、偏置高斯、多峰高斯，均使用数据表的 RBF 拟合面。每种数据各跑 ED 和 ED/PF。

## 变量由谁决定

| 类型 | 名称及当前默认值 | 在一次 BO 寻优中如何变化 |
| --- | --- | --- |
| 内层决策变量 | x、y、z，各轴边界 [−300, 300] | LHS 和 EI 依次选择设定位置 |
| 观测量 | ED、PF、ED/PF、y_obs | 每个设定位置采一发并追加观测。目标按 TARGET 取 ED 或 ED/PF |
| 每轮重学的 GP 参数 | Matérn 单一长度尺度、核幅度，可选白噪声方差 | 在指定界限内最大化对数边际似然。白噪声仅在 sigma3_rel>0 时启用 |
| 由公式算出的变量 | xi_i、r_explore、min_step_i、sigma_used、best_robust | 随迭代进度或观测更新。min_step_i=2(1−t)²；半径为 max(r_frac_min, 0.5(1−t)²)×平均跨度 |
| 需要选择的 BO 超参数 | init_points=40，max_iter=40，xi_init=0.01，xi_final=0，global_prob=0.40，r_frac_min=0.05，sigma3_rel=0，stop_no_improve=30，ls_bounds=(40,2000)，n_restarts_ei=20 | 上层指定，或沿用正常 run_search 入口的默认值。ls_bounds 限定拟合范围，具体长度尺度仍由 GP 学习 |
| 当前实现中的定值 | LHS 的 XY/Z Beta 形状参数 2/3，GP 优化随机重启 10，alpha=1e−10，后半程搜索边界倍数 5，EI 平坦阈值系数 0.001，局部候选 256 个 | 写在 LHS.py、search.py 等函数内部，本轮 GA 固定这些值 |
| 实验条件 | DATA_MODEL、FIT_METHOD、TARGET、BOUNDS、seed、位置扰动半宽 10、PF 基值 6 和抖动触发概率 10% | 它们定义评价环境。不同候选使用相同场景和配对种子 |

普通入口 `run_search` 会将文件顶部的 BO_PARAMETERS 与调用参数合并。直接调用底层 `bayesian_optimization` 有另一套函数签名默认值。这里采用普通入口，并显式给出染色体参数及预算。

## 哪些值交给 GA

| 基因 | 候选取值 |
| --- | --- |
| init_points | 10、20、30、40，只保留小于总预算的值 |
| xi_mode | fixed 采用显式 0.01 → 0，线性调度。auto 传入 None、None，按初始极差和启用的噪声下限确定起止值 |
| global_prob | 0.1、0.4、0.7，表示每个 EI 解算起点从全域抽取的概率 |
| r_frac_min | 0.025、0.05、0.10 |
| sigma3_rel | 0、0.03、0.10。以初始目标值极差为标度的相对 3σ |
| stop_no_improve | 10、20、30 |
| ls_bounds | (20,300)、(40,600)、(40,2000) |

编辑 `ga_tune.py` 的 `gene_space()` 即可更换候选范围。每套的 `max_iter = budget − init_points`，可以提前停止。总预算为 80 时，默认原算法的 40+40 配置作为对照进入初代。小于等于 40 发预算时，对照使用可行的最大初始档位。

初代包含人工对照及随机配置。每代保留冠军，通过三选一锦标赛选父母、逐基因均匀交叉和独立突变生成下一代。重复基因组合复用开发集评分。混合离散搜索空间中的优胜者是本次实际访问候选中的最佳。

## 评分和验证

每次内层执行带抖动的寻优，取观测最高那一发的**设定坐标**，用无抖动 RBF 回测 ED；ED/PF 回测取同位置 ED 除以 6。参考取 batch_test.make_references 生成的各模型数值参考峰。偏差 = 100 × |回测值 − 参考值| / |参考值|，默认不超过 10% 计一次命中。P3 默认全域参考点位于原试验数据范围之外，仿真成绩反映拟合面的表现。

通用配置按以下优先级比较，前一项相同才看后一项：最差模型/目标的开发集达标率最高 → 场景平均达标率最高 → 所有评估的平均总发数最少 → 平均百分比误差最小。总发数包含初始点和实际 BO 量测。专用配置从同一批已评估候选中，按单个模型与目标重新排名。X/Y/Z 路程单列留作比较。

内层沿用 search.py 原来的无显著改进停止条件；10% 是外层评价返回坐标的精度门槛。这样可以同时观察提前停止、省了多少发、返回位置仍能否达标。

开发集选完后锁定人工对照、通用候选和各场景专用候选，再用独立种子逐场景验证。默认开发 2 个种子、验证 3 个，适合初筛；正式定型应增加开发种子并预留未用于调参的验证种子，例如 20 个。种子少时一次成败即可改变排序。

## 运行和读取结果

在与 search.py 相同的项目目录和 Python 环境中运行：

```bash
python ga_tune.py --workers 4
```

默认种群 8、共 3 代，含初代；每基因突变率 0.2。默认六模型、两个目标，每次最多 80 发。若只关注 LT 精细和多峰的 ED，可先缩小规模：

```bash
python ga_tune.py --models 3 6 --targets ed --population 4 --generations 2 --train-seeds 42 --valid-seeds 142 --workers 2
```

进程数由本机可用 CPU 和内存决定。若 BLAS 已设置多线程，Linux 下可在命令前加 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`，避免每个进程重复开很多计算线程。每次新建 `ga_results/<时间戳>/`。

| 文件 | 用途 |
| --- | --- |
| selected.json | 通用方案和逐模型、逐目标的开发集入选参数，可用于 run_search 的 BO 参数 |
| validation.csv | 一行对应模型、目标和方案角色，比较 baseline、common、specialist 的独立验证成绩 |
| trials.csv | 每次仿真的种子、实测最高值、对应设定 XYZ、无抖动回测、误差、BO 次数、总发数、三轴路程 |
| candidates.csv / generations.csv | 每个候选编号的基因，以及各代排名、达标率、平均发数与误差 |
| config.json / references.csv | 本次运行参数和无抖动参考位置与目标值 |

复用 selected.json 中某套参数时，同时指定对应 data_model、target、source='fitted' 和 fit_method='rbf'，并用新的 seed 做实验。
