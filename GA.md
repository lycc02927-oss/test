# ED 仿真遗传算法调参

在仓库根目录运行 `python -m ga_tuning.run`。standard 方案使用 20 个个体、15 代，通过 `search.run_search()` 在 LT 整机仿真（2）、中心高斯（3）、偏置高斯（4）上评估同一组 BO 参数。终端逐个打印 loss，输出目录包含代际曲线 `ga_progress.png` 和 `GA_results.xlsx`。

[运行、评分和曲线说明](ga_tuning/README.md)
