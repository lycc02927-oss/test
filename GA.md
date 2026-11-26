# ED 仿真遗传算法调参

在 [ga_tuning/run.py](ga_tuning/run.py) 顶部编辑 `MODELS`，可组合数据拟合工况和自定义峰高、位置、宽度的公式工况。在仓库根目录运行 `python -m ga_tuning.run`。默认选择 LT 整机仿真、中心高斯和偏置高斯三份数据，采用 RBF 拟合。

每代按个体的综合 loss 升序排名，计算式为平均基础 loss 加 30 轮失败率的六分之一。standard 方案使用 20 个个体、15 代。终端逐个打印 loss，结果目录包含代际曲线和 Excel 工作簿。

[运行、模型配置和评分示例](ga_tuning/README.md)
