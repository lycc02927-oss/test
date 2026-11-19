"""ED 拟合模型的统一导入入口，供 fitter.py 使用。
通过 create_model(模型名称, 参数字典) 创建模型，再 fit(X, y)、predict(三维坐标)。
保留 manual_gpr、rbf、linear、polynomial、svm 五种方法，具体参数见对应文件。"""
from .factory import MODEL_TYPES, create_model
from .linear_model import LinearInterpolationModel
from .manual_gpr import ManualGPRModel
from .polynomial_model import PolynomialModel
from .rbf_model import RBFModel
from .svm_model import SVMModel

__all__ = [
    "MODEL_TYPES",
    "create_model",
    "ManualGPRModel",
    "RBFModel",
    "LinearInterpolationModel",
    "PolynomialModel",
    "SVMModel",
]
