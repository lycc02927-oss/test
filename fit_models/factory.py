"""根据名称创建 ED 拟合模型，主要由 fitter.py 调用。
model_type 可选 manual_gpr、rbf、linear、polynomial、svm。
parameters 是对应模型构造参数的字典；不传时采用默认参数。"""
from .linear_model import LinearInterpolationModel
from .manual_gpr import ManualGPRModel
from .polynomial_model import PolynomialModel
from .rbf_model import RBFModel
from .svm_model import SVMModel


MODEL_TYPES = ("manual_gpr", "rbf", "linear", "polynomial", "svm")


def create_model(model_type, parameters=None):
    """按 model_type 创建模型，parameters 传给构造函数；返回尚未训练的对象。"""
    factories = {
        "manual_gpr": ManualGPRModel,
        "rbf": RBFModel,
        "linear": LinearInterpolationModel,
        "polynomial": PolynomialModel,
        "svm": SVMModel,
    }
    model = factories[model_type](**(parameters or {}))
    model.model_type = model_type
    return model
