
import random
from dataclasses import dataclass

import numpy as np

from fitter import run_pipeline
from formula_models import build_formula_ed_model
from time_jitter import PFTimeJitter, TimeJitter

MODEL_MODE = "fitted"  # fitted, formula_gaussian, formula_offset_gaussian, formula_multi_gaussian

class MockDCDXPStatusServiceClient:
    """
    MOCK DC DCDXP Status RPC
    """

    def __init__(self, logger, ready_status):
        self.logger = logger
        self.ready_status = ready_status

    def GetStatus(self, request_msg, response_msg, context):
        self.logger.info("[MOCK] DCDXP GetStatus called.")

        # 模拟 DC 状态正常
        response_msg.status = self.ready_status

        self.logger.info(
            f"[MOCK] DCDXP GetStatus success, status={response_msg.status}"
        )

        # RPC调用成功
        return True, 0


class MockILIMPServiceClient:
    """
    MOCK IL ILIMP Status RPC
    """

    def __init__(self, logger, ready_status):
        self.logger = logger
        self.ready_status = ready_status

    def GetStatus(self, request_msg, response_msg, context):
        self.logger.info("[MOCK] ILIMP GetStatus called.")

        # 模拟 IL 状态正常
        response_msg.status = self.ready_status

        self.logger.info(
            f"[MOCK] ILIMP GetStatus success, status={response_msg.status}"
        )

        return True, 0

class MockRMPServiceAClient:
    """
    MOCK REMA RMP Status RPC
    """

    def __init__(self, logger, ready_status):
        self.logger = logger
        self.ready_status = ready_status

    def RMPGetStatusA(self, request_msg, response_msg, context):
        self.logger.info("[MOCK] RMP RMPGetStatusA called.")

        # 模拟 REMA 状态正常
        response_msg.status = self.ready_status

        self.logger.info(
            f"[MOCK] RMP RMPGetStatusA success, status={response_msg.status}"
        )

        return True, 0

class MockILIMPServiceTClient:
    """
    MOCK IL illumination mode RPC
    """

    def __init__(self, logger):
        self.logger = logger

    def SetIlluminationModeT(self, request_msg, response_msg, attr=None):
        self.logger.info("[MOCK] ILIMP SetIlluminationModeT called.")
        self.logger.info(
            f"[MOCK] SetIlluminationModeT request: {request_msg}"
        )

        # 不真正调用 IL，直接模拟 RPC 成功
        return True, 0


class MockRMPServiceTClient:
    """
    MOCK REMA close window RPC
    """

    def __init__(self, logger):
        self.logger = logger

    def RMPCloseWindowT(self, request_msg, response_msg, context):
        self.logger.info("[MOCK] RMP RMPCloseWindowT called.")

        # 不真正调用 REMA，直接模拟 RPC 成功
        return True, 0

class MockDCDXPServiceTClient:
    """
    Mock DCDXP T service:
    - 启动曝光
    - 停止曝光
    """

    def __init__(self, logger):
        self.logger = logger

    def StartContinuousPulseExpose(self, request_msg, response_msg, attr=None):
        """
        Mock 启动连续脉冲曝光。

        不校验任何曝光参数，所有请求直接返回成功。
        """
        self.logger.debug(
            f"[MOCK] StartContinuousPulseExpose request: {request_msg}"
        )
        return (True, 0)

    def StopContinuousPulseExpose(self, request_msg, response_msg, attr=None):
        """
        Mock 停止连续脉冲曝光。
        """
        self.logger.debug("[MOCK] StopContinuousPulseExpose.")
        return (True, 0)

class MockDCLGPServiceTClient:
    """
    MOCK DCLGP plasma position setpoint RPC
    """

    def __init__(self, logger, seed=42):
        self.logger = logger
        self._rng = random.Random(seed)
        #后续调用位置到达判断用
        # self.current_position = None
        self.current_position = (
            self._rng.uniform(-200.0, 200.0),
            self._rng.uniform(-200.0, 200.0),
            self._rng.uniform(-200.0, 200.0),
        )
        #posionmodel
        # self.position_model = position_model

    def GetPlasmaPositionSetpoint(
            self,
            request_msg,
            response_msg,
            attr=None
    ):
        """
        获取等离子体位置设定值。

        Mock初始化时随机生成 [-200, 200] um 范围的位置。
        后续返回最近一次设置的位置。
        """

        # 第一次获取时，随机生成初始位置
        if self.current_position is None:
            self.current_position = (
                self._rng.uniform(-200.0, 200.0),
                self._rng.uniform(-200.0, 200.0),
                self._rng.uniform(-200.0, 200.0),
            )

        x_um, y_um, z_um = self.current_position

        # DC接口单位为m
        response_msg.plasmaX = x_um / 1e6
        response_msg.plasmaY = y_um / 1e6
        response_msg.plasmaZ = z_um / 1e6

        self.logger.info(
            "[MOCK] GetPlasmaPositionSetpoint success, "
            f"x={x_um:.2f} um, "
            f"y={y_um:.2f} um, "
            f"z={z_um:.2f} um"
        )

        return True, 0

    # def SetPlasmaPositionSetpoint(
    #     self,
    #     request_msg,
    #     response_msg,
    #     attr=None
    # ):
    #     """
    #     拦截DC设置等离子体位置接口，
    #     将位置参数转发给位置模型。DC下发的是m，位置模型需要um，因此转换成um后转发给位置模型。
    #     """
    #     x_um = request_msg.plasmaX * 1e6
    #     y_um = request_msg.plasmaY * 1e6
    #     z_um = request_msg.plasmaZ * 1e6
    #
    #     self.logger.info(
    #         "[MOCK] SetPlasmaPositionSetpoint, "
    #         f"x={x_um} um, "
    #         f"y={y_um} um, "
    #         f"z={z_um} um"
    #     )
    #
    #     self.position_model.set_position(
    #         x_um,
    #         y_um,
    #         z_um
    #     )
    #
    #     return True, 0
    def SetPlasmaPositionSetpoint(
            self,
            request_msg,
            response_msg,
            attr=None
    ):
        """
        MOCK DC设置等离子体位置接口。

        PBOP算法计算得到新的目标位置后，通过该接口下发。
        DC接口单位为m，Mock内部统一保存为um。
        """
        self.logger.info(
            "[MOCK DEBUG] raw request: "
            f"x={request_msg.plasmaX}, "
            f"y={request_msg.plasmaY}, "
            f"z={request_msg.plasmaZ}"
        )

        x_um = request_msg.plasmaX * 1e6
        y_um = request_msg.plasmaY * 1e6
        z_um = request_msg.plasmaZ * 1e6

        self.logger.info(
            "[MOCK] SetPlasmaPositionSetpoint, "
            f"x={x_um:.2f} um, "
            f"y={y_um:.2f} um, "
            f"z={z_um:.2f} um"
        )

        # Mock环境认为位置模型能够执行该位置
        self.current_position = (
            x_um,
            y_um,
            z_um,
        )
        return True, 0

    def GetPlasmaPosition(
            self,
            request_msg,
            response_msg,
            attr=None
    ):
        """
        MOCK 获取等离子体位置量测值。
        request_msg.value 为采样次数 n_burst。Mock环境下认为实际位置已经到达最近一次下发的位置，
        回 n_burst 个相同的位置量测值。Mock内部 current_position 单位为 um；DC接口 position 中的位置单位为 m。
        """

        n_burst = request_msg.value

        if self.current_position is None:
            self.current_position = (
                self._rng.uniform(-200.0, 200.0),
                self._rng.uniform(-200.0, 200.0),
                self._rng.uniform(-200.0, 200.0),
            )

        x_um, y_um, z_um = self.current_position

        # Mock n_burst次量测
        for _ in range(n_burst):
            position = response_msg.position.add()

            # um -> m
            position.plasmaX = x_um / 1e6
            position.plasmaY = y_um / 1e6
            position.plasmaZ = z_um / 1e6

        self.logger.info(
            "[MOCK] GetPlasmaPosition success, "
            f"n_burst={n_burst}, "
            f"x={x_um:.2f} um, "
            f"y={y_um:.2f} um, "
            f"z={z_um:.2f} um"
        )

        return True, 0

    def GetPpoData(self, request_msg, response_msg, attr=None):
        """
        MOCK 获取等离子体位置相关执行单元量测信息
        """
        if self.logger is not None:
            self.logger.info("[MOCK] DCLGP GetPpoData called.")

        # 暂时不填充具体字段，保持 protobuf 默认值
        if self.logger is not None:
            self.logger.info(
                f"[MOCK] DCLGP GetPpoData success, response={response_msg}."
            )

        return True, 0

class MockDCSAPServiceTClient:
    """
    MOCK DCSAP ServiceT
    """

    def __init__(self, logger=None):
        self.logger = logger

    def GetPSDPosition(self, request_msg, response_msg, attr=None):
        n_burst = request_msg.value

        if self.logger is not None:
            self.logger.info(
                f"[MOCK] GetPSDPosition called, n_burst={n_burst}"
            )

        for _ in range(n_burst):
            psd_pos = response_msg.positions.add()
            psd_pos.x = 0.0
            psd_pos.y = 0.0

        if self.logger is not None:
            self.logger.info(
                f"[MOCK] GetPSDPosition success, count={len(response_msg.positions)}"
            )

        return True, 0

    def SetZeroPosition(self, request_msg, response_msg, attr=None):
        if self.logger is not None:
            self.logger.info("[MOCK] SetZeroPosition success")
        return True, 0

    def SaveZeroPosition(self, request_msg, response_msg, attr=None):
        if self.logger is not None:
            self.logger.info("[MOCK] SaveZeroPosition success")
        return True, 0

    def LoadZeroPosition(self, request_msg, response_msg, attr=None):
        if self.logger is not None:
            self.logger.info("[MOCK] LoadZeroPosition success")
        return True, 0

# ========================
# ED Energy Data
# ========================
@dataclass
class MockEDEnergy:
    """
    单个 Burst 的 ED 能量数据
    """
    hitmiss: bool
    energy: float
    pfEnergy: float

class DCStatusMockLib:
    """
    对真实 lib 做代理。

    MOCK:
        DCDXPStatusServiceClient
        ILIMPServiceClient
        RMPServiceAClient
        ILIMPServiceTClient
        RMPServiceTClient
        DCDXPServiceTClient

    其他 protobuf / RPC Client 继续使用真实 lib。
    """
    is_mock = True

    def __init__(self, real_lib, logger, *, seed=42, ed_time_model=None, pf_model=None):
        self._real_lib = real_lib
        self._logger = logger
        self._position_client = MockDCLGPServiceTClient(logger, seed)
        position_seed, ed_time_seed, pf_seed = np.random.SeedSequence(seed).spawn(3)
        self._position_rng = np.random.default_rng(position_seed)
        self._ed_time_model = TimeJitter(seed=ed_time_seed) if ed_time_model is None else ed_time_model
        self._pf_model = PFTimeJitter(seed=pf_seed) if pf_model is None else pf_model
        self._formula_fit = None if MODEL_MODE == "fitted" else build_formula_ed_model(MODEL_MODE)
        self._ed_key_index = 0
        # 模拟 KVS / DataPool
        # key -> List[MockEDEnergy]
        self._ed_data_cache = {}



    # ========================
    # DC status
    # ========================
    def DCDXPStatusServiceClient(self):
        return MockDCDXPStatusServiceClient(
            self._logger,
            self._real_lib.dcdxp_msg.READY
        )
    # ========================
    # IL status
    # ========================
    def ILIMPServiceClient(self):
        return MockILIMPServiceClient(
            self._logger,
            self._real_lib.ilimp_msg.READY
        )
    # ========================
    # REMA status
    # ========================
    def RMPServiceAClient(self):
        return MockRMPServiceAClient(
            self._logger,
            self._real_lib.rmpbf_msg.READY
        )

    # ========================
    # IL SetIlluminationMode
    # ========================

    def ILIMPServiceTClient(self):
        return MockILIMPServiceTClient(
            self._logger
        )

    # ========================
    # REMA CloseWindow
    # ========================

    def RMPServiceTClient(self):
        return MockRMPServiceTClient(
            self._logger
        )
    # ========================
    # # DCDXP exposure
    # ========================
    def DCDXPServiceTClient(self):
        return MockDCDXPServiceTClient(self._logger)

    # ========================
    # Plasma Position Setpoint
    # ========================
    def DCLGPServiceTClient(self):
        return self._position_client

    def DCSAPServiceTClient(self):
        return MockDCSAPServiceTClient(self.logger)
    # ========================
    # Plasma Position SetpointModel
    # ========================
    # def DCLGPServiceTClient(self):
    #     return MockDCLGPServiceTClient(
    #         self._logger,
    #         self._position_model
    #     )

    # ========================
    # ED Energy
    # ========================
    def get_ed_data_key(self, n_burst=1):
        """
        Mock 获取 ED 数据 DataPool key。

        固定生成一发 ED/PF 并缓存；n_burst 只保留旧调用签名，不再控制发数。
        """
        self._ed_key_index += 1
        key = f"mock_ed_{self._ed_key_index}"

        result = run_pipeline(
            center=self._position_client.current_position,
            random_seed=self._position_rng, fitter=self._formula_fit,
            ed_time_model=self._ed_time_model, pf_model=self._pf_model,
        )
        ed_energy_list = [
            MockEDEnergy(
                hitmiss=True,
                energy=result["ed"],
                pfEnergy=result["pf"],
            )
        ]

        self._ed_data_cache[key] = ed_energy_list

        self._logger.debug(
            f"Mock generate ED data, "
            f"n_burst: 1, "
            f"key: {key}, "
            f"ED: {result['ed']}, "
            f"PF: {result['pf']}."
        )

        return key

    def get_ed_data(self, key):
        """
        Mock 根据 DataPool key 获取 ED 能量数据。
        """
        if key not in self._ed_data_cache:
            raise KeyError(
                f"Mock ED data key not found: {key}"
            )

        ed_energy_list = self._ed_data_cache.pop(key)

        self._logger.debug(
            f"Mock get ED data, "
            f"key: {key}, "
            f"size: {len(ed_energy_list)}."
        )

        return ed_energy_list

    # ========================
    # PSD
    # ========================
    def DCSAPServiceTClient(self):
        return MockDCSAPServiceTClient()

    def __getattr__(self, name):
        """
        没有被 MOCK 的属性全部转发到真实 lib。
        """
        return getattr(self._real_lib, name)
