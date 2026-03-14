# configs/ablation_config.py
import os

class AblationConfig:
    # 🎯 【唯一控制中枢】在这里修改你想跑的实验编号：
    # 可选值: 'A', 'B', 'C1', 'C2', 'C3', 'D', 'final'
    EXPERIMENT_ID = "final" 

    # ==========================================
    # 以下为根据 EXPERIMENT_ID 自动推导的组件开关，无需手动修改
    # ==========================================
    use_semantic_stream = True       # 是否启用 CLIP 语义流
    use_artifact_stream = True       # 是否启用 SRM+CNN 伪影流
    active_srm_filters = [1, 2, 3]   # 启用的 SRM 滤波器编号
    fusion_strategy = "mome"         # 融合策略: "none", "mlp", "mome"

    @classmethod
    def apply_config(cls):
        if cls.EXPERIMENT_ID == "A":
            cls.use_artifact_stream = False
            cls.fusion_strategy = "none"
        elif cls.EXPERIMENT_ID == "B":
            cls.use_semantic_stream = False
            cls.fusion_strategy = "none"
        elif cls.EXPERIMENT_ID == "C1":
            cls.active_srm_filters = [1]
        elif cls.EXPERIMENT_ID == "C2":
            cls.active_srm_filters = [2]
        elif cls.EXPERIMENT_ID == "C3":
            cls.active_srm_filters = [3]
        elif cls.EXPERIMENT_ID == "D":
            cls.fusion_strategy = "mlp"
        elif cls.EXPERIMENT_ID == "final":
            pass # final 保持所有组件默认全部开启