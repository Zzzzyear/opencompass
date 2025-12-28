from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import CodeEvaluator, MathEvaluator
from opencompass.datasets import JsonlDataset

# 1. 引入官方已有的标准配置 (Math-500, AIME24, HumanEval, LCB, GPQA, ARC)
with read_base():
    from .math.math_prm800k_500_gen import math_datasets as math500_datasets
    from .aime2024.aime2024_gen import aime2024_datasets
    from .humaneval.humaneval_gen import humaneval_datasets
    from .livecodebench.livecodebench_gen import LCB_datasets as livecodebench_datasets
    from .gpqa.gpqa_gen import gpqa_datasets
    from .ARC_c.ARC_c_gen import ARC_c_datasets
    from .ARC_e.ARC_e_gen import ARC_e_datasets

# =========================================================
# 2. 自定义 AIME 2025 配置 (Rule-based, 本地数据)
# =========================================================
aime2025_datasets = [
    dict(
        abbr='aime2025',
        type=dict(
            type=JsonlDataset,
            # 请确保此路径指向您的真实 AIME 2025 jsonl 文件
            path='/data/zhaoqn/workspace/EGPO/datasets/raw/AIME-2025/aime2025-test.jsonl',
            reader_cfg=dict(
                input_key='question', # 请确认 jsonl 中的问题字段名
                output_key='answer',  # 请确认 jsonl 中的答案字段名
            )
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
                        dict(role='BOT', prompt=''),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            # 使用基于规则的数学评测器，提取 \\boxed{} 中的内容与 GT 比对
            evaluator=dict(type=MathEvaluator, version='v1'), 
            pred_role='BOT',
        )
    )
]

# =========================================================
# 3. 自定义 LeetCode 配置 (Rule-based, 本地数据)
# =========================================================
leetcode_datasets = [
    dict(
        abbr='leetcode',
        type=dict(
            type=JsonlDataset,
            path='/data/zhaoqn/workspace/EGPO/datasets/raw/LeetCodeDataset/LeetCodeDataset-test.jsonl',
            reader_cfg=dict(
                input_key='prompt', 
                output_key='test',  
            )
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt=''),
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            # LeetCode 也是代码任务，复用 CodeEvaluator
            evaluator=dict(type=CodeEvaluator, evaluator_type='leetcode'),
            pred_role='BOT',
        )
    )
]

# =========================================================
# 4. 汇总所有数据集
# =========================================================
datasets = (
    math500_datasets + 
    aime2024_datasets + 
    aime2025_datasets +   # 使用上面自定义的配置
    humaneval_datasets + 
    livecodebench_datasets + 
    leetcode_datasets +   # 使用上面自定义的配置
    gpqa_datasets +
    ARC_c_datasets + 
    ARC_e_datasets
)