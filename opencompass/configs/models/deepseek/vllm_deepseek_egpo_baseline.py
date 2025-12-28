from opencompass.models import VLLMwithChatTemplate

models = [
    # 1. DeepSeek-Math-7B-RL
    dict(
        type=VLLMwithChatTemplate,
        abbr='deepseek-math-7b-rl-vllm',
        path='/data/zhaoqn/models/DeepSeek/deepseek-math-7b-rl',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(temperature=0.6, top_p=0.95)
    ),
    # 2. DeepSeek-Math-7B-Instruct
    dict(
        type=VLLMwithChatTemplate,
        abbr='deepseek-math-7b-instruct-vllm',
        path='/data/zhaoqn/models/DeepSeek/deepseek-math-7b-instruct',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        generation_kwargs=dict(temperature=0.6, top_p=0.95)
    ),
    # 3. DeepSeek-R1-Distill-Qwen-1.5B (使用内置后处理)
    dict(
        type=VLLMwithChatTemplate,
        abbr='deepseek-r1-distill-qwen-1.5b-vllm',
        path='/data/zhaoqn/models/DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=16384, 
        max_seq_len=32768,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        # 使用 opencompass/utils/text_postprocessors.py 中已注册的模块
        pred_postprocessor=dict(type='extract-non-reasoning-content'), 
        generation_kwargs=dict(temperature=0.6, top_p=0.95)
    ),
]