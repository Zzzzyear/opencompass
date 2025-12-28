from opencompass.models import VLLMwithChatTemplate

models = [
    # 1. Qwen3-1.7B
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-1.7b-vllm',
        path='/data/zhaoqn/models/Qwen/Qwen3-1.7B',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=16384,
        max_seq_len=32768,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        # 使用内置后处理去除 <think>
        pred_postprocessor=dict(type='extract-non-reasoning-content'),
        generation_kwargs=dict(temperature=0.6, top_p=0.95)
    ),
    # 2. Qwen3-4B
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-4b-vllm',
        path='/data/zhaoqn/models/Qwen/Qwen3-4B',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=16384,
        max_seq_len=32768,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type='extract-non-reasoning-content'),
        generation_kwargs=dict(temperature=0.6, top_p=0.95)
    ),
    # 3. Qwen3-8B
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-8b-instruct-vllm',
        path='/data/zhaoqn/models/Qwen/Qwen3-8B',
        model_kwargs=dict(tensor_parallel_size=1),
        max_out_len=16384,
        max_seq_len=32768,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type='extract-non-reasoning-content'),
        generation_kwargs=dict(temperature=0.6, top_p=0.95)
    ),
]