export WANDB_PROJECT=Conan-7b
export WANDB_API_KEY=YOUR_API_KEY
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
NPROC_PER_NODE=6 \
swift rlhf \
    --rlhf_type grpo \
    --model Conan-SFT-7B \
    --train_type full \
    --dataset Conan-RLVR-31k.json \
    --use_vllm true \
    --save_only_model true \
    --vllm_mode server \
    --vllm_server_host 0.0.0.0 \
    --vllm_server_port 8000 \
    --vllm_server_pass_dataset true \
    --vllm_gpu_memory_utilization 0.8 \
    --torch_dtype bfloat16 \
    --system 'You are a helpful assistant.' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --save_steps 200 \
    --save_total_limit 100 \
    --logging_steps 5 \
    --report_to wandb \
    --output_dir output/Conan-7b \
    --gradient_accumulation_steps 4 \
    --deepspeed zero3 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 16384 \
    --max_length 16384 \
    --external_plugins \
    ms-swift/examples/train/grpo/plugin/plugin.py \
    ms-swift/examples/train/grpo/plugin/video_clip_scheduler.py \
    --reward_funcs freeformaccuracy mcaccuracy  multiturnformat \
    --attn_impl flash_attn \
    --num_generations 8 \
    --sleep_level 1 \
    --temperature 1.0 \
    --top_p 0.85 \
    --log_level info \
    --model_type qwen2_5_vl \
    --log_completions true
