NFRAMES=16 MAX_PIXELS=401408 VIDEO_MAX_PIXELS=401408 \
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model Conan-SFT-7B \
    --vllm_use_async_engine true \
    --external_plugins ms-swift/examples/train/grpo/plugin/video_clip_scheduler.py \
    --multi_turn_scheduler VideoClipScheduler \
    --vllm_max_model_len 32768 \
    --vllm_gpu_memory_utilization 0.8 \
    --max_length 16384 \
    --loss_scale last_round \
    --model_type qwen2_5_vl \
    --max_turns 6 \
    --port 8000