export WANDB_PROJECT=Conan-7B-sft-stage3
export WANDB_API_KEY=YOUR_API_KEY
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Conan-7B-sft-stage2 \
    --train_type full \
    --dataset Conan-CoT-60k.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --freeze_vit true \
    --freeze_aligner false \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --attn_impl flash_attn \
    --eval_steps 100 \
    --save_steps 1000 \
    --save_total_limit 50 \
    --logging_steps 5 \
    --max_length 4000 \
    --output_dir output/Conan-7B-sft \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_name Conan-7B-sft
