python main.py \
    --benchmark mmr-v,videoholmes,vrbench,vcrbench,videommmu,videomme \
    --mode step \
    --gpu_ids 0,1,2,3 \
    --num_processes 4 \
    --model_path Conan \
    --max_frames 16 \
    --max_steps 3 \
    --max_new_tokens 4000 \
    --temperature 1.0
    # --debug \
    # --debug_mode random \
    # --debug_num 4