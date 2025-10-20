#!/usr/bin/env python3
"""
main.py
"""
import argparse
import json
import os
import random
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict
import utils
import video_io  
from tqdm import tqdm
import model as M
BENCHMARK_MODULES = {
    "videomme": __import__("videomme"),
    "mmr-v": __import__("mmr-v"),
    "videoholmes": __import__("videoholmes"),
    "glimpse": __import__("glimpse"),
    "vrbench": __import__("vrbench"),
    "vcrbench": __import__("vcrbench"),
    "videommmu": __import__("videommmu"),
    "longvideobench": __import__("longvideobench"),
    "mlvu": __import__("mlvu"),
    "lvbench": __import__("lvbench"),
    "longvideoreason": __import__("longvideoreason"),
    "humanpcr": __import__("humanpcr"),
}
MODEL_MODULES = ["Qwen2_5_VL","KimiVL"]
def worker(rank: int,
           gpu_ids: List[int],
           model_type: str,
           benchmark: str,
           tasks: List[Dict],
           model_path: str,
           mode: str,
           max_frames: int,
           max_steps: int,
           max_new_tokens: int,
           temperature: float,
           run_dir: Path):

    bench = BENCHMARK_MODULES[benchmark]
    if model_type not in MODEL_MODULES:
        print("Please specify your model_type within Qwen2.5-VL or Kimi-VL using --model_type")
    if model_type=="Qwen2_5_VL":
        model=M.Qwen2_5_VL(model_path=model_path, gpu_ids=gpu_ids, max_new_tokens=max_new_tokens,temperature=temperature)
    elif model_type=="KimiVL":
        model=M.KimiVL(model_path=model_path, gpu_ids=gpu_ids, max_new_tokens=max_new_tokens,temperature=temperature)
    results = []
    for ex in tqdm(tasks,desc=f"rank {rank}"):
        pred, prompt, messages, ok, rounds = bench.evaluate_example(model, ex, mode=mode, max_frames=max_frames,max_steps=max_steps)
        results.append(bench.make_result_record(ex, prompt, messages, pred, ok, rounds))
        if pred=="None":
            utils.log_sample(run_dir, rank, gpu_ids, ex, pred, ok, rounds)
    utils.save_rank_result(run_dir, rank, results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--mode", choices=["step", "uniform","cot"], required=True)
    parser.add_argument("--gpu_ids", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_type", choices=["Qwen2_5_VL","KimiVL"], type=str, default="Qwen2_5_VL")

    # ===== debug 参数 =====
    parser.add_argument("--debug", action="store_true",
                        help="enable debug mode")
    parser.add_argument("--debug_mode", choices=["random", "default"],
                        default="default",
                        help="how to pick debug samples")
    parser.add_argument("--debug_num", type=int, default=10,
                        help="number of debug samples")

    args = parser.parse_args()
    # 加载全部任务
    benchmarks=args.benchmark.split(',')
    if "Kimi-VL" in args.model_path:
        args.model_type="KimiVL"
    elif "Qwen2.5-VL" in args.model_path:
        args.model_type="Qwen2_5_VL"
    print(f"Benchmarks:{benchmarks}")
    print(f"Model type:{args.model_type}")
    for bench_i in benchmarks:
        print(f"Evaluate {bench_i}")
        bench = BENCHMARK_MODULES[bench_i]
        qa = bench.load_data() 

        # ===== debug 模式下采样 =====
        if args.debug:
            if args.debug_mode == "random":
                qa = random.sample(qa, min(args.debug_num, len(qa)))
            else:
                qa = qa[:min(args.debug_num, len(qa))]
            print(f"[DEBUG] running {len(qa)} samples ({args.debug_mode})")

        gpu_ids = utils.split_gpu_ids(args.gpu_ids)
        num_procs = args.num_processes
        assert num_procs >= 1 and len(gpu_ids) >= 1

    
        if num_procs == 1:
            gpu_per_proc = [gpu_ids]
        else:
            if len(gpu_ids) == num_procs:
                gpu_per_proc = [[g] for g in gpu_ids]
            elif len(gpu_ids) > num_procs:
                k = len(gpu_ids) // num_procs
                gpu_per_proc = [gpu_ids[i*k:(i+1)*k] for i in range(num_procs)]
            else:
                gpu_per_proc = [[gpu_ids[i % len(gpu_ids)]] for i in range(num_procs)]

     
        model_name = Path(args.model_path).name
        if args.model_name!="":
            model_name=f"{args.model_name}/{model_name}"
        run_dir = utils.make_run_dir(bench_i, model_name)
        utils.save_run_meta(run_dir, vars(args))

 
        mp.set_start_method("spawn", force=True)
        processes = []
        for rank, sub_tasks in enumerate(utils.split_tasks(qa, num_procs)):
            print(f"rank {rank} uses gpu {gpu_per_proc[rank]}")
            p = mp.Process(target=worker,
                        args=(rank,
                                gpu_per_proc[rank],
                                args.model_type,
                                bench_i,
                                sub_tasks,
                                args.model_path,
                                args.mode,
                                args.max_frames,
                                args.max_steps,
                                args.max_new_tokens,
                                args.temperature,
                                run_dir))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        all_results = []
        for r in range(num_procs):
            with open(run_dir / f"results_rank{r}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if line.strip():                    
                        all_results.append(json.loads(line))
        utils.save_final_results(run_dir, all_results)
        print(f"All done. Results saved to {run_dir.resolve()}")
        res_summary=bench.result_statistics(run_dir)
        print("summary:",res_summary)
if __name__ == "__main__":
    main()