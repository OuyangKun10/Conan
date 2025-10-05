"""
utils.py 
"""
import json
import os
from pathlib import Path
import datetime
from typing import List, Dict, Any, Tuple
import pandas as pd
import re
from video_io import load_video
import numpy as np
# ---------- 评测工具 ----------
def _to_seconds(ts: str) -> float:
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)
def extract_timestamps(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r'(?P<start>\d{1,2}:\d{2}:\d{2})'
        r'\s*-\s*'
        r'(?P<end>\d{1,2}:\d{2}:\d{2})'
    )
    return [(m.group('start'), m.group('end')) for m in pattern.finditer(text)]
def clips_to_frame_indices(
    video_path: str,
    timestamps: List[Tuple[str, str]],
    max_frames: int,
    history_frames_ts: List[str] 
) -> List[int]:
    vr, fps = load_video(video_path)
    total_frames = len(vr)
    history_frame_indices = set()
    for ts in history_frames_ts:
        sec = _to_seconds(ts)
        idx = int(np.round(sec * fps))
        if 0 <= idx < total_frames:
            history_frame_indices.add(idx)

    clips = []
    total_duration_sec = 0.0
    for start_ts, end_ts in timestamps:
        start_sec = _to_seconds(start_ts)
        end_sec = _to_seconds(end_ts)
        start_idx = max(0, int(np.round(start_sec * fps)))
        end_idx = min(total_frames - 1, int(np.round(end_sec * fps)))
        if start_idx < end_idx:
            clips.append((start_idx, end_idx))
            total_duration_sec += (end_sec - start_sec)

    if not clips:
        all_possible_indices = sorted(list(set(range(total_frames)) - history_frame_indices))
        if len(all_possible_indices) <= max_frames:
            return all_possible_indices
        else:
            if max_frames <= 0:
                return []
            if max_frames == 1:
                return [all_possible_indices[0]] 
            try:
                step = (len(all_possible_indices) - 1) / (max_frames - 1)
                sampled_indices = [all_possible_indices[int(round(i * step))] for i in range(max_frames)]
            except: sampled_indices=[]
            return sorted(set(sampled_indices))



    n_clips = len(clips)
    min_per_clip = 1 
    remaining_budget_for_weighted_allocation = max_frames - n_clips * min_per_clip
    if remaining_budget_for_weighted_allocation < 0:
        remaining_budget_for_weighted_allocation = 0 
    effective_weights = []
    for start, end in clips:
        available_frames_in_clip = 0
        for i in range(start, end + 1):
            if i not in history_frame_indices:
                available_frames_in_clip += 1
        effective_weights.append(available_frames_in_clip)

    weights_array = np.array(effective_weights, dtype=float)

    quotas = np.zeros_like(weights_array, dtype=int)
    if weights_array.sum() > 0:
        quotas = (weights_array / weights_array.sum() * remaining_budget_for_weighted_allocation).astype(int)


    deficit = int(remaining_budget_for_weighted_allocation - quotas.sum())
    sorted_indices_by_weight = np.argsort(weights_array)[::-1] 
    for i in range(deficit):
        quotas[sorted_indices_by_weight[i % n_clips]] += 1

    sampled_indices_within_clips = set() 
    for (start, end), quota_raw in zip(clips, quotas):
        quota = quota_raw + min_per_clip
        
        current_clip_available_frames = [i for i in range(start, end + 1) if i not in history_frame_indices]
        
        if not current_clip_available_frames:
            continue 

        if len(current_clip_available_frames) <= quota:
            sampled_indices_within_clips.update(current_clip_available_frames)
        else:
            if quota <= 1:
                sampled_indices_within_clips.add(current_clip_available_frames[0])
            else:
                step = (len(current_clip_available_frames) - 1) / (quota - 1)
                for i in range(quota):
                    sampled_indices_within_clips.add(current_clip_available_frames[int(round(i * step))])
    
    final_indices = sorted(list(sampled_indices_within_clips))

    if len(final_indices) < max_frames:
        needed_frames = max_frames - len(final_indices)
        
        all_non_history_frames = sorted(list(set(range(total_frames)) - history_frame_indices - set(final_indices)))

        if not all_non_history_frames:
            return final_indices
        supplementary_frames = []
        if needed_frames > 0:
            if len(all_non_history_frames) <= needed_frames:
                supplementary_frames = all_non_history_frames
            else:
                if needed_frames == 1:
                    supplementary_frames = [all_non_history_frames[0]]
                else:
                    step_supplement = (len(all_non_history_frames) - 1) / (needed_frames - 1)
                    supplementary_frames = [all_non_history_frames[int(round(i * step_supplement))] for i in range(needed_frames)]

        final_indices.extend(supplementary_frames)
        final_indices = sorted(list(set(final_indices))) 

        if len(final_indices) < max_frames:
            if not final_indices:
                return []

            extended_indices = list(final_indices)
            while len(extended_indices) < max_frames:
                extended_indices.extend(final_indices)
            return sorted(extended_indices[:max_frames])

    elif len(final_indices) > max_frames:

        if max_frames <= 0:
            return []
        if max_frames == 1:
            return [final_indices[0]] 
        step_oversample = (len(final_indices) - 1) / (max_frames - 1)
        print(f"step_oversample: ",step_oversample)
        print(f"max_frames: ",max_frames)
        resampled_indices=[]
        for i in range(max_frames):
            print(f"frame_indices: {i * step_oversample}, round: {round(i * step_oversample)}")
            resampled_indices.append(final_indices[round(i * step_oversample)])
        return sorted(list(set(resampled_indices))) 


    return final_indices
def identify_clip(response):
    if '<clip>' in response:
        clip_match = re.search(r"<clip>(.*?)</clip>", response, re.DOTALL)
        if clip_match:                      
            timestamps = extract_timestamps(clip_match.group(1))
        else:
            timestamps = []                 
    else:
        timestamps = []                     
    valid = []
    for s, e in timestamps:
        try:
            s_sec, e_sec = _to_seconds(s), _to_seconds(e)
            if s_sec < e_sec:                     
                valid.append((s, e))
        except ValueError:
            continue
    return valid
def identify_replay(response):
    match_clip = re.search(r"<clip>(.*?)</clip>", response, re.DOTALL)
    match_ans = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    timestamps=identify_clip(response)
    if match_clip and len(timestamps)>0:
        return True,timestamps
    else:
        return False,None
def extract_answer(text_with_tags):
    match = re.search(r"<answer>(.*?)</answer>", text_with_tags, re.DOTALL)
    if match:
        return match.group(1).strip()  
    else:
        return None
def extract_characters(s):
    if s is None:
        return None
    s = s.strip()
    answer_prefixes = [
        "The best answer is", "The correct answer is", "The answer is", "The answer",
        "The best option is", "The correct option is", "Best answer:", "Best option:",
        "Answer:", "Option:", "The correct answer", "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCDEFGHIJKL]', s)
    if matches is None:
        return None
    return matches[0]
def normalize_number(num_str):
    try:
        num_str = num_str.replace(',', '')
        return float(num_str)
    except Exception as e:
        return None
        
def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)
    
    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
    
    thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
    
    conditions = rel_error < (1 - thresholds)  
    mra = conditions.float().mean()  
    return mra.item()
# ---------- 路径 ----------
def make_run_dir(benchmark: str, model_name: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("res") / benchmark / model_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def split_gpu_ids(gpu_ids_str: str) -> List[int]:
    return [int(x) for x in gpu_ids_str.split(",") if x.strip()]
def filter_mmrv(tasks):
    filter_task=[]
    for item in tasks:
        if "Burnout.f137" in item['video_path']:
            filter_task.append(item)
    return filter_task

def split_tasks(tasks: List[Any], num_procs: int):
    import random
    # tasks=filter_mmrv(tasks)
    random.shuffle(tasks)
    chunk = (len(tasks) + num_procs - 1) // num_procs
    for i in range(num_procs):
        yield tasks[i * chunk : (i + 1) * chunk]
# ---------------- 日志 ----------------
def save_run_meta(run_dir: Path, meta: Dict):
    """保存一次运行的全局参数"""
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def log_sample(run_dir: Path,
               rank: int,
               gpu_ids: List[int],
               sample: Dict,
               pred: str,
               correct: bool):
    """逐条样本日志追加写入"""
    rec = {
        "rank": rank,
        "gpu_ids": gpu_ids,
        "video": sample["video_path"],
        "duration": sample.get("duration", None),   # 秒
        "question": sample["question"],
        "gt": str(sample["answer"]),
        "pred": pred,
        "correct": correct
    }
    with open(run_dir / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
# ---------- 万能读取 ----------
def load_qa(path: str) -> List[Dict[str, Any]]:
    """
    自动识别后缀并读取：
        .json   -> 单 json 对象
        .jsonl  -> 每行一个 json
        .parquet-> pandas.read_parquet
        .tsv    -> pandas.read_csv(sep='\t')
    返回统一 list[dict]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    if suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported file type: {suffix}")
def save_rank_result(run_dir: Path, rank: int, results: List[Dict[str, Any]]):
    with open(run_dir / f"results_rank{rank}.jsonl", "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def save_final_results(run_dir: Path, all_results: List[Dict[str, Any]]):
    with open(run_dir / "results_all.jsonl", "w", encoding="utf-8") as f:
        for rec in all_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
