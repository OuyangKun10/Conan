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
    """把 HH:MM:SS[.fff] 转成秒"""
    # if '.' in ts:
    #     h, m, s = ts.replace('.', ':').split(':')
    #     return int(h) * 3600 + int(m) * 60 + float(s) / 1000
    # else:
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)
def extract_timestamps(text: str) -> List[Tuple[str, str]]:
    """
    从字符串中提取所有 HH:MM:SS-HH:MM:SS 形式的时间戳对。
    返回：
      List[(start, end)]
    """
    # 正则：捕获两组完整时间
    pattern = re.compile(
        r'(?P<start>\d{1,2}:\d{2}:\d{2})'
        r'\s*-\s*'
        r'(?P<end>\d{1,2}:\d{2}:\d{2})'
    )
    # pattern = re.compile(
    #     r'(?P<start>\d{2}:\d{2}:\d{2}(?:[:\.]\d{3})?)'
    #     r'\s*-\s*'
    #     r'(?P<end>\d{2}:\d{2}:\d{2}(?:[:\.]\d{3})?)'
    # )
    return [(m.group('start'), m.group('end')) for m in pattern.finditer(text)]
def clips_to_frame_indices(
    video_path: str,
    timestamps: List[Tuple[str, str]],
    max_frames: int,
    history_frames_ts: List[str] # 修改参数名，更清晰表示是时间戳列表
) -> List[int]:
    """
    在全局帧预算 max_frames 内，将多段 clip 按时长加权分配帧数并均匀采样。
    抽取时跳过 history_frames_ts 对应帧，最后抽取的帧数量必须等于 max_frames，
    若少了可以从片段外临近抽取。
    """
    # 1. 获取总帧数及 fps
    vr, fps = load_video(video_path)
    total_frames = len(vr)

    # 将 history_frames_ts 转换为帧索引集合，方便查找
    history_frame_indices = set()
    for ts in history_frames_ts:
        sec = _to_seconds(ts)
        idx = int(np.round(sec * fps))
        if 0 <= idx < total_frames:
            history_frame_indices.add(idx)

    # 2. 计算每段 clip 的帧范围
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
        # 如果没有有效的 clip，尝试从整个视频中均匀采样 max_frames 帧，跳过历史帧
        all_possible_indices = sorted(list(set(range(total_frames)) - history_frame_indices))
        if len(all_possible_indices) <= max_frames:
            return all_possible_indices
        else:
            if max_frames <= 0:
                return []
            if max_frames == 1:
                return [all_possible_indices[0]] # 或者选择中间帧
            try:
                step = (len(all_possible_indices) - 1) / (max_frames - 1)
                sampled_indices = [all_possible_indices[int(round(i * step))] for i in range(max_frames)]
            except: sampled_indices=[]
            return sorted(set(sampled_indices))


    # 3. 按时长加权分配全局帧数 (考虑跳过历史帧的有效帧数)
    n_clips = len(clips)
    min_per_clip = 1 # 确保每个clip至少分配一帧
    remaining_budget_for_weighted_allocation = max_frames - n_clips * min_per_clip
    if remaining_budget_for_weighted_allocation < 0:
        remaining_budget_for_weighted_allocation = 0 # 保证至少 1 帧/clip

    # 计算每个 clip 内可用的非历史帧数量作为权重
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

    # 处理四舍五入后的余量，保证总和 = remaining_budget_for_weighted_allocation
    deficit = int(remaining_budget_for_weighted_allocation - quotas.sum())
    # 优先给有效帧数更多的clip分配余量
    sorted_indices_by_weight = np.argsort(weights_array)[::-1] # 降序
    for i in range(deficit):
        quotas[sorted_indices_by_weight[i % n_clips]] += 1


    # 4. 为每段 clip 均匀采样 (跳过历史帧)
    sampled_indices_within_clips = set() # 使用 set 去重
    for (start, end), quota_raw in zip(clips, quotas):
        quota = quota_raw + min_per_clip
        
        # 收集 clip 内所有非历史帧
        current_clip_available_frames = [i for i in range(start, end + 1) if i not in history_frame_indices]
        
        if not current_clip_available_frames:
            continue # 如果clip内没有可用的非历史帧，则跳过

        if len(current_clip_available_frames) <= quota:
            sampled_indices_within_clips.update(current_clip_available_frames)
        else:
            if quota <= 1:
                # 尽量选择 clip 开头附近的非历史帧
                sampled_indices_within_clips.add(current_clip_available_frames[0])
            else:
                # 均匀采样
                step = (len(current_clip_available_frames) - 1) / (quota - 1)
                for i in range(quota):
                    sampled_indices_within_clips.add(current_clip_available_frames[int(round(i * step))])
    
    final_indices = sorted(list(sampled_indices_within_clips))

    # 5. 确保最终帧数等于 max_frames
    if len(final_indices) < max_frames:
        needed_frames = max_frames - len(final_indices)
        
        # 收集所有非历史帧，并排除已经采样的帧
        all_non_history_frames = sorted(list(set(range(total_frames)) - history_frame_indices - set(final_indices)))

        if not all_non_history_frames:
            # 如果没有更多非历史帧可以补充，返回现有帧
            return final_indices
        
        # 从 all_non_history_frames 中选择最靠近已采样帧的帧进行补充
        # 简单策略：选择离现有帧最近的帧。这里我们采用更简单的策略，从头开始取needed_frames个
        # 或者可以从所有非历史帧中均匀补充，或者随机补充
        
        # 策略1: 从所有非历史且未采样的帧中，按索引顺序补充
        supplementary_frames = []
        if needed_frames > 0:
            if len(all_non_history_frames) <= needed_frames:
                supplementary_frames = all_non_history_frames
            else:
                # 均匀补充，确保分布均匀
                if needed_frames == 1:
                    supplementary_frames = [all_non_history_frames[0]]
                else:
                    step_supplement = (len(all_non_history_frames) - 1) / (needed_frames - 1)
                    supplementary_frames = [all_non_history_frames[int(round(i * step_supplement))] for i in range(needed_frames)]

        final_indices.extend(supplementary_frames)
        final_indices = sorted(list(set(final_indices))) # 再次去重并排序

        # 如果补充后仍然不足 max_frames，这可能是由于历史帧过多或者视频太短导致
        # 此时只能返回所有可用的非历史帧，并可能重复帧（为了达到max_frames）
        if len(final_indices) < max_frames:
            # 如果实在凑不够，我们只能重复已有的帧来达到 max_frames
            # 优先重复那些在原始 clips 内部被采样的帧
            if not final_indices: # 极端情况，如果 final_indices 为空
                return []
            
            # 简单重复策略：循环复制已有的帧直到达到 max_frames
            extended_indices = list(final_indices)
            while len(extended_indices) < max_frames:
                extended_indices.extend(final_indices)
            return sorted(extended_indices[:max_frames])

    elif len(final_indices) > max_frames:
        # 如果超出 max_frames，需要删除一些帧
        # 策略：均匀删除，或者优先删除那些不是在 clip 内部采样的帧 (这里简单均匀删除)
        # 为了均匀删除，我们可以重新进行一次均匀采样，确保数量为 max_frames
        if max_frames <= 0:
            return []
        if max_frames == 1:
            return [final_indices[0]] # 或者选择中间帧
        step_oversample = (len(final_indices) - 1) / (max_frames - 1)
        print(f"step_oversample: ",step_oversample)
        print(f"max_frames: ",max_frames)
        resampled_indices=[]
        for i in range(max_frames):
            print(f"frame_indices: {i * step_oversample}, round: {round(i * step_oversample)}")
            resampled_indices.append(final_indices[round(i * step_oversample)])
        # resampled_indices = [final_indices[round(i * step_oversample)] for i in range(max_frames)]
        return sorted(list(set(resampled_indices))) # 确保去重并排序


    return final_indices
def identify_clip(response):
    if '<clip>' in response:
        clip_match = re.search(r"<clip>(.*?)</clip>", response, re.DOTALL)
        if clip_match:                       # 匹配成功
            timestamps = extract_timestamps(clip_match.group(1))
        else:
            timestamps = []                  # 没有匹配到
    else:
        timestamps = []                      # 连 <clip> 都没有
    valid = []
    for s, e in timestamps:
        try:
            s_sec, e_sec = _to_seconds(s), _to_seconds(e)
            if s_sec < e_sec:                       # 起始 < 终止
                valid.append((s, e))
        except ValueError:
            continue
    return valid
def identify_replay(response):
    # match = re.search(r"<replay>(.*?)</replay>", response, re.DOTALL)
    # if not match:
    #     return False
    # response=match.group(1).strip() 
    # if 'true' in response.lower():
    #     return True
    # else:
    #     return False
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

    if len(s.split()) > 10 and not re.search("[ABCDEFGHIJKL]", s):
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
               correct: bool,
               rounds: int):
    """逐条样本日志追加写入"""
    rec = {
        "rank": rank,
        "gpu_ids": gpu_ids,
        "video": sample.get("video_path",None),
        "duration": sample.get("duration", None),   # 秒
        "question": sample["question"],
        "gt": str(sample["answer"]),
        "pred": pred,
        "correct": correct,
        "rounds": rounds,
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
# ---------- 统计 ----------
def count_total_input_frames(sample: dict) -> int:
    """
    统计该样本所有 user 轮次里给出的 image 总数
    """
    frames = 0
    for msg in sample.get("messages", []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        frames += sum(1 for c in content if isinstance(c, dict) and c.get("type") == "image")
    return frames


def avg_input_frames(data: List) -> float:
    total_frames = 0
    total_samples = 0
    for sample in data:
        total_frames += count_total_input_frames(sample)
        total_samples += 1
    return total_frames / total_samples if total_samples else 0.0