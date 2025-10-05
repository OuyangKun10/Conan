"""
videomme.py 
"""
import os
import re
from typing import Tuple
import video_io
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from utils import *
from prompt_temp import *
# ========== 1. 路径配置 ==========
QA_FNAME   = "Video-MME/videomme/test-00000-of-00001.parquet"
VIDEO_DIR  = "Video-MME/data"   
PROMPT_TEMP=""""
{question}
{options}
Please answer the question with an optoin letter.
The answer is:
"""
def get_paths() -> Tuple[str, str]:
    return  QA_FNAME, VIDEO_DIR
def load_data() -> list[dict]:
    """
    统一字段后返回任务列表
    字段:
        video_name  : str   原始文件名
        video_path  : str   完整绝对路径
        question    : str
        answer      : str   ground truth
        duration    : float 秒(可选)
    """
    qa_path, video_dir = get_paths()
    raw = load_qa(qa_path)     
    out = []
    for item in raw:
        video_id=item["videoID"]
        out.append({
            "video_name": video_id,           
            "video_path": os.path.join(video_dir, f"{video_id}.mp4"),
            "question":   item["question"],
            "options":    item["options"],
            "answer":     str(item["answer"]),
            "duration":   item.get("duration")
        })
    return out
# ========== 2. prompt & 单条评测 ==========
def build_prompt(question: str, mode: str, options: List[str], init_flag: bool,final_flag: bool) -> str:
    if mode == "uniform":
        return PROMPT_TEMP.format(
            question=question,
            options=options,
        )
    elif mode == "step":
        if init_flag:
            return Prompt_temp_init_mc.format(
            question=question,
            options=options,
        )
        elif final_flag:
            return Prompt_temp_final_mc.format(
            question=question,
            options=options,
        )
        else:
            return Prompt_temp_round_mc.format(
            question=question,
            options=options,
        )
def construct_message(prompt,frames,timestamps=[],insert_timestamp=False):
    content=[]
    if insert_timestamp:
        assert len(frames) == len(timestamps)
        for ts, frame in zip(timestamps,frames):
            content.append({"type": "text", "text": ts})
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": prompt})
    else:
        content.append({"type": "video", "video": frames})
        content.append({"type": "text","text": prompt})
    message = [
            {
                "role": "user",
                "content": content,
            }
        ]
    return message

def eval_gt(response,gt):
    if "<answer>" in response:
        response=extract_answer(response)
    if response == None:
        return 0
    response = extract_characters(response)
    if response == None:
        return 0
    else: return response.lower()==gt.lower()
def evaluate_example(model, example: dict, mode: str, max_frames: int, max_steps=1) -> Tuple[str, bool]:
    video_path = example["video_path"]
    question   = example["question"]
    gt         = str(example["answer"]).strip()
    options    = example["options"]
    if mode == "uniform":
        vr, fps = video_io.load_video(video_path)
        idxs = video_io.uniform_sample_idx(len(vr), max_frames)
        frames = video_io.save_video_frames(video_path, idxs,"frames/videomme")
        imgs, _ = zip(*frames)
        prompt = build_prompt(question, mode, options,False,False) 
        message=construct_message(prompt,list(imgs))
        response, _ = model.chat(message)
        pred = response.strip()
        return pred, prompt, message, eval_gt(pred, gt)

    # step 模式
    
    history = []
    history_frames = []
    step_idx=0
    while True:
        if step_idx == 0 or not history_frames:
            prompt = build_prompt(question, mode, options,True,False) 
            vr, fps = video_io.load_video(video_path)
            idxs = video_io.uniform_sample_idx(len(vr), max_frames)
            frames = video_io.save_video_frames(video_path, idxs, "frames/videomme")
        elif step_idx == 2:
            prompt = build_prompt(question, mode, options,False,True)
        else:
            prompt = build_prompt(question, mode, options,False,False)
        imgs, ts = zip(*frames)
        history_frames = history_frames+list(ts)
        message=construct_message(prompt,list(imgs),ts,True)
        response, history = model.chat(message,history=history)

        # replay 逻辑
        m,timestamps = identify_replay(response)
        if m:
            replay_idxs = clips_to_frame_indices(video_path, timestamps, 8, history_frames)
            if replay_idxs:
                frames = video_io.save_video_frames(video_path, replay_idxs, "frames/videomme")
            else:break
        else: break
        step_idx=step_idx+1
        if step_idx>=max_steps:break

    pred = response.strip()
    return pred, prompt, history, eval_gt(pred, gt)
def make_result_record(ex: dict, prompt: str, messages: List[dict], pred: str, correct: bool) -> dict:
    """
    构造需要保存的字段，benchmark 可自由增删
    """
    return {
        "video": ex["video_path"],
        "duration": ex.get("duration"),   
        "question": ex["question"],
        "prompt": prompt,
        "messages": messages,
        "gt": str(ex["answer"]),
        "pred": pred,
        "correct": correct
    }
def result_statistics(run_dir: Path) -> Dict[str, Any]:
    """
    读取所有 results_all.jsonl 并返回统计字典
    结构：
        overall_accuracy
        total_samples
        bucket_accuracy
    """
    all_results = []
    with open(run_dir / "results_all.jsonl", encoding="utf-8") as f:
        for line in f:
            all_results.append(json.loads(line))
    failed_res=[]
    try:
        with open(run_dir / "log.jsonl", encoding="utf-8") as f:
            for line in f:
                failed_res.append(json.loads(line))
    except:
        print("No generation failed samples")
    bad_counts={}
    total = len(all_results)
    overall_acc = np.mean([r["correct"] for r in all_results])

    # 按 duration 分组
    buckets = {"short": [], "medium": [], "long": []}
    no_response=0
    for r in all_results:
        duration = r.get("duration", "unknown")
        response=r['pred']
        if "<answer>" in response:
            response=extract_answer(response)
        if response!='None':
            response = extract_characters(response)
            if response is None:
                no_response=no_response+1
        if duration in buckets:
            buckets[duration].append(r)
    bad_counts={"inference error":len(failed_res),"retrieval error":no_response}
    bucket_acc = {k: (np.mean([r["correct"] for r in v]) if v else None)
                  for k, v in buckets.items()}

    summary = {
        "overall_accuracy": float(overall_acc),
        "total_samples": total,
        "bad_counts": bad_counts,
        "bucket_accuracy": bucket_acc
    }

    # 同时保存 summary.json
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary