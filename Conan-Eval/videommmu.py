"""
videommmu.py 
"""
import os, glob, json, pandas as pd
from os import path as osp
import re
import video_io
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from utils import *
from prompt_temp import *
# ========== 1. 路径配置 ==========
PROMPT_TEMP_mc=""""
{question}
{options}
Please only answer the question with an optoin letter (e.g., A, B, C, D, etc.).
The answer is:
"""
PROMPT_TEMP_num=""""
{question}
Please only answer the question with the numerical value (e.g., 42 or 3.14).
The answer is:
"""
ROOT_DIR  = "VideoMMMU" 
QA_FNAME  = "VideoMMMU/eval_videommmu.json"
def get_paths() -> Tuple[str, str]:
    """
    返回 (qa_path, video_root_path)
    方便其它 benchmark 复用 main.py
    """
    return  ROOT_DIR,QA_FNAME
def load_data() -> list[dict]:
    raw = load_qa(QA_FNAME)
    out = []    
    for item in raw:
        video_id = osp.basename(item["path"])                     
        video_path = glob.glob(osp.join(ROOT_DIR, f"*/{video_id}"))[0]
        if item["problem_type"]=="multiple choice":
            options_str ="\n".join(item["options"])
        else: options_str=None
        out.append({
            "Question ID": item['problem_id'],
            "video_id": video_id,
            "video_path": video_path,
            "question": item["problem"],             
            "options": options_str,
            "answer": extract_answer(item["solution"]),
            "qa_type": item["problem_type"],
            "task": item["problem_type"],
        })
    return out
# ========== 2. prompt & 单条评测 ==========
def build_prompt(question: str, qa_type:str, mode: str, options: List[str], init_flag: bool,final_flag: bool) -> str:
    type_fix=TYPE_TEMPLATE[qa_type]
    if mode == "uniform":
        if qa_type=='multiple choice':
            return PROMPT_TEMP_mc.format(
                question=question,
                options=options,
            )
        else:
            return PROMPT_TEMP_num.format(
                question=question,
            )
            
    elif mode == "step":
        if init_flag:
            if qa_type=='multiple choice':
                return Prompt_temp_init_mc.format(
                question=question,
                options=options,
            )
            else:
                return Prompt_temp_init_gen.format(
                question=question,
                type_fix=type_fix,
            )
        elif final_flag:
            if qa_type=='multiple choice':
                return Prompt_temp_final_mc.format(
                question=question,
                options=options,
            )
            else:
                return Prompt_temp_final_gen.format(
                question=question,
                type_fix=type_fix,
            )
        else:
            if qa_type=='multiple choice':
                return Prompt_temp_round_mc.format(
                question=question,
                options=options,
            )
            else:
                return Prompt_temp_round_gen.format(
                question=question,
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

def eval_gt(response,gt,mode):
    if "<answer>" in response:
        response=extract_answer(response)
    if response == None:
        return 0
    try:
        output_ans = response
        gt_ans = gt
        if mode == "multiple choice":
            return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
        elif mode == "numerical":
            gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
            out_has_decimal = ("." in output_ans) or ("," in output_ans)
            if gt_has_decimal != out_has_decimal:
                return 0.0
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                return 0.0
            return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
    except Exception as e:
        return 0.0
def evaluate_example(model, example: dict, mode: str, max_frames: int, max_steps=1) -> Tuple[str, bool]:
    video_path = example["video_path"]
    question   = example["question"]
    gt         = str(example["answer"]).strip()
    options    = example.get("options",None)
    qa_type    = example["qa_type"]
    if mode == "uniform":
        vr, fps = video_io.load_video(video_path)
        idxs = video_io.uniform_sample_idx(len(vr), max_frames)
        frames = video_io.save_video_frames(video_path, idxs, "frames/videommmu")
        imgs, _ = zip(*frames)
        prompt = build_prompt(question, qa_type, mode, options,False,False) 
        message=construct_message(prompt,list(imgs))
        response, _ = model.chat(message)
        pred = response.strip()
        return pred, prompt, message, eval_gt(pred, gt, qa_type)

    
    history = []
    history_frames = []
    step_idx=0
    while True:
        if step_idx == 0 or not history_frames:
            prompt = build_prompt(question, qa_type, mode, options,True,False) 
            vr, fps = video_io.load_video(video_path)
            idxs = video_io.uniform_sample_idx(len(vr), max_frames)
            frames = video_io.save_video_frames(video_path, idxs, "frames/videommmu")
        elif step_idx == 2:
            prompt = build_prompt(question, qa_type, mode, options,False,True)
        else:
            prompt = build_prompt(question, qa_type, mode, options,False,False)
        imgs, ts = zip(*frames)
        history_frames = history_frames+list(ts)
        message=construct_message(prompt,list(imgs),ts,True)
        response, history = model.chat(message,history=history)
     
        m,timestamps = identify_replay(response)
        if m:
            replay_idxs = clips_to_frame_indices(video_path, timestamps, 8, history_frames)
            if replay_idxs:
                frames = video_io.save_video_frames(video_path, replay_idxs, "frames/videommmu")
            else:break
        else: break
        step_idx=step_idx+1
        if step_idx>=max_steps:break
        

    pred = response.strip()
    return pred, prompt, history, eval_gt(pred, gt, qa_type)
def make_result_record(ex: dict, prompt: str, messages: List[dict], pred: str, correct: bool) -> dict:
    """
    构造需要保存的字段，benchmark 可自由增删
    """
    return {
        "Question ID": ex["Question ID"],
        "video_path": ex["video_path"],
        "question": ex["question"],
        "options": ex["options"],
        "prompt": prompt,
        "messages": messages,
        "gt": str(ex["answer"]),
        "pred": pred,
        "correct": correct,
        "task": ex["task"],
    }
def result_statistics(run_dir: Path) -> Dict[str, Any]:
    """
    读取 results_all.jsonl，按 task 分组统计准确率与样本数
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
    overall_acc = float(np.mean([r["correct"] for r in all_results]))

    # 按 task 分组
    task_buckets: Dict[str, List[Dict[str, Any]]] = {}
    no_response=0
    for r in all_results:
        if "<answer>" in r['pred']:
            response=extract_answer(r['pred'])
        else:
            response=r['pred']
        if response!='None':
            response = extract_characters(response)
            if response is None:
                no_response=no_response+1
        task = r.get("task", "unknown_task")
        task_buckets.setdefault(task, []).append(r)
    bad_counts={"inference error":len(failed_res),"retrieval error":no_response}
    task_accuracy: Dict[str, float] = {}
    task_samples: Dict[str, int] = {}
    for task, items in task_buckets.items():
        task_accuracy[task] = float(np.mean([r["correct"] for r in items])) if items else 0.0
        task_samples[task] = len(items)

    summary = {
        "overall_accuracy": overall_acc,
        "total_samples": total,
        "bad_counts": bad_counts,
        "task_accuracy": task_accuracy,
        "task_total_samples": task_samples,
    }

    # 落盘
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary