"""
vrbench.py 
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
PROMPT_TEMP_mc=""""
Question:\n{question}
Options:\n{options}
Please answer the question with an optoin letter.
The answer is:
"""
PROMPT_TEMP_free=""""
Question:\n{question}
The answer is:
"""
VIDEO_DIR  = "VCR-Bench/v1/videos"
QA_FNAME   = "VCR-Bench/v1/videos/meta_info_video.json"   
def get_paths() -> List[Tuple[str, str]]:
    """
    返回 (qa_path, video_root_path)
    """
    
    return  VIDEO_DIR,QA_FNAME
def load_data() -> list[dict]:
    VIDEO_DIR,QA_FNAME= get_paths()
    out = []
    raw = load_qa(QA_FNAME)   
    for item in raw:
        video_id=os.path.basename(item["video_path"])
        video_path=item["video_path"]
        if item["multiple-choice"]:
            out.append({
                "Question ID": item['id'],
                "video_id": video_id,           
                "video_path": f"{VIDEO_DIR}/{video_path}",
                "task": item["dimension"],
                "question_type": "multiple-choice" if item["multiple-choice"] else "free-form",
                "question": item["question"],
                "options": item["choices"],
                "answer": item["answer"],
                "duration": item["duration"]
            })
    return out
# ========== 2. prompt & 单条评测 ==========
def build_prompt(question: str, mode: str, options: str, init_flag: bool, final_flag: bool) -> str:
    if mode == "uniform":
        if options:
            return PROMPT_TEMP_mc.format(
                question=question,
                options=options,
            )
        else:
            return PROMPT_TEMP_free.format(
                question=question,
            )

    elif mode == "step":
        if init_flag:
            if options:
                return Prompt_temp_init_mc.format(
                question=question,
                options=options,
                )
            else:
                return Prompt_temp_init_gen.format(
                question=question,
                )

        elif final_flag:
            if options:
                return Prompt_temp_final_mc.format(
                question=question,
                options=options,
                )
            else:
                return Prompt_temp_final_gen.format(
                question=question,
                )
        else:
            if options:
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

def eval_gt(response,gt):
    if "<answer>" in response:
        response=extract_answer(response)
    if response == None:
        return 0
    # if mode=="multiple-choice":
    response = extract_characters(response)
    gt = extract_characters(gt)
    if response == None:
        return 0
    else: return response.lower()==gt.lower()
    # else:
        
    # print(f"Extracted answer: {response}")
    # print(f"Ground truth: {gt}")
    

def evaluate_example(model, example: dict, mode: str, max_frames: int, max_steps=1) -> Tuple[str, bool]:
    video_path = example["video_path"]
    question   = example["question"]
    options    = example["options"]
    opt_str=""
    for cha,opt in options.items():
        opt_str=opt_str+f"{cha}. {opt}\n"
    gt         = str(example["answer"]).strip()
    if mode == "uniform":
        vr, fps = video_io.load_video(video_path)
        idxs = video_io.uniform_sample_idx(len(vr), max_frames)
        frames = video_io.save_video_frames(video_path, idxs, "frames/vcrbench")
        imgs, _ = zip(*frames)
        prompt = build_prompt(question,mode,opt_str,False,False) 
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
            prompt = build_prompt(question, mode, opt_str,True,False) 
            vr, fps = video_io.load_video(video_path)
            idxs = video_io.uniform_sample_idx(len(vr), max_frames)
            frames = video_io.save_video_frames(video_path, idxs, "frames/vcrbench")
        elif step_idx == 2:
            prompt = build_prompt(question, mode,opt_str,False,True)
        else:
            prompt = build_prompt(question, mode,opt_str,False,False)
        # print("prompt: ",prompt)
        imgs, ts = zip(*frames)
        history_frames = history_frames+list(ts)
        # print("history_frames:",history_frames)
        # print(f"retrived frames:\n",ts)
        message=construct_message(prompt,list(imgs),ts,True)
        # try:
        response, history = model.chat(message,history=history)
        # except:
        #     response="None"
        #     break
        # replay 逻辑
        m,timestamps = identify_replay(response)
        # print("Replay: ",timestamps)
        if m:
            replay_idxs = clips_to_frame_indices(video_path, timestamps, 8, history_frames)
            # print("replay_idxs: ",replay_idxs)
            if replay_idxs:
                frames = video_io.save_video_frames(video_path, replay_idxs, "frames/vcrbench")
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
        "Question ID": ex["Question ID"],
        "video_path": ex["video_path"],
        "question": ex["question"],
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
    # 计算每个 task 的准确率与样本数
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