"""
humanpcr.py 
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
import random
from prompt_temp import *
# ========== 1. 路径配置 ==========
QA_FNAME   = "HumanPCR/HumanPC_preview.json"
VIDEO_DIR  = "HumanPCR/visual_data"   
PROMPT_TEMP=""""
Question: {question}
Options: {options}
Please answer the question with an optoin letter.
The answer is:
"""
def get_paths() -> Tuple[str, str]:
    """
    返回 (qa_path, video_root_path)
    方便其它 benchmark 复用 main.py
    """
    return  QA_FNAME, VIDEO_DIR
def construct_options(distractors: List[str], answer: str) -> Tuple[str, str]:
    candidates = [(d, False) for d in distractors] + [(answer, True)]
    random.shuffle(candidates)
    lines = []
    correct_letter = None
    for idx, (text, is_answer) in enumerate(candidates):
        letter = chr(ord('A') + idx)
        lines.append(f"{letter}. {text}")
        if is_answer:
            correct_letter = letter

    return "\n".join(lines), correct_letter
def load_data() -> list[dict]:

    qa_path, video_dir = get_paths()
    raw = load_qa(qa_path)     
    out = []
    data_types={"0":"single image","1":"multi images","2":"video"}
    for idx,item in enumerate(raw):
        input_path=[os.path.join(VIDEO_DIR,pt) for pt in item["input_path"]]
        options,answer=construct_options(item['distractors'],item['answer'])
        out.append({
            "question_id": idx,         
            "input_path": input_path,
            "data_type": data_types[str(item['input_type'])],
            "question":   item["question"],
            "options":    options,
            "answer":     answer,
            "task":   item.get("level")
        })
    return out
# ========== 2. prompt & 单条评测 ==========
def build_prompt(question: str, mode: str, options: List[str], data_type: str, init_flag: bool,final_flag: bool) -> str:
    if mode == "uniform":
        return PROMPT_TEMP.format(
            question=question,
            options=options,
        )
    elif mode == "step":
        if data_type=="video":
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
        else:
            # return Prompt_temp_cot_mc.format(
            #     question=question,
            #     options=options,
            # )
            return PROMPT_TEMP.format(
                question=question,
                options=options,
            )
    elif mode == "cot":
        return Prompt_temp_cot_mc.format(
            question=question,
            options=options,
        )
def construct_message(prompt,frames,data_type,timestamps=[],insert_timestamp=False):
    content=[]
    if data_type=="video":
        if insert_timestamp:
            assert len(frames) == len(timestamps)
            for ts, frame in zip(timestamps,frames):
                content.append({"type": "text", "text": ts})
                content.append({"type": "image", "image": frame})
            content.append({"type": "text", "text": prompt})
        else:
            content.append({"type": "video", "video": frames})
            content.append({"type": "text","text": prompt})
    elif data_type=="single image":
        content.append({"type": "image", "image": frames[0]})
        content.append({"type": "text", "text": prompt})
    elif data_type=="multi images":
        for frm in frames:
            content.append({"type": "image", "image": frm})
        content.append({"type": "text", "text": prompt})
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
    question   = example["question"]
    gt         = str(example["answer"]).strip()
    options    = example["options"]
    data_type=example['data_type']
    if mode == "uniform":
        if data_type=='video':
            video_path=example["input_path"][0]
            vr, fps = video_io.load_video(video_path)
            idxs = video_io.uniform_sample_idx(len(vr), max_frames)
            frames = video_io.save_video_frames(video_path, idxs,"frames/humanpcr")
            imgs, _ = zip(*frames)
        else:imgs=example["input_path"]
        prompt = build_prompt(question, mode, options, data_type, False,False) 
        message=construct_message(prompt,list(imgs),data_type)
        response, _ = model.chat(message)
        pred = response.strip()
        return pred, prompt, message, eval_gt(pred, gt), 1
    
    history = []
    history_frames = []
    step_idx=0
    while True:
        if data_type=='video':
            video_path=example["input_path"][0]
            if step_idx == 0 or not history_frames:
                prompt = build_prompt(question, mode, options,data_type,True,False) 
                vr, fps = video_io.load_video(video_path)
                idxs = video_io.uniform_sample_idx(len(vr), max_frames)
                frames = video_io.save_video_frames(video_path, idxs, "frames/humanpcr")
            elif step_idx == 2:
                prompt = build_prompt(question, mode, options,data_type,False,True)
            else:
                prompt = build_prompt(question, mode, options,data_type,False,False)
        else:
            imgs=example["input_path"]
            prompt = build_prompt(question, mode, options,data_type,True,False)
            message=construct_message(prompt,list(imgs),data_type)
            response, history = model.chat(message,history=history)
            break
        imgs, ts = zip(*frames)
        history_frames = history_frames+list(ts)
        message=construct_message(prompt,list(imgs),data_type,ts,True)
        response, history = model.chat(message,history=history)

        # replay 逻辑
        m,timestamps = identify_replay(response)
        if m:
            replay_idxs = clips_to_frame_indices(video_path, timestamps, 8, history_frames)
            if replay_idxs:
                frames = video_io.save_video_frames(video_path, replay_idxs, "frames/humanpcr")
            else:break
        else: break
        step_idx=step_idx+1
        if step_idx>=max_steps:break

    pred = response.strip()
    return pred, prompt, history, eval_gt(pred, gt), step_idx+1
def make_result_record(ex: dict, prompt: str, messages: List[dict], pred: str, correct: bool, rounds: int) -> dict:
    return {
        "question_id": ex["question_id"],
        "data": ex["input_path"],  
        "question": ex["question"],
        "data_type": ex["data_type"],
        "task": ex["task"],
        "prompt": prompt,
        "messages": messages,
        "gt": str(ex["answer"]),
        "pred": pred,
        "correct": correct,
        "rounds": rounds
    }
def result_statistics(run_dir: Path) -> Dict[str, Any]:
    all_results = [json.loads(l) for l in open(run_dir / "results_all.jsonl", encoding="utf-8")]

    # ----  bad counts  ----
    failed_res = []
    if (run_dir / "log.jsonl").exists():
        failed_res = [json.loads(l) for l in open(run_dir / "log.jsonl", encoding="utf-8")]
    no_response = 0
    for r in all_results:
        pred = r["pred"]
        response = extract_answer(pred) if "<answer>" in pred else pred
        if response != "None":
            response = extract_characters(response)
            if response is None:
                no_response += 1
    bad_counts = {"inference error": len(failed_res), "retrieval error": no_response}
    total = len(all_results)
    overall_acc = float(np.mean([r["correct"] for r in all_results]))
    rounds_vals = [int(r["rounds"]) for r in all_results if r.get("rounds") is not None]
    avg_rounds = float(np.mean(rounds_vals)) if rounds_vals else None
    avg_frm = avg_input_frames(all_results)
    dt_task_buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    task_buckets: Dict[str, List[Dict[str, Any]]] = {}

    for r in all_results:
        dt = r.get("data_type", "unknown_data_type")
        task = r.get("task", "unknown_task")
        task_buckets.setdefault(task, []).append(r)
        dt_task_buckets.setdefault(dt, {}).setdefault(task, []).append(r)

    def _acc_and_total(items: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            "accuracy": float(np.mean([r["correct"] for r in items])) if items else 0.0,
            "total": len(items),
        }

    by_task = {t: _acc_and_total(items) for t, items in task_buckets.items()}

    by_data_type = {}
    for dt, task_dict in dt_task_buckets.items():
        dt_items = [r for lst in task_dict.values() for r in lst]  
        by_data_type[dt] = {
            "accuracy": _acc_and_total(dt_items)["accuracy"],
            "total": len(dt_items),
            "by_task": {t: _acc_and_total(lst) for t, lst in task_dict.items()},
        }

    summary = {
        "overall_accuracy": overall_acc,
        "total_samples": total,
        "bad_counts": bad_counts,
        "avg_rounds": avg_rounds,
        "avg_frames": avg_frm,
        "by_task": by_task,          
        "by_data_type": by_data_type, 
    }

    # 落盘
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary