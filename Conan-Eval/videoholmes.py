"""
videoholmes.py 
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
QA_FNAME   = "Video-Holmes/test_Video-Holmes.json"
VIDEO_DIR  = "Video-Holmes/videos"   
ANALYZE_PROMPT ="""
[[INSTRUCTIONS]]
Please select the best answer to the following multiple-choice question based on the video. 
Only one option is the most accurate answer in relation to the question and the video.
Each frame is labeled with a timestamp in the format HH:MM:SS.

What is the correct answer to this question [[QUESTION]]
Options:
[[OPTIONS]]

[[END OF INSTRUCTIONS]]
[[QUESTION]]
{question}
[[END OF QUESTION]]
[[OPTIONS]]
{options}
[[END OF OPTIONS]]
[[OUTPUT FORMAT]]

Task:

1. Decide whether the **provided frames contain enough information** to answer the question.
2. Output **exactly** in the following XML-like tags:\n<replay>True/False</replay>\n<clip>…</clip>\n<answer>…</answer>

Rules:

• If the frames are sufficient → set <replay>False</replay>, leave <clip></clip> empty, and place the correct **option letter** inside <answer></answer>.
• If the frames are **not** sufficient → set <replay>True</replay>, list the required clip(s) inside <clip></clip> (e.g., 00:05:20-00:06:30).
Do **not** add any extra text outside these tags.
"""

PROMPT_TEMP = """
[[INSTRUCTIONS]]
Please select the best answer to the following multiple-choice question based on the video. 
Only one option is the most accurate answer in relation to the question and the video.

What is the correct answer to this question [[QUESTION]]
Options:
[[OPTIONS]]

[[END OF INSTRUCTIONS]]
[[QUESTION]]
{question}
[[END OF QUESTION]]
[[OPTIONS]]
{options}
[[END OF OPTIONS]]
[[OUTPUT FORMAT]]
Format your answer as follows:

Give the final correct option letter in the following format: A, B, C, D, E, F, etc.
[[END OF OUTPUT FORMAT]]
"""
def get_paths() -> Tuple[str, str]:
    """
    返回 (qa_path, video_root_path)
    方便其它 benchmark 复用 main.py
    """
    return  QA_FNAME, VIDEO_DIR
def load_data() -> list[dict]:
    qa_path, video_dir = get_paths()
    raw = load_qa(qa_path)     
    out = []
    for item in raw:
        video_id=item["video ID"]
        out.append({
            "Question ID": item["Question ID"],
            "video_id": video_id,           
            "video_path": os.path.join(video_dir, f"{video_id}.mp4"),
            "task": item["Question Type"],
            "question": item["Question"],
            "options": ', '.join([f"{key}: {value}" for key, value in item["Options"].items()]),
            "answer": item["Answer"],
            "duration": item.get("duration"),
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
    gt = extract_characters(gt)
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
        frames = video_io.save_video_frames(video_path, idxs, "frames/videoholmes")
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
            frames = video_io.save_video_frames(video_path, idxs, "frames/videoholmes")
        elif step_idx == 2:
            prompt = build_prompt(question, mode, options,False,True)
        else:
            prompt = build_prompt(question, mode, options,False,False)
        imgs, ts = zip(*frames)
        history_frames = history_frames+list(ts)
        message=construct_message(prompt,list(imgs),ts,True)
        response, history = model.chat(message,history=history)
        m,timestamps = identify_replay(response)
        if m:
            replay_idxs = clips_to_frame_indices(video_path, timestamps, 8, history_frames)
            if replay_idxs:
                frames = video_io.save_video_frames(video_path, replay_idxs, "frames/videoholmes")
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

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary