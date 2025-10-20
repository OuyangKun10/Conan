"""
longvideoreason.py 
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
QA_FNAME   = "LongVideo-Reason/test.jsonl"
VIDEO_DIR  = "LongVideo-Reason/longvila_videos"   
PROMPT_TEMP=""""
Question: {question}
Please answer the question with an optoin letter.
The answer is:
"""
Prompt_temp_init_mc="""
You are given a single-choice question, options, and several video frames with their timestamps.
For each frame/clip, assign a relevance score on a scale of 1 to 5 (where 5 = highly relevant, 3 = medium relevant and 1 = not relevant), and include the medium or high scoring clip(s) within <score></score>.
And you should perform step-by-step reasoning before making final action.
Guidelines for reasoning:
1. Begin by analyzing the question, clarifying what kind of evidence is required.
2. Analyze the relevant frames with high scores that help answer the question.
3. Compare the available evidence across frames, giving a summary.
4. Justify whether the available information is sufficient to answer accurately.
Action:
If not, you should retrieve additional clip(s) and specify them in <clip></clip>, e.g., <score>the scores corresponding to the clips</score><think>your reasoning process</think><clip>00:00:05-00:00:10</clip><answer></answer>.
If yes, you should answer the question with an option letter in <answer></answer>, e.g., <score>the scores corresponding to the clips</score><think>your reasoning process</think><clip></clip><answer>C</answer>.

Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
"""
Prompt_temp_round_mc="""
Please identify the new frame scores, perform step-by-step reasoning, and make final action based on the history and new information.
Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
"""
Prompt_temp_final_mc="""
Please identify the new frame scores, perform step-by-step reasoning, and answer the question based on the history and new information.
You should output an option letter in <answer></answer> tag.
Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
"""
Prompt_temp_cot_mc="""
    Question: {question}
    Please think about this question as if you were a human pondering deeply.
    Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions
    It's encouraged to include self-reflection or verification in the reasoning process.
    Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
    Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.
"""
def get_paths() -> Tuple[str, str]:
    return  QA_FNAME, VIDEO_DIR
def load_data() -> list[dict]:
    qa_path, video_dir = get_paths()
    raw = load_qa(qa_path)     
    out = []
    for item in raw:
        video_id=item["videos"].split('/')[-1]
        if "webm" in video_id:
            video_id=video_id.replace("webm","mp4")
        elif "mkv" in video_id:
            video_id=video_id.replace("mkv","mp4")
        out.append({
            "question_id": item["problem_id"],
            "video_name": video_id,           
            "video_path": os.path.join(video_dir, video_id),
            "question":   item["problem"],
            "answer":     extract_answer(item['answer']),
            "task":       item['problem_type']
        })
    return out
# ========== 2. prompt & 单条评测 ==========
def build_prompt(question: str, mode: str, init_flag: bool,final_flag: bool) -> str:
    if mode == "uniform":
        return PROMPT_TEMP.format(
            question=question,
        )
    elif mode == "step":
        if init_flag:
            return Prompt_temp_init_mc.format(
            question=question,
        )
        elif final_flag:
            return Prompt_temp_final_mc.format(
            question=question,
        )
        else:
            return Prompt_temp_round_mc.format(
            question=question,
        )
    elif mode == "cot":
        return Prompt_temp_cot_mc.format(
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
    response = extract_characters(response)
    if response == None:
        return 0
    else: return response.lower()==gt.lower()
def evaluate_example(model, example: dict, mode: str, max_frames: int, max_steps=1) -> Tuple[str, bool]:
    video_path = example["video_path"]
    question   = example["question"]
    gt         = str(example["answer"]).strip()
    if mode == "uniform":
        vr, fps = video_io.load_video(video_path)
        idxs = video_io.uniform_sample_idx(len(vr), max_frames)
        frames = video_io.save_video_frames(video_path, idxs,"frames/longvideoreason")
        imgs, _ = zip(*frames)
        prompt = build_prompt(question, mode, False,False) 
        message=construct_message(prompt,list(imgs))
        response, _ = model.chat(message)
        pred = response.strip()
        return pred, prompt, message, eval_gt(pred, gt),1

    
    history = []
    history_frames = []
    step_idx=0
    while True:
        if step_idx == 0 or not history_frames:
            prompt = build_prompt(question, mode,True,False) 
            vr, fps = video_io.load_video(video_path)
            idxs = video_io.uniform_sample_idx(len(vr), max_frames)
            frames = video_io.save_video_frames(video_path, idxs, "frames/longvideoreason")
        elif step_idx == 2:
            prompt = build_prompt(question, mode, False,True)
        else:
            prompt = build_prompt(question, mode, False,False)
        imgs, ts = zip(*frames)
        history_frames = history_frames+list(ts)
        message=construct_message(prompt,list(imgs),ts,True)
        response, history = model.chat(message,history=history)
        m,timestamps = identify_replay(response)
        if m:
            replay_idxs = clips_to_frame_indices(video_path, timestamps, 8, history_frames)
            if replay_idxs:
                frames = video_io.save_video_frames(video_path, replay_idxs, "frames/longvideoreason")
            else:break
        else: break
        step_idx=step_idx+1
        if step_idx>=max_steps:break

    pred = response.strip()
    return pred, prompt, history, eval_gt(pred, gt),step_idx+1
def make_result_record(ex: dict, prompt: str, messages: List[dict], pred: str, correct: bool, rounds: int) -> dict:
    return {
        "question_id": ex["question_id"],
        "video": ex["video_path"],
        "question": ex["question"],
        "prompt": prompt,
        "messages": messages,
        "task": ex["task"],
        "gt": str(ex["answer"]),
        "pred": pred,
        "correct": correct,
        "rounds": rounds
    }
def result_statistics(run_dir: Path) -> Dict[str, Any]:
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
    rounds_vals = [int(r['rounds']) for r in all_results if r.get('rounds') is not None]
    avg_rounds = sum(rounds_vals) / len(rounds_vals) if rounds_vals else None
    avg_frm=avg_input_frames(all_results)
    summary = {
        "overall_accuracy": overall_acc,
        "total_samples": total,
        "bad_counts": bad_counts,
        "task_accuracy": task_accuracy,
        "task_total_samples": task_samples,
        "avg_rounds": avg_rounds,
        "avg_frames": avg_frm,
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary