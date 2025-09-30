"""
mmr-v.py 
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
QA_FNAME   = "/mnt/moonfs/ouyangkun-m2/dataset/MMR-VBench/data/test-00000-of-00001.parquet"
VIDEO_DIR  = "/mnt/moonfs/ouyangkun-m2/dataset/MMR-VBench"   
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

Give the final correct option number in the following format: \"[[A]]\" or \"[[B]]\" or \"[[C]]\" or \"[[D]]\" ...
[[END OF OUTPUT FORMAT]]
"""
def get_paths() -> Tuple[str, str]:
    """
    返回 (qa_path, video_root_path)
    方便其它 benchmark 复用 main.py
    """
    return  QA_FNAME, VIDEO_DIR
# 放在 videomme.py 末尾即可
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
        video_id=item["video"]
        out.append({
            "video_id": video_id,           
            "video_path": os.path.join(video_dir, video_id),
            "videoType": item["videoType"],
            "question":   item["question"],
            "options":    item["options"],
            "answer":     str(item["correctAnswer"]),
            "duration":   item.get("duration"),
            "abilityType_L2": item["abilityType_L2"],
            "abilityType_L3": item["abilityType_L3"],
            "question_idx": item["question_idx"]
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
    # print(f"Extracted answer: {response}")
    # print(f"Ground truth: {gt}")
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
        frames = video_io.save_video_frames(video_path, idxs, "/mnt/moonfs/ouyangkun-m2/code/videore/frames/mmr-v")
        imgs, _ = zip(*frames)
        prompt = build_prompt(question, mode, options,False,False) 
        message=construct_message(prompt,list(imgs))
        # try:
        response, _ = model.chat(message)
        # except:
        #     response="None"
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
            frames = video_io.save_video_frames(video_path, idxs, "/mnt/moonfs/ouyangkun-m2/code/videore/frames/mmr-v")
        elif step_idx == 2:
            prompt = build_prompt(question, mode, options,False,True)
        else:
            prompt = build_prompt(question, mode, options,False,False)
        imgs, ts = zip(*frames)
        history_frames = history_frames+list(ts)
        message=construct_message(prompt,list(imgs),ts,True)
        # try:
        response, history = model.chat(message,history=history)
        # except:
        #     response="None"
        #     break
        # replay 逻辑
        m,timestamps = identify_replay(response)
        if m:
            replay_idxs = clips_to_frame_indices(video_path, timestamps, 8, history_frames)
            if replay_idxs:
                frames = video_io.save_video_frames(video_path, replay_idxs, "/mnt/moonfs/ouyangkun-m2/code/videore/frames/mmr-v")
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
        "question": ex["question"],
        "prompt": prompt,
        "messages": messages,
        "gt": str(ex["answer"]),
        "pred": pred,
        "correct": correct,
        "abilityType_L2": ex["abilityType_L2"],
        "abilityType_L3": ex["abilityType_L3"],
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
    bad_counts={}
    try:
        with open(run_dir / "log.jsonl", encoding="utf-8") as f:
            for line in f:
                failed_res.append(json.loads(line))
    except:
        print("No generation failed samples")
    total = len(all_results)
    # 按 reasoning type 分组
    implicit_buckets = {"Metaphor Understanding": [], "Theme Understanding": [], "Emotion Recognition": [], "Comment Matching": [], "Implicit Symbol": []}
    explicit_buckets = {"Causal Reasoning": [], "Sequential Structure Reasoning": [], "Counterintuitive Reasoning": [], "Cross-modal Creative Transfer": [], "Video Type and Intent": []}
    unknown_bucket: List[Dict[str, Any]] = [] 
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
        rea_type = r.get("abilityType_L2", "unknown")
        if rea_type in implicit_buckets:
            implicit_buckets[rea_type].append(r)
        elif rea_type in explicit_buckets:
            explicit_buckets[rea_type].append(r)
    bad_counts={"inference error":len(failed_res),"retrieval error":no_response}
    overall_acc = np.mean([r["correct"] for r in all_results])
    # 1. 计算每个子类型的准确率
    implicit_subtype_accuracy: Dict[str, float] = {}
    for k, v in implicit_buckets.items():
        if v:
            implicit_subtype_accuracy[k] = np.mean([r["correct"] for r in v])
        else:
            implicit_subtype_accuracy[k] = None # 或 0.0，取决于你希望如何处理空类别
    implicit_all_results = [r for k in implicit_buckets for r in implicit_buckets[k]]
    implicit_category_accuracy = np.mean([r["correct"] for r in implicit_all_results]) if implicit_all_results else None
    implicit_category_samples = len(implicit_all_results)
    implicit_subtype_accuracy["overall"]=implicit_category_accuracy
    implicit_subtype_accuracy["total_samples"]=implicit_category_samples

    explicit_subtype_accuracy: Dict[str, float] = {}
    for k, v in explicit_buckets.items():
        if v:
            explicit_subtype_accuracy[k] = np.mean([r["correct"] for r in v])
        else:
            explicit_subtype_accuracy[k] = None # 或 0.0
    explicit_all_results = [r for k in explicit_buckets for r in explicit_buckets[k]]
    explicit_category_accuracy = np.mean([r["correct"] for r in explicit_all_results]) if explicit_all_results else None
    explicit_category_samples = len(explicit_all_results)
    explicit_subtype_accuracy["overall"]=explicit_category_accuracy
    explicit_subtype_accuracy["total_samples"]=explicit_category_samples
    summary = {
        "overall_accuracy": float(overall_acc),
        "total_samples": total,
        "bad_counts": bad_counts,
        "implicit_subtype_accuracy": implicit_subtype_accuracy,
        "explicit_subtype_accuracy": explicit_subtype_accuracy,
    }

    # 同时保存 summary.json
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary