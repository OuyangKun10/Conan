# video_clip_scheduler.py
from __future__ import annotations

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from swift.plugin.orm import ORM, orms
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
def _to_seconds(ts: str) -> float:
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

import re
from typing import List, Tuple

def extract_timestamps(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r'(?P<start>\d{1,2}:\d{2}:\d{2})'
        r'(?:'
        r'\s*-\s*'
        r'(?P<end>\d{1,2}:\d{2}:\d{2})'
        r')?'
    )
    results = []
    for m in pattern.finditer(text):
        start = m.group('start')
        end = m.group('end') if m.group('end') else start
        results.append((start, end))
    return results
def _fmt_timestamp(sec: float) -> str:
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = int(sec % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

def _get_clip(response: str) -> List[Tuple[str, str]]:
    if '<clip>' not in response:
        return []
    m = re.search(r"<clip>(.*?)</clip>", response, re.DOTALL)
    if not m:
        return []
    timestamps = extract_timestamps(m.group(1))
    valid = []
    for s, e in timestamps:
        try:
            s_sec, e_sec = _to_seconds(s), _to_seconds(e)
            if s_sec <= e_sec:
                valid.append((s, e))
        except ValueError:
            continue
    return valid
def _idx2sec(idx: int, fps: float, start_frm_sec: float) -> float:
    return idx / fps + start_frm_sec
def _get_frm_paths(
    frm_dir: str,
    frame_files: List[str],
    timestamps: List[Tuple[str, str]],
    max_frames: int,
    total_frms: int,
    fps: float,
    history_frames_ids: Optional[List[float]] = None,
) -> Tuple[List[float], List[str]]:
    if history_frames_ids is None:
        history_frames_ids = []
    start_frm_sec=float((os.path.basename(frame_files[0])).split('.')[0])
    idx2path={}
    all_seconds=[]
    for idx,fp in enumerate(frame_files):
        basename = os.path.basename(fp)
        second = float(basename.split('.')[0])
        idx2path[idx]=fp
        all_seconds.append(second)
    if not idx2path:
        return [], []
    all_seconds = sorted(all_seconds)
    max_second = all_seconds[-1]
    def _sec2idx(sec: float, fps: float, start_frm_sec: float) -> int:
        return int(np.clip(round((sec-start_frm_sec) * fps), 0, round(max_second * fps-1)))
    history_frames_ids=[_sec2idx(i,fps,start_frm_sec) for i in history_frames_ids]
    history_set = set(history_frames_ids)
    clips = []
    for start_ts, end_ts in timestamps:
        s_sec, e_sec = _to_seconds(start_ts), _to_seconds(end_ts)
        s_idx, e_idx = _sec2idx(s_sec,fps,start_frm_sec), _sec2idx(e_sec,fps,start_frm_sec)
        if any(i not in history_set for i in range(s_idx, e_idx + 1)):
            clips.append((s_idx, e_idx))
    if not clips:
        return [], []
    min_per_clip = 1
    remain = max_frames - len(clips) * min_per_clip
    remain = max(remain, 0)
    weights = []
    for start, end in clips:
        avail = sum(1 for i in range(start, end + 1) if i not in history_set)
        weights.append(avail)
    weights = np.array(weights, dtype=float)

    quotas = np.zeros_like(weights, int)
    if weights.sum() > 0:
        quotas = (weights / weights.sum() * remain).astype(int)
    deficit = int(remain - quotas.sum())
    for i in range(deficit):
        quotas[np.argsort(weights)[::-1][i % len(weights)]] += 1
    quotas += min_per_clip
    selected = set()
    for (start, end), quota in zip(clips, quotas):
        avail = [i for i in range(start, end + 1) if i not in history_set]
        if not avail:
            continue
        if len(avail) <= quota:
            selected.update(avail)
        else:
            step = (len(avail) - 1) / (quota - 1) if quota > 1 else 0
            selected.update([avail[round(i * step)] for i in range(quota)])

    selected = list(selected)
    if len(selected) < max_frames:
        pass
    elif len(selected) > max_frames:
        step = (len(selected) - 1) / (max_frames - 1) if max_frames > 1 else 0
        selected = [selected[round(i * step)] for i in range(max_frames)]
    selected=sorted(selected)
    seconds = [_idx2sec(i, fps, start_frm_sec) for i in sorted(set(selected))]
    selected_frm=[idx2path[idx] for idx in sorted(set(selected))]
    return seconds, selected, selected_frm

def construct_next_message(frm_sec: List[float],
                           frm_paths: List[str],
                           question: str,
                           options: Optional[str] = None) -> List[Dict]:
    prompt = (f"Please identify the new frame scores, perform step-by-step reasoning, "
              f"and make final action based on the history and new information.\n"
              f"Output format:\n<score>...</score><think>...</think><clip>...</clip><answer>...</answer>\n"
              f"Question: {question}")
    if options:
        prompt += f"\nOption: {options}"
    query=""
    content = []
    for sec, path in zip(frm_sec, frm_paths):
        query=query+_fmt_timestamp(sec)+": <image>\n"
        content.append({"type": "text", "text": _fmt_timestamp(sec)})
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": prompt})
    query=query+prompt
    return query
def identify_stop(response,data):
    timestamps = _get_clip(response)
    if not timestamps:
        return True
    try:
        frm_dir = data["frame_dir"]
        frame_files = data["frames"]
        total_frms=data["frame_num"]
        max_frames = 8
        fps = data["fps"]
        question = data["question"]
        options = data.get("options")
        history = data.get("history_frames", [])
        new_secs, new_idx, new_paths = _get_frm_paths(
            frm_dir, frame_files, timestamps, max_frames, total_frms, fps, history
        )
    except:
        return True
    if not new_paths:
        return True
    return False
def _get_score(content: str, round_frm: List[int], fps: float, start_frm_sec: float) -> Dict[int, int]:
    score_map: Dict[int, int] = {}
    
    if '<score>' not in content:
        return {}
    
    m = re.search(r"<score>(.*?)</score>", content, re.DOTALL)
    if not m:
        return {}
    
    score_content = m.group(1)
    total_frms = max(round_frm) if round_frm else 16 

    def _sec2idx(sec: float, fps: float, start_frm_sec: float) -> int:
        if fps == 0:  
            return 0
        return int(np.clip(round((sec - start_frm_sec) * fps), 0, total_frms))

    pattern = re.compile(
        r'(?:(\d{2}):(\d{2}):(\d{2}))'        
        r'(?:-(\d{2}):(\d{2}):(\d{2}))?'      
        r':\s*(\d+)'                         
    )
    
    for seg in score_content.split(','):
        seg = seg.strip()
        m = pattern.match(seg)
        if not m:
            continue
        
        sh, sm, ss, eh, em, es, score_val = m.groups()
        
        start_sec = int(sh) * 3600 + int(sm) * 60 + float(ss)
        end_sec = int(eh) * 3600 + int(em) * 60 + float(es) if eh else start_sec
        
        start_frm = _sec2idx(start_sec, fps, start_frm_sec)
        end_frm   = _sec2idx(end_sec, fps, start_frm_sec)
        
        for frm in range(start_frm, end_frm + 1):
            if frm in round_frm:
                score_map[int(frm)] = int(score_val)
    
    return score_map
class VideoClipScheduler(MultiTurnScheduler):
    def __init__(self, max_turns: Optional[int] = 8, **kwargs):
        super().__init__(max_turns=max_turns, **kwargs)
    def check_finished(self,
                       infer_request,
                       response_choice,
                       current_turn):
        stop = super().check_finished(infer_request, response_choice, current_turn)
        if stop:
            return True
        last_completion = infer_request.messages[-1]['content']
        stop=identify_stop(last_completion,infer_request.data_dict)
        return stop
    def step(self,
             infer_request,
             response_choice,
             current_turn):
        content = response_choice.message.content
        
        timestamps = _get_clip(content)
 
        data = infer_request.data_dict
        frm_dir = data["frame_dir"]
        frame_files = data["frames"]
        total_frms=data["frame_num"]
        max_frames = 8
        fps = data["fps"]
        question = data["question"]
        options = data.get("options")
        history = data.get("history_frames", [])
        last_round_frm_num = data.get("last_round_frm_num", 16)
        start_second=data.get("start_second",5.0)
        frm_score = _get_score(content,data["round_frame_id"][current_turn-1],fps,start_second)
        try:
            new_secs, new_idx, new_paths = _get_frm_paths(
                frm_dir, frame_files, timestamps, max_frames, total_frms, fps, history
            )
        except:
            new_secs, new_paths= [], []

        data["round_frame_id"].append(new_idx)
        data["history_frames"] = history + [
            s for s in new_secs
        ]
        data["last_round_frm_num"] = len(new_secs)
        extra_data_info={"round_frame_id":data["round_frame_id"],"history_frames":data["history_frames"],"last_round_frm_num":data["last_round_frm_num"]}
        infer_request.data_dict = data
        new_msg = construct_next_message(new_secs, new_paths, question, options)
        if new_paths:
            for path_i in new_paths:
                infer_request.images.append(path_i)
        infer_request.messages.append({
            "role": "user",
            "content": new_msg
        })
        extra_info={"identified_frames_score":frm_score,"new_frames_sec":new_secs,"new_frames_idx":new_idx,"current_turn":current_turn}
        return {'infer_request': infer_request, 'rollout_infos': {f"round{current_turn}":extra_info,"extra_data_info":extra_data_info}}
multi_turns['VideoClipScheduler'] = VideoClipScheduler