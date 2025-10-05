"""
video_io.py
"""
import decord
import math
from PIL import Image
from typing import List, Tuple
from decord import VideoReader, cpu
import logging
import os
import os.path as osp
import numpy as np
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

decord.bridge.set_bridge("torch")  

def load_video(path: str):
    """
    加载视频并返回 VideoReader 对象和平均帧率。
    """
    try:
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        fps = vr.get_avg_fps()
        return vr, fps
    except decord.DECORDError as e:
        logger.error(f"Failed to load video {path}: {e}")
        raise

def uniform_sample_idx(total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= max_frames:
        return list(range(total_frames))
    step = total_frames / max_frames
    return [int(round(i * step)) for i in range(max_frames)]

def idx_to_timestamp(idx: int, fps: float) -> str:
    sec = idx / fps
    hh = int(sec // 3600)
    mm = int((sec % 3600) // 60)
    ss = int(sec % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"
def save_video_frames(
    video_path: str,
    frame_indices: List[int],
    output_dir: str, 
    frame_name_prefix: str = "", 
    max_len: int = 448
) -> List[Tuple[str, float]]:
    """
    Args:
        video_path (str): 视频文件的路径。
        frame_indices (List[int]): 需要提取的帧的索引列表。
        output_dir (str): 保存提取出的帧图像的目录。
        frame_name_prefix (str): 帧文件名的前缀，例如 "frame_"，最终文件名将是 "frame_00001.jpg"。
        max_len (int): 保持图片长宽比的前提下，最长边不能超过max_len进行缩放。如果为None，则不进行缩放。
    Returns:
        List[Tuple[str, float]]: 一个列表，其中每个元素是一个元组 (frame_path, timestamp_seconds)，
                                   表示保存的帧的路径和其对应的时间戳（秒）。
    """
    if not osp.exists(video_path):
        raise FileNotFoundError(f"视频文件未找到: {video_path}")
    if max_len>0:
        output_dir=output_dir.replace("frames",f"frames_{max_len}")
    video_name=video_path.split('/')[-1].replace(".mp4","")
    frame_dir=osp.join(output_dir,video_name)
    os.makedirs(frame_dir, exist_ok=True)
    extracted_frames_info = []
    target_frame_paths = []
    for index in frame_indices:
        frame_filename = f"{frame_name_prefix}{index:05d}.jpg"
        target_frame_paths.append(osp.join(frame_dir, frame_filename))
    all_frames_exist = np.all([osp.exists(p) for p in target_frame_paths])

    extracted_frames_info = []

    if all_frames_exist:
        vid = VideoReader(video_path,ctx=cpu(0), num_threads=1)
        fps = vid.get_avg_fps()
        for i, index in enumerate(frame_indices):
            timestamp_seconds = idx_to_timestamp(index, fps)
            extracted_frames_info.append((target_frame_paths[i], timestamp_seconds))
        return extracted_frames_info
    else:
        vid = VideoReader(video_path,ctx=cpu(0), num_threads=1)
        total_frames = len(vid)
        fps = vid.get_avg_fps()

        for i, index in enumerate(frame_indices):
            frame_path = target_frame_paths[i] 

            if not (0 <= index < total_frames):
                print(f"警告: 帧索引 {index} 超出视频范围 [0, {total_frames-1}]，已跳过。")
                continue
            if not osp.exists(frame_path):
                frame_data = vid[index]
                if isinstance(frame_data, torch.Tensor):
                    image_array = frame_data.cpu().numpy()
                elif isinstance(frame_data, np.ndarray):
                    image_array = frame_data
                image = Image.fromarray(image_array)
                if max_len is not None and max_len > 0:
                    width, height = image.size
                    if max(width, height) > max_len:
                        if width > height:
                            new_width = max_len
                            new_height = int(height * (max_len / width))
                        else:
                            new_height = max_len
                            new_width = int(width * (max_len / height))
                        image =  image.resize((new_width, new_height), Image.LANCZOS) 
                try:
                    image.save(frame_path)
                except Exception as e:
                    print(f"保存帧 {index} 失败: {e}")
                    continue 
            timestamp_seconds = idx_to_timestamp(index, fps)
            extracted_frames_info.append((frame_path, timestamp_seconds))

    return extracted_frames_info
def sample_frames(video_path: str,
                  frame_indices: List[int]) -> List[Tuple[Image.Image, str]]:
    """
    从视频中采样指定索引的帧，并返回 PIL 图像和时间戳列表。
    会跳过无法解码的坏帧。
    """
    try:
        vr, fps = load_video(video_path)
    except Exception:
        return []

    total_frames = len(vr)
    valid_frame_indices = [i for i in frame_indices if 0 <= i < total_frames]
    
    if not valid_frame_indices:
        logger.warning(f"No valid frame indices found for video: {video_path}")
        return []

    sampled_imgs_with_timestamps: List[Tuple[Image.Image, str]] = []

    if valid_frame_indices:
        frames_tensor = vr.get_batch(valid_frame_indices)
        if hasattr(frames_tensor, 'cpu') and hasattr(frames_tensor, 'numpy'): 
            frames_np = frames_tensor.cpu().numpy()
        else:
            frames_np = frames_tensor

        for i, frame_np in zip(valid_frame_indices, frames_np):
            img = Image.fromarray(frame_np)
            timestamp = idx_to_timestamp(i, fps)
            sampled_imgs_with_timestamps.append((img, timestamp))
        logger.info(f"Successfully batch decoded {len(sampled_imgs_with_timestamps)} frames from {video_path}.")
        return sampled_imgs_with_timestamps
    
    return sampled_imgs_with_timestamps
