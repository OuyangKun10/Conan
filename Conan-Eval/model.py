"""
model.py  支持单进程多卡 / 多进程多卡
"""
import os
import torch 
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from PIL import Image
from typing import List, Dict, Optional
from qwen_vl_utils import process_vision_info
import random
import time
from accelerate import Accelerator
BACKOFF_BASE   = 1.0   
BACKOFF_FACTOR = 2.0   
BACKOFF_MAX    = 60.0 
BACKOFF_JITTER = 0.1  
# ====================== Qwen2.5VL ======================
class Qwen2_5_VL:
    def __init__(self,
                 model_path: str,
                 gpu_ids: List[int],          
                 max_new_tokens: int = 128,
                 temperature: float = 0.01,
                 load_in_4bit: bool = False):
        if len(gpu_ids)==1:
            torch.cuda.set_device(int(gpu_ids[0]))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        accelerator = Accelerator()
        self.device = accelerator.device
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if load_in_4bit else None

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True,trust_remote_code=True)
        self.processor.tokenizer.padding_side = 'left'
        if len(gpu_ids)==1:
            self.model = accelerator.prepare(Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).eval().to(self.device))
        else:
            self.model = accelerator.prepare(Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).eval())
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.inference_mode()
    def chat(self,
        message: List[Dict],
        history: Optional[List[Dict]] = None,
        max_retry=3) -> str:
        if history is None:
            history = []

        batch_messages_list = [history + message]

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages_list
        ]
        try:
            image_inputs_batch, video_inputs_batch = process_vision_info(batch_messages_list)
            model_max = self.processor.tokenizer.model_max_length   
            max_new = getattr(self, "max_new_tokens", 512)
            hard_limit = model_max - max_new - 128 
            inputs_batch = self.processor(
                text=texts,
                images=image_inputs_batch,
                videos=video_inputs_batch,
                padding=True,
                truncation=True,
                max_length=hard_limit+max_new,
                return_tensors="pt",
            ).to(self.device)
        except Exception as e:
            print(f"[ERROR] processor failed: {type(e).__name__}: {e}")
        response="None"
        retry_count=0
        while max_retry>0 and response=="None":
            try:
                generated_ids_batch = self.model.generate(**inputs_batch, use_cache=True, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
                generated_ids_trimmed_batch = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
                ]
                predicted_answers_batch = self.processor.batch_decode(
                    generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response=predicted_answers_batch[0]
            except:
                max_retry -= 1
                retry_count += 1
                if max_retry == 0:
                    print(f"generation error for {message}")
                else:
                    sleep_time = min(
                        BACKOFF_BASE * (BACKOFF_FACTOR ** (retry_count - 1)),
                        BACKOFF_MAX
                    )
                    sleep_time *= (1 + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER))
                    print(f"Retry {retry_count}, sleep {sleep_time:.2f}s")
                    time.sleep(sleep_time)
        history.append(message[0])
        history.append({"role": "assistant", "content": response})
        return response, history
# ====================== KimiVL ======================
class KimiVL:
    def __init__(self,
                 model_path: str,
                 gpu_ids: List[int],
                 max_new_tokens: int = 128,
                 temperature: float = 0.2,
                 load_in_4bit: bool = False):
        if len(gpu_ids) == 1:
            torch.cuda.set_device(int(gpu_ids[0]))
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        accelerator = Accelerator()
        self.device = accelerator.device

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if load_in_4bit else None

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if len(gpu_ids)==1:
            self.model = accelerator.prepare(AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).eval().to(self.device))
        else:
            self.model = accelerator.prepare(AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            ).eval())

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    @torch.inference_mode()
    def chat(self,
        message: List[Dict],
        history: Optional[List[Dict]] = None,
        max_retry=3) -> str:
        if history is None:
            history = []
        batch_messages_list = [history + message]
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages_list
        ]
        try:
            image_inputs_batch, video_inputs_batch = process_vision_info(batch_messages_list)
            model_max = self.processor.tokenizer.model_max_length   
            max_new = getattr(self, "max_new_tokens", 512)
            hard_limit = model_max - max_new - 128 
            inputs_batch = self.processor(
                text=texts,
                images=image_inputs_batch,
                videos=video_inputs_batch,
                padding=True,
                truncation=True,
                max_length=hard_limit+max_new,
                return_tensors="pt",
            ).to(self.device)
        except Exception as e:
            print(f"[ERROR] processor failed: {type(e).__name__}: {e}")
        response="None"
        retry_count=0
        while max_retry>0 and response=="None":
            try:
                generated_ids_batch = self.model.generate(**inputs_batch, use_cache=True, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
                generated_ids_trimmed_batch = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
                ]
                predicted_answers_batch = self.processor.batch_decode(
                    generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response=predicted_answers_batch[0]
            except:
                max_retry -= 1
                retry_count += 1
                if max_retry == 0:
                    print(f"generation error for {message}")
                else:
                    sleep_time = min(
                        BACKOFF_BASE * (BACKOFF_FACTOR ** (retry_count - 1)),
                        BACKOFF_MAX
                    )
                    sleep_time *= (1 + random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER))
                    print(f"Retry {retry_count}, sleep {sleep_time:.2f}s")
                    time.sleep(sleep_time)
        history.append(message[0])
        history.append({"role": "assistant", "content": response})
        return response, history