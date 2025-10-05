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
Option: {options}
"""
Prompt_temp_round_mc="""
Please identify the new frame scores, perform step-by-step reasoning, and make final action based on the history and new information.
Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
Option: {options}
"""
Prompt_temp_final_mc="""
Please identify the new frame scores, perform step-by-step reasoning, and answer the question based on the history and new information.
You should output an option letter in <answer></answer> tag.
Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
Option: {options}
"""
Prompt_temp_init_gen="""
You are given a generative question, and several video frames with their timestamps.
For each frame/clip, assign a relevance score on a scale of 1 to 5 (where 5 = highly relevant, 3 = medium relevant and 1 = not relevant), and include the medium or high scoring clip(s) within <score></score>.
And you should perform step-by-step reasoning before making final action.
Guidelines for reasoning:
1. Begin by analyzing the question, clarifying what kind of evidence is required.
2. Analyze the relevant frames with high scores that help answer the question.
3. Compare the available evidence across frames, giving a summary.
4. Justify whether the available information is sufficient to answer accurately.
Action:
If not, you should retrieve additional clip(s) and specify them in <clip></clip>, e.g., <score>the scores corresponding to the clips</score><think>your reasoning process</think><clip>00:00:05-00:00:10</clip><answer></answer>.
If yes, you should answer the question with {type_fix} in <answer></answer>, e.g., <score>the scores corresponding to the clips</score><think>your reasoning process</think><clip></clip><answer>C</answer>.

Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
"""
Prompt_temp_round_gen="""
Please identify the new frame scores, perform step-by-step reasoning, and make final action based on the history and new information.
Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
"""
Prompt_temp_final_gen="""
Please identify the new frame scores, perform step-by-step reasoning, and answer the question based on the history and new information.
Output format:
<score>...</score><think>...</think><clip>...</clip><answer>...</answer> 
Question: {question}
Pleae answer the question with {type_fix} in <answer></answer>.
"""
TYPE_TEMPLATE = {
    "multiple choice": "the single option letter (e.g., A, B, C, D, etc.)",
    "numerical": "the numerical value (e.g., 42 or 3.14)",
    "OCR": "text answer",
    "free-form": "text answer",
    "regression": "the numerical value (e.g., 42 or 3.14)",
    }