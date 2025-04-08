from unsloth import FastLanguageModel
import torch
import os
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer  #一种强化学习优化方法
import torch.distributed as dist

#这个模型是Meta的Llama3.1 8B指令模型，使用了LoRA和4bit量化，​​ 对 ​​Llama-3.1-8B-Instruct​​ 模型进行微调
#使用GRPO（一种强化学习优化方法），训练的数据集是GSM8K（一个数学推理数据集），目标是让模型学会按照特定格式（XML标签）回答数学推理问题（GSM8K数据集）
#定义使用了多种奖励函数来评估模型的输出质量，包括正确性、整数格式、XML格式等。

#-------------------------define model-------------------------
max_seq_length = 1024 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

#-------------------------dataset pre-------------------------
# Load and prep dataset
#定义的​​结构化指令模板​​，用于强制模型按照特定格式生成回答
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()


#-------------------------定义了很多个奖励函数，用于GRPO强化学习训练-------------------------
# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    #print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]



#-------------------------define trainer-------------------------
#RTX 40系列显卡（如你的RTX 4090）的NCCL库默认尝试使用P2P（点对点）和IB（InfiniBand）高速通信，但消费级显卡不支持这些功能，需手动禁用
os.environ["NCCL_P2P_DISABLE"] = "1" # 禁用点对点通信
os.environ["NCCL_IB_DISABLE"] = "1"  # 禁用InfiniBand
torch.cuda.set_device(0) # 强制单卡训练
max_prompt_length = 256
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    local_rank=0,  # 强制单卡训练
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 150,
    save_steps = 150,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases， # 禁用Weights & Biases等日志
    output_dir = "Unsloth/outputs_example",  # Save model to this directory；这里设置保存的路径
)
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# 在程序结束时调用
if dist.is_initialized():
    dist.destroy_process_group()

#------------------------------------1. 训练完成后，保存模型的LoRA参数部分------------------------------------
#关于训练结果会保存在你设置的路径下：
# Unsloth/outputs_example/
# ├── adapter_config.json          # LoRA结构配置（rank/alpha等）
# ├── adapter_model.safetensors    # 实际的LoRA权重（核心文件）
# ├── optimizer.pt                 # 优化器状态（用于恢复训练）
# ├── trainer_state.json           # 训练步数/epoch等元数据
# └── ...（其他tokenizer/训练配置）

#在trainer中设置训练后的模型保存路径等同于手动设置：model.save_lora("grpo_saved_lora")
# grpo_saved_lora/
# ├── adapter_config.json          # LoRA配置
# └── adapter_model.safetensors    # LoRA权重



#------------------------------------3. 训练完成后，想保存LoRA参数，但是推理的时候跟1一样还是得加base model参数再推理------------------------------------
# 只保存你训练的小部分参数（就像只保存对原模型的"修改笔记"）
# 需要配合原始模型使用，文件非常小（通常几十MB）
# model.save_pretrained_merged("my_lora", tokenizer, save_method="lora")



#------------------------------------2. 训练完成后，想保存完整的模型，即base model参数 + LoRA参数合成一个整体的版本，方便部署------------------------------------
# # 训练用的4-bit → 转换为16-bit便于推理
# model.save_pretrained_merged("my_model", tokenizer, save_method="merged_16bit")


#------------------------------------4. 训练完成后，想保存完整的模型，即base model参数 + LoRA参数合成一个整体的版本，方便部署，且想再edge设备部署------------------------------------
# GGUF格式（适合手机/树莓派等设备）​​
# 专门为llama.cpp优化的格式，能在低配设备运行
# 有多种压缩等级可选：
# q8_0：快速转换，质量较好
# q4_k_m：推荐选项，体积小质量佳
# f16：最高质量（但文件很大）
# # 示例：生成手机能用的模型
# model.save_pretrained_gguf("my_gguf", tokenizer, quantization_method="q4_k_m")