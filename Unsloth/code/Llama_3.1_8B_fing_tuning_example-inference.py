from unsloth import FastLanguageModel
from vllm import SamplingParams
from peft import PeftModel
import torch
import os
import torch.distributed as dist
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 按物理顺序分配设备号
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # 只使用 GPU 0（物理第一个）
os.environ["NCCL_P2P_DISABLE"] = "1" # 禁用点对点通信
os.environ["NCCL_IB_DISABLE"] = "1"  # 禁用InfiniBand



# 1. 加载基础模型（与训练时相同配置）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",  # 基础模型名称
    max_seq_length = 1024,
    load_in_4bit = True,  # 保持4-bit量化
    fast_inference = True,  # 启用vLLM加速
)

# 2. 加载训练好的LoRA适配器，这种方式是永久加载，后面的推理都用的是加载后的
# model = PeftModel.from_pretrained(model, "Unsloth/outputs_example/checkpoint-150")     #这样用PEFT库的api这样用也能兼容
model.load_adapter("/home2/wzc/Unsloth_learning/Unsloth/outputs_example/checkpoint-150") #用Unsloth的api加载LoRA适配器,这个必须是绝对路径

#查看是否正确加载
# print(model.active_adapters)  # 应显示已加载的适配器名称
# print(model.peft_config)       # 查看LoRA配置

# 3. 定义系统提示（与训练时一致）
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# 4. 准备输入（模仿训练时的格式）
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "There are 10 apples in total which need to be handout for 5 people, how many apples can be get for each person?"}  # 替换为你的问题
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
)

# 5. 配置生成参数
sampling_params = SamplingParams(
    temperature = 0.7,  # 控制随机性（0=确定，1=随机）
    top_p = 0.9,        # 核采样阈值
    max_tokens = 256,   # 生成的最大token数
)

# 6. 生成回答
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
)[0].outputs[0].text

print("Model's answer:\n", output)

# 在程序结束时调用
if dist.is_initialized():
    dist.destroy_process_group()