from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image

# 配置路径
LOCAL_MODEL_PATH = r"D:\\models\\Qwen3-VL-2B-Thinking"
TEST_IMAGE_PATH = r"F:\bishe\Qwen3-vl-2b-demo\data\test.png"

# 智能选择数据类型
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 加载模型和处理器
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    LOCAL_MODEL_PATH,
    dtype=dtype,  
    device_map="auto",
    trust_remote_code=True
).eval()

# 加载图片
image = Image.open(TEST_IMAGE_PATH).convert("RGB")

# 构造消息
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image in detail."}
    ]
}]

# 准备输入
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# 直接使用模型自带的 generation_config
generation_config = model.generation_config
generation_config.max_new_tokens = 256  # 仅覆盖需要修改的参数
generation_config.do_sample = False     # 确定性输出
generation_config.temperature = 0.0     # 贪婪解码

# 生成输出 (只传 config)
outputs = model.generate(
    **inputs,
    generation_config=generation_config
)

# 解码结果
result = processor.decode(
    outputs[0][inputs.input_ids.shape[1]:],  # 精确截断输入部分
    skip_special_tokens=True
).strip()

print("模型输出：", result)