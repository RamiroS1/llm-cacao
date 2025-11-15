import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "YuchengShi/LLaVA-v1.5-7B-Plant-Leaf-Diseases-Detection"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "What disease does this leaf have?"},
          {"type": "image"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
image_file = "plant-disease/test1.png"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to("cuda", torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))