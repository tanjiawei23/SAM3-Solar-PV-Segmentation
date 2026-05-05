import os
import numpy as np
from PIL import Image
from ultralytics import SAM
from ultralytics.models.sam import SAM3SemanticPredictor

INPUT_DIR = "./ori_fig"
OUTPUT_DIR = "./output_fig"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = SAM("sam3.pt") 

overrides = dict(
    conf=0.3,       
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=False,        
)
predictor = SAM3SemanticPredictor(overrides=overrides)

text_prompts = ["building roof"] 

image_files = ["1.png", "2.png"]

for img_name in image_files:
    img_path = os.path.join(INPUT_DIR, img_name)
    if not os.path.exists(img_path):
        print(f"跳过不存在的文件: {img_name}")
        continue
        
    print(f"\n正在处理: {img_name}")
    print(f"使用的文本提示词: {text_prompts}")
    
    predictor.set_image(img_path)
    
    results = predictor(
        text=text_prompts,
        save=False
    )
    
    result = results[0]
    
    num_roofs = len(result.masks) if result.masks is not None else 0
    print(f"{num_roofs}")
    
    if num_roofs > 0:
        save_path = os.path.join(OUTPUT_DIR, f"sam3_{img_name}")
        annotated = result.plot(
            boxes=False,
            labels=False,
            probs=False,
            masks=True
        )
        Image.fromarray(annotated).save(save_path)
        print(f"-> 结果已保存至: {save_path}")
        
        mask_path = os.path.join(OUTPUT_DIR, f"sam3_mask_{img_name}")
        combined_mask = np.any(result.masks.data.cpu().numpy(), axis=0)
        height, width = combined_mask.shape
        rgba_mask = np.ones((height, width, 4), dtype=np.uint8) * 255
        rgba_mask[combined_mask] = [255, 0, 0, 77]
        Image.fromarray(rgba_mask).save(mask_path)
        print(f"{mask_path}")
    else:
        print("-> 未检测到屋顶")

