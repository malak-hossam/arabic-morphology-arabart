# filename: app.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# تحميل الموديل
model_path = "D:\\ai\\project\\morphems\\trained_arabart_morph_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# دالة استخراج الحقول من النص
def extract_field(text, field_name):
    pattern = rf"{field_name}:\s*(.*?)(?=\s+\w+:|$)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""

# دالة تنظيف الجذر: نرجع "N/A" لو فيه حروف إنجليزية أو مش شكل عربي
def clean_root(root):
    return root if re.fullmatch(r"[اأإآء-ي]+", root) else "N/A"

# شكل البيانات المدخلة
class InputData(BaseModel):
    text: str  # ممكن تكون كلمة أو جملة

@app.post("/analyze")
def analyze_text(data: InputData):
    device = model.device
    words = data.text.strip().split()  # تقسيم الجملة لكلمات

    results = []
    for word in words:
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # استخراج وتحسين الحقول
        sarf_type = extract_field(decoded, "الصنف الصرفي")
        state = extract_field(decoded, "الحالة")
        raw_root = extract_field(decoded, "الجذر")
        cleaned_root = clean_root(raw_root)

        result = {
            "الكلمة": word,
            "الصنف الصرفي": sarf_type,
            "الحالة": state,
            "الجذر": cleaned_root
        }
        results.append(result)

    return {"result": results}
