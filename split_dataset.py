import pandas as pd
from sklearn.model_selection import train_test_split
import os

# تحميل الداتا النهائية
df = pd.read_csv("D:/ai/project/morphems/morphological_descriptions_cleaned.csv")

# حذف أي صفوف ناقصة أو مكررة للاحتياط
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# تقسيم Train / Temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# تقسيم Temp إلى Validation / Test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# حفظ الملفات
save_path = "D:/ai/project/morphems/data_ready"
os.makedirs(save_path, exist_ok=True)

train_df.to_csv(os.path.join(save_path, "train.csv"), index=False, encoding="utf-8-sig")
val_df.to_csv(os.path.join(save_path, "val.csv"), index=False, encoding="utf-8-sig")
test_df.to_csv("D:/ai/project/morphems/morph_test.csv", index=False, encoding="utf-8-sig")

print("✅ تم التقسيم والحفظ بنجاح:")
print("🟢 Train:", len(train_df))
print("🟡 Validation:", len(val_df))
print("🔵 Test:", len(test_df))
