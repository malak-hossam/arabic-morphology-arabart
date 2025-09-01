import pandas as pd
from morph_maps import morph_tag_ar, declinability_ar
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
import os

# === تحميل البيانات
df = pd.read_csv("D:\\ai\\project\\morphems\\MASAQ.csv", low_memory=False)

# === تحميل قاعدة بيانات CAMeL Tools
db = MorphologyDB.builtin_db()
analyzer = Analyzer(db)

# === استخراج المقاطع
stems = df[df["Morph_Type"] == "Stem"].groupby(["ID", "Word_No"])[["Segmented_Word", "Morph_Tag", "Invariable_Declinable"]].first().reset_index()
prefixes = df[df["Morph_Type"] == "Prefix"].groupby(["ID", "Word_No"])["Segmented_Word"].apply(lambda x: ''.join(x.dropna().astype(str))).reset_index(name="prefix")
suffixes = df[df["Morph_Type"] == "Suffix"].groupby(["ID", "Word_No"])["Segmented_Word"].apply(lambda x: ''.join(x.dropna().astype(str))).reset_index(name="suffix")

# === دمج المقاطع
merged = stems.merge(prefixes, on=["ID", "Word_No"], how="left").merge(suffixes, on=["ID", "Word_No"], how="left")
merged.fillna('', inplace=True)

# === إعادة بناء الكلمة
merged["Full_Word"] = merged["prefix"] + merged["Segmented_Word"] + merged["suffix"]
merged["Morph_Tag"] = merged["Morph_Tag"]
merged["Declinability"] = merged["Invariable_Declinable"]

# === دالة استخراج الجذر باستخدام CAMeL
def extract_root(word):
    try:
        analyses = analyzer.analyze(word)
        for analysis in analyses:
            root = analysis.get("root", "")
            if root and "#" not in root and root != word and len(root) <= len(word):
                return root
    except:
        pass
    return ""

# === إضافة الجذر للبيانات
merged["Root"] = merged["Full_Word"].apply(extract_root)

# === فلترة الكلمات غير المناسبة
filtered = merged[
    (merged["Root"] != "") &
    (~merged["Root"].str.contains("#")) &
    (merged["Root"] != merged["Full_Word"]) &
    (merged["Root"].str.len() <= merged["Full_Word"].str.len())
]

# === توليد الهدف
def generate_target(row):
    word = row["Full_Word"]
    stem = row["Root"]
    morph_tag = row["Morph_Tag"]
    decl = row["Declinability"]

    tag_ar = morph_tag_ar.get(morph_tag, morph_tag)
    decl_ar = declinability_ar.get(decl, "")

    lines = [
        f"الكلمة: {word}",
        f"الصنف الصرفي: {tag_ar}",
        f"الجذر: {stem}"
    ]
    if decl_ar:
        lines.append(f"الحالة: {decl_ar}")

    return "\n".join(lines)

# === إنشاء input و target
filtered["input"] = filtered["Full_Word"]
filtered["target"] = filtered.apply(generate_target, axis=1)

# === حفظ الملف النهائي
final_df = filtered[["input", "target"]].drop_duplicates()
output_path = "D:\\ai\\project\\morphems\\morphological_descriptions.csv"
final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

# === طباعة الملخص
print("✅ تم إنشاء الملف:", output_path)
print("🔢 عدد الجمل:", len(final_df))
print("📁 المسار الكامل:", os.path.abspath(output_path))
