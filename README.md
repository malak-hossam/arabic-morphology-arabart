# Arabic Morphology Model (AraBART)

## Project Description
This project focuses on building an **Arabic Morphology Model** using the **AraBART** architecture fine-tuned on the **MASAQ dataset**.  
The main objective is to analyze Arabic text, extract key morphological features such as **root, part-of-speech, and case**, and deploy the model via a **FastAPI microservice** for real-time usage.

---

## Dataset
The dataset used is **MASAQ (Morphologically-Analyzed and Syntactically-Annotated Quran)**, which provides a rich source of Arabic text annotated with detailed morphological and syntactic information.

### Key Columns:
- **Word**: Arabic token.  
- **Root**: Extracted triliteral/quadriliteral root.  
- **POS (Part of Speech)**: Word type (noun, verb, particle, etc.).  
- **Case**: Grammatical case (nominative, accusative, etc.).  
- **Morphological Features**: Gender, number, definiteness, etc.  

---

## Objectives
1. Prepare and clean the MASAQ dataset for modeling.  
2. Split the dataset into training, validation, and test sets.  
3. Fine-tune **AraBART** for morphological feature extraction.  
4. Deploy the trained model as a FastAPI service.  
5. Provide inference results with high accuracy (achieved ~82%).  

---

## Project Structure
The project consists of the following files:

- **`prepare_dataset.py`** → Cleans and prepares the raw MASAQ dataset.  
- **`split_dataset.py`** → Splits data into training, validation, and test sets.  
- **`train_model.py`** → Fine-tunes AraBART on the cleaned dataset.  
- **`main.py`** → FastAPI application for real-time inference.  
- **`requirements.txt`** → List of required dependencies.  
- **`data/`** → Folder containing the dataset files.  

---

## Analysis Overview
### Data Preparation
- Cleaned raw MASAQ data and extracted morphological fields.  
- Normalized root forms for consistency.  
- Handled missing and inconsistent annotations.  

### Dataset Splitting
- Applied an **80/10/10 split** for training, validation, and testing.  
- Ensured balanced distribution of key morphological categories.  

### Model Training
- Fine-tuned **AraBART** using the Hugging Face `transformers` library.  
- Optimized with **AdamW optimizer** and **early stopping**.

### Results
- Achieved 82% accuracy on morphological feature extraction.
- Successfully extracted root, part-of-speech, and case.
- Validated using multiple evaluation metrics across train/val/test sets.
