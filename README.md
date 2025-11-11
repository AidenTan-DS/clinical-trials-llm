# 🧠 LLM-Powered Clinical Trial Data Pipeline & Treatment Recommendation System  

**Author:** Xingye (Aiden) Tan  
**Institution:** University of Washington / Penn State University  
**Timeline:** Jun 2024 – May 2025  
**Advisor:** Prof. Le Bao, Department of Statistics, Penn State  

---

## 🩺 Overview  

This project builds a **Large Language Model (LLM)-driven ETL and normalization pipeline** for clinical trial data to enable **rare-cancer treatment recommendations**.  
The system integrates **data extraction, cleaning, normalization, and semantic matching** into an end-to-end workflow—transforming unstructured ClinicalTrials.gov text into a structured, queryable dataset for downstream research.

---

## 🔧 Methods  

### 1. Data Extraction  
- Queried **22 K + ClinicalTrials.gov** studies using **Hugging Chat’s Nous API**.  
- Parsed key metadata fields: *NCT ID, Title, Cancer Type, Treatment Protocol, Phase, Result Summary*.  
- Achieved **93 % successful parsing rate** after iterative prompt refinement.  

### 2. Data Standardization  
- Implemented two-stage normalization:  
  1. **Database Mapping** – cross-referenced extracted names against standard NIH cancer and treatment lists.  
  2. **NLP Validation** – applied fuzzy matching + edit distance + TF-IDF cosine similarity + Word2Vec semantic similarity to unify variant terms.  
- Improved treatment-matching accuracy from **63 % → 95 %**, reducing manual review time by **85 %**.  

### 3. Knowledge Base Construction  
- Combined normalized trials into a structured CSV/SQL dataset for analytics and retrieval.  
- Enabled **phase-based filtering**, **treatment comparison**, and **similarity-driven recommendations**.  

---

## 📊 Results  

| Metric | Before LLM | After LLM + NLP |
|:---|:---:|:---:|
| Parsing Success Rate | 68 % | **93 %** |
| Treatment Matching Accuracy | 63 % | **95 %** |
| Manual Review Time | 100 % | **15 %** (-85 %) |

**Impact:** Provided the first structured rare-cancer trial dataset integrated with AI-based normalization—now supports automated treatment recommendation and phase-specific comparison.  

🏆 **3rd Place — Eberly College Data Science Poster Competition**

---

## 🧰 Tech Stack  

| Category | Tools |
|:---|:---|
| **Programming Languages** | Python (Pandas, NumPy, regex), SQL |
| **LLM & NLP** | Azure OpenAI, Hugging Chat Nous API, TF-IDF, Word2Vec, Fuzzy Matching, Levenshtein Distance |
| **Data Engineering** | CSV → SQL ETL Pipelines, Batch Processing, Logging |
| **Evaluation** | Manual Validation Scripts, Fuzzy Score Reports, Error Sampling |
| **Visualization** | Matplotlib, Seaborn, Tableau (for summary dashboards) |

---

## 📂 Repository Structure  

