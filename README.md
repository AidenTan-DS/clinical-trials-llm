# 🧠 LLM-Powered Clinical Trial Data Pipeline & Treatment Recommendation System  

**Author:** Xingye (Aiden) Tan  
**Institution:** University of Washington · Penn State University  
**Timeline:** Jun 2024 – May 2025  
**Advisor:** Prof. Le Bao, Department of Statistics, Penn State  

---

## 🩺 Overview  

This project develops a **Large Language Model (LLM)-driven ETL and normalization pipeline** for oncology clinical trial data, enabling **rare-cancer treatment recommendation** and cross-trial comparison.  

The pipeline integrates **data extraction**, **cleaning**, **normalization**, and **semantic matching** into an end-to-end system — transforming unstructured ClinicalTrials.gov text into a structured, queryable knowledge base for downstream research and model training.

---

## 🔧 Methods  

### 1. Data Extraction  
- Queried **22K+ ClinicalTrials.gov** studies using **Hugging Chat’s Nous API**.  
- Parsed structured metadata: *NCT ID, Title, Cancer Type, Treatment Protocol, Phase, and Result Summary*.  
- Achieved a **93% parsing success rate** after iterative prompt engineering and output validation.  

### 2. Data Standardization  
- Built a **two-stage normalization pipeline**:  
  1. **Database Mapping** — cross-referenced extracted names with NIH cancer and treatment ontologies.  
  2. **NLP Validation** — applied Fuzzy Matching, Levenshtein Edit Distance, TF-IDF Cosine Similarity, and Word2Vec Embeddings to unify spelling and synonym variants.  
- Improved **treatment-matching accuracy from 63% → 95%**, reducing manual review time by **85%**.  

### 3. Knowledge Base Construction  
- Merged all normalized trials into a structured **CSV / SQL database** for reproducible analytics.  
- Enabled **phase-based filtering**, **drug similarity search**, and **recommendation-ready retrieval** through standardized entity linking.

---

## 📊 Results  

| Metric | Before LLM | After LLM + NLP |
|:--|:--:|:--:|
| Parsing Success Rate | 68% | **93%** |
| Treatment Matching Accuracy | 63% | **95%** |
| Manual Review Time | 100% | **15%** (↓85%) |

**Impact:**  
Provided the first **AI-normalized rare-cancer clinical trial dataset**, supporting automated treatment recommendation and phase-specific statistical comparison.  

🏆 **3rd Place — Eberly College Data Science Poster Competition**

📄 [**View Poster (PDF)**](./LLM%20POSTER%20XINGYE%20TAN.pdf)  
📁 [**View Summary Results (CSV)**](./results/summary.csv)

---

## 🧰 Tech Stack  

| Category | Tools & Libraries |
|:--|:--|
| **Programming Languages** | Python (Pandas, NumPy, regex), SQL |
| **LLM & NLP** | HuggingChat Nous API, Azure OpenAI, TF-IDF, Word2Vec, FuzzyWuzzy, Levenshtein Distance |
| **Data Engineering** | CSV → SQL ETL Pipelines, Batch Processing, Logging, Evaluation Scripts |
| **Evaluation** | Accuracy Metrics, Fuzzy Similarity Reports, Manual Error Sampling |
| **Visualization** | Matplotlib, Seaborn, Tableau Dashboards |

---

## 👨‍🔬 Author  

**Xingye (Aiden) Tan**  
🎓 M.S. in Data Science @ University of Washington  
📫 [xtan4@uw.edu](mailto:xtan4@uw.edu)  
🌐 [LinkedIn](https://www.linkedin.com/in/xingye-tan-817b7a225)

---

⭐ *If you find this project useful or inspiring, please consider giving it a star!*


