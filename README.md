# 🧠 LLM-Powered Clinical Trial Data Pipeline & Treatment Recommendation System  

**Author:** Xingye (Aiden) Tan  
**Institution:** University of Washington · Penn State University  
**Timeline:** Jun 2024 – May 2025  
**Advisor:** Prof. Le Bao, Department of Statistics, Penn State  

---

## 🩺 Overview  

This project is part of a broader effort to develop a **rare-cancer treatment recommendation system** that learns from historical oncology trial data.  
To make such a system possible, we first needed to **rapidly collect** and **standardize** large-scale clinical trial information.  
However, most trial records on [ClinicalTrials.gov](https://clinicaltrials.gov) are **unstructured, inconsistently formatted, and difficult to compare**, making manual extraction slow and unreliable.

To overcome these challenges, we built a **Large Language Model (LLM)-powered data pipeline** that automates the entire process of:
1. **Rapid data acquisition** — queried and summarized **22K+ clinical trials** via the **HuggingChat Nous API**, enabling large-scale retrieval in hours instead of weeks.  
2. **Normalization and cleaning** — standardized cancer and treatment names using multi-stage NLP (Fuzzy Matching, Edit Distance, TF-IDF Cosine Similarity, and Word2Vec Embeddings).  
3. **Knowledge base construction** — integrated cleaned data into a structured, queryable database for cross-trial comparison and recommendation readiness.

By transforming raw ClinicalTrials.gov text into a **consistent, machine-readable knowledge base**, the pipeline lays the foundation for a **rare-cancer treatment recommendation system**—allowing insights from common cancers to be effectively transferred to rare ones.

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


