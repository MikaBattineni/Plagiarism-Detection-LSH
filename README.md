# Plagiarism Detection via Locality-Sensitive Hashing (LSH)

## Project Overview
This project implements a highly scalable document similarity detection system designed to efficiently compare massive text datasets. Utilizing Locality-Sensitive Hashing (LSH), the system bypasses the computational bottleneck of traditional pairwise comparison, making it viable for large-scale enterprise or academic database auditing. 

## Technical Stack
* **Language:** Python
* **Algorithms:** Locality-Sensitive Hashing (LSH), Min-Hashing, k-gram Shingling
* **Dataset:** 20 Newsgroups Corpus

## Repository Contents
* `lsh_detector.py`: The core Python engine containing the LSH algorithm, signature matrix generation, and similarity threshold logic.
* `Project Report.pdf`: Comprehensive documentation detailing the mathematical foundation of the hashing parameters and system performance metrics.
* `Project_CIS430.pptx.pdf`: A high-level presentation summarizing the project architecture and outcomes.

## Methodology & Results
1. **Data Processing:** Applied k-gram shingling to convert textual documents into robust numerical sets.
2. **Signature Matrix:** Built a Min-Hashing pipeline to compress large sets into manageable signatures while preserving Jaccard similarity.
3. **LSH Implementation:** Divided the signature matrix into strategically tuned bands and rows to identify candidate pairs.
4. **Optimization:** Successfully tuned the hashing bands to drastically minimize false positives, resulting in a highly accurate, production-ready similarity detection system.
