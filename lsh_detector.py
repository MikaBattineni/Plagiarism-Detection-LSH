"""
Document Plagiarism Detector using LSH
CIS 430 Project - Complete Implementation
"""

import re
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
import time

print("=" * 60)
print("DOCUMENT PLAGIARISM DETECTOR USING LSH")
print("=" * 60)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[STEP 1] Loading 20 Newsgroups dataset...")
start_time = time.time()

# Load dataset - using 3 categories for manageable size
dataset = fetch_20newsgroups(
    subset='all',
    categories=['comp.graphics', 'sci.med', 'talk.politics.mideast'],
    remove=('headers', 'footers', 'quotes')
)

documents = dataset.data
print(f"Loaded {len(documents)} documents in {time.time()-start_time:.2f}s")

# =============================================================================
# STEP 2: SHINGLING (Convert documents to sets of hashed shingles)
# =============================================================================
print("\n[STEP 2] Converting documents to shingles...")
start_time = time.time()

def clean_text(text):
    """Clean text: lowercase, remove punctuation."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def get_shingles(text, k=5):
    """Convert document to set of k-character shingles (hashed)."""
    shingles = set()
    cleaned = clean_text(text)
    
    for i in range(len(cleaned) - k + 1):
        shingle = cleaned[i:i+k]
        # Hash to 32-bit integer to save memory
        shingles.add(hash(shingle) & 0xFFFFFFFF)
    
    return shingles

# Create shingle sets for all documents
shingled_docs = [get_shingles(doc) for doc in tqdm(documents, desc="Shingling")]
print(f"Created shingle sets in {time.time()-start_time:.2f}s")
print(f"Example: Doc 0 has {len(shingled_docs[0])} unique shingles")

# =============================================================================
# STEP 3: MINHASHING (Create signatures for each document)
# =============================================================================
print("\n[STEP 3] Creating MinHash signatures...")
start_time = time.time()

# Parameters
NUM_HASHES = 100  # Signature length
MOD_PRIME = 4294967311  # Large prime > 2^32

# Create random hash functions: (a*x + b) % p
np.random.seed(42)
hash_params_a = np.random.randint(1, MOD_PRIME - 1, NUM_HASHES)
hash_params_b = np.random.randint(0, MOD_PRIME - 1, NUM_HASHES)

def compute_minhash_signature(shingle_set):
    """Compute MinHash signature for a document."""
    signature = np.full(NUM_HASHES, np.inf)
    
    for shingle in shingle_set:
        # Apply all hash functions
        hash_values = (hash_params_a * shingle + hash_params_b) % MOD_PRIME
        # Keep minimum values
        signature = np.minimum(signature, hash_values)
    
    return signature

# Build signature matrix (rows=hash functions, cols=documents)
num_docs = len(shingled_docs)
signature_matrix = np.full((NUM_HASHES, num_docs), np.inf)

for i in tqdm(range(num_docs), desc="MinHashing"):
    signature_matrix[:, i] = compute_minhash_signature(shingled_docs[i])

print(f"Created signature matrix {signature_matrix.shape} in {time.time()-start_time:.2f}s")

# =============================================================================
# STEP 4: LSH (Find candidate pairs using banding)
# =============================================================================
print("\n[STEP 4] Applying LSH to find candidate pairs...")
start_time = time.time()

# LSH Parameters
BANDS = 20
ROWS = 5  # BANDS * ROWS = NUM_HASHES (20 * 5 = 100)

candidate_pairs = set()
hash_tables = [dict() for _ in range(BANDS)]

for doc_id in tqdm(range(num_docs), desc="LSH Banding"):
    for band_idx in range(BANDS):
        # Extract band from signature
        start_row = band_idx * ROWS
        end_row = (band_idx + 1) * ROWS
        band_sig = tuple(signature_matrix[start_row:end_row, doc_id])
        
        # Hash the band
        bucket = hash(band_sig)
        
        # Check for collisions (potential duplicates)
        if bucket in hash_tables[band_idx]:
            for other_id in hash_tables[band_idx][bucket]:
                pair = tuple(sorted((doc_id, other_id)))
                candidate_pairs.add(pair)
            hash_tables[band_idx][bucket].add(doc_id)
        else:
            hash_tables[band_idx][bucket] = {doc_id}

total_possible = num_docs * (num_docs - 1) // 2
print(f"LSH completed in {time.time()-start_time:.2f}s")
print(f"Total possible pairs: {total_possible:,}")
print(f"Candidate pairs found: {len(candidate_pairs):,}")
print(f"Reduction: {total_possible / len(candidate_pairs):.1f}x fewer comparisons!")

# =============================================================================
# STEP 5: VERIFICATION (Check actual similarity of candidates)
# =============================================================================
print("\n[STEP 5] Verifying candidate pairs...")
start_time = time.time()

THRESHOLD = 0.8  # 80% similarity = plagiarism

def jaccard_similarity(set1, set2):
    """Calculate true Jaccard similarity."""
    if len(set1) == 0 and len(set2) == 0:
        return 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

true_positives = []
false_positives = []

for doc1, doc2 in tqdm(list(candidate_pairs)[:1000], desc="Verifying"):  # Check first 1000 for speed
    sim = jaccard_similarity(shingled_docs[doc1], shingled_docs[doc2])
    
    if sim >= THRESHOLD:
        true_positives.append((doc1, doc2, sim))
    else:
        false_positives.append((doc1, doc2, sim))

print(f"Verification completed in {time.time()-start_time:.2f}s")

# =============================================================================
# STEP 6: RESULTS
# =============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Dataset: {num_docs} documents from 20 Newsgroups")
print(f"Total possible comparisons: {total_possible:,}")
print(f"LSH candidate pairs: {len(candidate_pairs):,}")
print(f"Speedup: {total_possible / len(candidate_pairs):.1f}x")
print(f"\nPlagiarism Detection (Jaccard ≥ {THRESHOLD}):")
print(f"True Positives: {len(true_positives)}")
print(f"False Positives: {len(false_positives)}")

# Show an example if we found plagiarism
if true_positives:
    print("\n" + "=" * 60)
    print("EXAMPLE OF DETECTED PLAGIARISM")
    print("=" * 60)
    doc1, doc2, sim = true_positives[0]
    print(f"Documents {doc1} and {doc2} are {sim*100:.1f}% similar\n")
    print(f"Document {doc1} (first 300 chars):")
    print(documents[doc1][:300] + "...\n")
    print(f"Document {doc2} (first 300 chars):")
    print(documents[doc2][:300] + "...")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)