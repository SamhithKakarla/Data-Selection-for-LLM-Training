# --------------------------------------------------------------------------------------------------------------------
# Use this file to select the coreset through lazy greedy
# e.g. python greedy_lazy_greedy.py --inputDir embeddings.npz  --fraction 0.30 --outputDir dataset_0.30.jsonl
# --------------------------------------------------------------------------------------------------------------------

import numpy as np
from heapq import heappush, heappop
from sklearn.metrics.pairwise import cosine_similarity
import json
import argparse

# ---------------------------------------------------------
# Lazy Greedy Facility Location
# ---------------------------------------------------------
def lazy_greedy_facility_location(embeddings, k):
    """
    Parameters:
        embeddings (np.ndarray): Input embedding matrix (N x D)
        k (int): Number of points to select

    Returns:
        list[int]: Indices of selected coreset points
    """
    N = embeddings.shape[0]

    # Distance matrix: (1 - cosine similarity) → higher = less similar
    sim = 1 - cosine_similarity(embeddings)

    best_sim = np.zeros(N)

    selected = []
    selected_set = set()

    # Initial gains (sum over each column)
    initial_gains = sim.sum(axis=0)

    # Max-heap of candidate gains
    heap = []
    for j in range(N):
        heappush(heap, (-initial_gains[j], j))

    # Greedy selection
    for i in range(min(k, N)):
        print(f"Selecting item {i}/{k}")

        while True:
            _, j = heappop(heap)

            if j in selected_set:
                continue

            improved = np.maximum(best_sim, sim[:, j])
            true_gain = improved.sum() - best_sim.sum()

            next_best_est = -heap[0][0] if heap else -1

            if true_gain < next_best_est:
                heappush(heap, (-true_gain, j))
                continue

            selected.append(j)
            selected_set.add(j)
            best_sim = improved
            break

    return selected


def main():
    parser = argparse.ArgumentParser(description="Coreset selection using Lazy Greedy Facility Location")
    parser.add_argument("--inputDir", type=str, required=True, help="Path to NPZ embeddings file") 
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction of samples to select")
    parser.add_argument("--outputDir", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    # Load embeddings, sentences, labels
    data = np.load(args.inputDir, allow_pickle=True)
    embeddings = data["embeddings"]
    sentences = data["sentences"]
    labels = data["labels"]

    # Compute k
    k = int(len(embeddings) * args.fraction)
    print(f"Selecting {k} samples out of {len(embeddings)} total embeddings")

    # Run coreset selection
    indices = lazy_greedy_facility_location(embeddings, k)

    # Extract selected samples
    coreset_sentences = [sentences[i] for i in indices]
    coreset_labels = [int(labels[i]) for i in indices]

    # Save to JSONL file
    with open(args.outputDir, "w") as f:
        for text, label in zip(coreset_sentences, coreset_labels):
            f.write(json.dumps({"text": text, "label": label}) + "\n")

    print(f"Coreset successfully saved to {args.outputDir}")


if __name__ == "__main__":
    main()
