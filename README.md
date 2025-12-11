# Data-Selection-for-LLM-Training

This repository evalutes the efficacy of various coreset selection methods in training lightweight language models compared to random subsets of data. Specifically, we train 7-8 million parameter tinyLLM models and evaluate their performance on the SST2 sentiment dataset across pruning ratios spanning from 5% to 40% of the full dataset. 

The coreset selection methods we consider:
- Forgetting Score Analysis
- Greedy Facility Location Selection
- CRUST
- CRAIG
