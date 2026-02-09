I've counted users from the JSON files: train (16,556), dev (4,730), and test (2,366), for a total of 23,652.

## Comparison: TwiBot-20 Implementation vs. Research Paper Model

### TwiBot-20 Model (Your Implementation) vs. Research Paper Model

| Area                  | Your TwiBot-20 Implementation             | Research Paper Model                                      |
| :-------------------- | :------------------------------------- | :-------------------------------------------------------- |
| **Dataset Used**      | TwiBot-20                            | TwiBot-20                                                 |
| **Text Encoder**      | BERT base uncased                      | BERT base uncased                                         |
| **BERT Training Strategy** | Partial fine-tuning (last layer + pooler only) | Full transformer fine-tuning (implied from training setup stability + scheduler + warmup) |
| **Metadata Features Count** | 5 features (followers, following, listed_count, tweets, verified) | 5 features (followers, friends, listed count, statuses count, verified) |
| **Metadata Processing** | StandardScaler                         | Z-score normalization                                     |
| **Metadata Network**  | 5 → 64 → 32                            | 5 → 64 → 32 → 128                                         |
| **Fusion Dimension**  | 768 + 32                               | 768 + 128                                                 |
| **Classification Head** | Direct linear output                   | Dense (256) + Dropout + Sigmoid                           |
| **Loss Function**     | Focal Loss                             | Weighted Binary Cross Entropy                             |
| **Batch Strategy**    | Weighted Random Sampler                | Stratified mini-batching                                  |
| **Regularization**    | Dropout only                           | Dropout + Batch Normalization                             |
| **Scheduler**         | Cosine Annealing                       | Cosine + Linear Warmup                                    |
| **Early Stopping**    | Yes (Patience=3)                 | Used based on Validation F1                               |
| **Training Epochs**   | Up to 20 (with early stopping)                              | 5 epochs + Early stopping                                 |
| **Batch Size**        | 16                                     | 32                                                        |
| **Hardware**          | Not specified                          | RTX 3080 + 32GB RAM                                       |
| **Evaluation Metrics** | Accuracy + F1                          | Accuracy + Precision + Recall + F1 + ROC-AUC              |

### Exact Performance Numbers

**Your TwiBot-20 Results**

| Metric    | Value  |
| :-------- | :----- |
| Accuracy  | 0.6771 |
| F1 Score  | 0.6959 |

**Research Paper Results**

| Metric    | Value   |
| :-------- | :------ |
| Accuracy  | 94.3%   |
| F1 (Bot)  | 0.935   |
| F1 (Human)| 0.950   |
| ROC-AUC   | 0.960   |

(Source: Paper evaluation section and tables)

### Biggest Structural Differences (Most Important)

1️⃣ **Metadata Power Difference**

*   **Paper:**
    *   Uses a significantly larger metadata embedding (128 vs your 32), allowing it to capture more complex relationships between user features.

2️⃣ **Training Stability Stack**

*   **Paper uses a full training stability pipeline:**
    *   **Linear Warmup:** Helps stabilize training in the initial epochs.
    *   **Batch Normalization:** Regularizes the metadata MLP and improves gradient flow.
    *   **Stratified Batching:** Ensures each batch has a representative class distribution, which is more stable than random sampling.
    *   **Weighted BCE:** A standard and stable loss function for imbalanced datasets.

3️⃣ **Classification Head Depth**

*   **Paper uses a deeper decision layer** (an extra Dense layer with 256 neurons) before the final output, which can help in better feature separation and more complex decision-making.

### Simple Reality (Very Important)

Your implementation captures the core idea of the paper but is missing several key architectural and training components that contribute to the performance gap:

*   **Stronger metadata MLP** (128 dim output vs. 32)
*   **Deeper classifier head** (256-neuron dense layer)
*   **Advanced training stability:**
    *   Linear Warmup scheduler
    *   Batch Normalization
    *   Stratified batching strategy