# RetiXfer: Transfer Elite Knowledge Sparks for Retinal Vision-Language Pretraining

## :sparkles: Overview

**RetiXfer** is a knowledge-driven vision-language pretraining (VLP) framework. While existing VLP models often require massive image-text datasets, RetiXfer leverages **"Elite Knowledge Sparks"** to maximize the value of categorical data under limited paired data supervision.

### Highlights:
* **Dual Knowledge Sparks:** Integrates **Expert Knowledge** for clinical prior enrichment and **Exemplar Knowledge** for evidence-based modeling.
* **Hybrid Expert Knowledge Injection:** Combines semantic-level and appearance-level matching to facilitate effective expert knowledge transfer.
* **Hierarchical Exemplar Knowledge Reference:** Organizes exemplar knowledge into structured references to support evidence-aware knowledge transfer.

> Additional code and documentation will be released upon acceptance.

## :wrench: Installation

```bash
# 1. Clone the repository
git clone https://github.com/lxirich/RetiXfer.git
cd RetiXfer

# 2. Create and activate the environment
conda create -n retixfer python=3.10 -y
conda activate retixfer

# 3. Install requirements
pip install -r requirements.txt
```

## 📂 Data Setup
Please organize your datasets as follows:

```bash
local_data/
└── dataframes/
    ├── pretraining/             # Pre-training categorical data
    ├── expert_knowledge.csv     # Expert image-text knowledge 
    └── exemplar_knowledge.pkl   # Hierarchical exemplar knowledge 
```

## ⚙️ Usage Pipeline
### 1. Pre-training
Run the knowledge-driven pre-training with HEKI and HEKR modules:
```bash
python train.py \
    --batch_size 48 \
    --lr 1e-4 \
    --expert_knowledge_path ./local_data/dataframes/expert_knowledge.csv \
    --exemplar_knowledge_path ./local_data/dataframes/exemplar_knowledge.pkl \
    --output_dir ./checkpoints/retixfer_pretrain/
```

### 2. Evaluation
Evaluate the model on benchmarks:
```bash
python evaluate.py \
    --task zero_shot \
    --dataset REFUGE \
    --weights_path ./checkpoints/retixfer_pretrain/model.pth \
```
