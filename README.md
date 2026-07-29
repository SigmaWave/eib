# LLM and Graphs for Financial Texts 


---

## Project Structure

```
project_root/
├── causal_files/                        # go to the folder and follow the README.md instruction
├── Rolling Window/                      # new versions for fixing data leakage issue
│   ├── gat.py  
│   ├── gat_utils.py  
│   ├── data_utils.py
│   ├── datetime_utils.py  
├── components/
│   ├── gat_loss.py                      # Loss functions (BPR, InfoNCE, CE, FindKG, Hybrid)
│   ├── gat_model.py                     # GAT model definition
│   ├── gat.py                           # GAT training and evaluation
│   ├── judgeLLM.py                      # Judge LLM for triplet validation
│   ├── metrics_computator.py            # Triplet quality scoring pipeline
│   ├── run_visualizer.py                # Knowledge graph visualisation
│   ├── triplet_generator.py             # LLM-based triplet extraction
│   ├── tgp.txt                          # Triplet generation prompt template
│   ├── consistency_prompt.txt        
│   ├── coverage_prompt.txt            
│   ├── semantic_validity_prompt.txt   
│   ├── uniqueness_prompt.txt           
│   └── fin_ontology.rdf                 # Financial ontology for entity typing
├── utils/
│   ├── data_utils.py      
│   ├── gat_utils.py                     # Node features, graph packing utilities
│   ├── visualization_utils.py    
│   ├── logger.py                
│   ├── timer.py                 
│   ├── datetime_utils.py        
│   └── pathlib_utils.py        
├── database/                            # FNSPID full dataset. https://huggingface.co/datasets/Zihan1004/FNSPID/tree/main/Stock_news
│   └── nasdaq_semi_data_cleaned.csv     # Semiconductor-focused subset
├── output/                              # Generated outputs (triplets (including last group's), plots)
│   ├── summary_triplets_19_20.csv       
│   ├── summary_triplets_metrics_19_20.csv       
│   ├── gat_results_strict_summary.csv   
│   ├── ...
├── weights/                             # Saved GAT model checkpoints
└── README.md
```
### Remark
Folder **Rolling Window** contains the corrected version addressing the data leakage issue from the previous group. We have preserved the old version in the original folder. Please replace the old files with the new ones using the matching file names.

---


## Step 0 - Setup

### 0.0 Virtual environment setup (optional)

```bash
source venv/bin/activate
```

### 0.1 Install dependencies:
```bash
pip install -r requirements.txt
```

### 0.2 Ollama Setup

On MacOs:
```bash
brew install ollama #downloads ollama
ollama pull qwen2.5:14b #downloads the open souce qwen2.5:14b model
```
## Step 1 — Triplet Extraction

Extracts structured triplets from financial news articles using a locally deployed LLM.

```bash
python3 components/triplet_generator.py \
    --model-name qwen2.5:14b \ #choose the model you want (qwen2.5:32b...)
    --data-type semi_cleaned_data \ #choose your dataset
    --text-column text \ # triplets generated from text column; use '--text-column summary' to switch to generating from summary column
    --start-date 2020-01-01 \  #choose your own start/end date
    --end-date 2021-01-02

# try this
python3 components/triplet_generator.py --model-name qwen2.5:14b --data-type semi_cleaned_data  --text-column summary --start-date 2020-01-03  --end-date 2021-01-03
```


**Output:** `output/triplets_{model}_{dataset}.csv`


## Step 2 — Triplet Quality Evaluation

Runs each triplet through the Judge LLM pipeline, scoring on four dimensions (Coverage, Uniqueness, Semantic Validity, Consistency) and applying three filtering strategies.

```bash
python components/metrics_computator.py  --triplets-path output/triplets_qwen2.5:14b_semi_cleaned_data_summary.csv

python3 components/metrics_computator.py  --triplets-path output/triplets_gpt-5.5_semi_cleaned_data_summary.csv
```


**Output:** `output/triplets_{model}_{dataset}_JudgeLLM_metrics_computation.csv`. Remember to **rename** the output file about what column (text/summary) they are generated from.

The output CSV contains the original triplets plus quality scores and strategy labels. Three filtering strategies are applied:

| Strategy | Description |
|---|---|
| `original` | Raw LLM output, no Judge intervention |
| `fallback` | Judge revisions applied where valid; falls back to original if revision is malformed |
| `strict` | Only triplets explicitly validated or corrected by the Judge are retained |

> **Note:** Step 2 is the most computationally expensive step due to the Judge LLM calls. It can take several hours on large datasets.

### OPTIONAL: compute stats

```bash
python components/analyse_metrics.py output/triplets_claude-sonnet-5_semi_cleaned_data_summary_JudgeLLM_metrics_computation.csv 
```

Outputs stats for the llm as a judge evaluations in stats folder. 

## Step 3 — GAT Training and Evaluation

Trains Graph Attention Network models using a rolling window protocol and evaluates link prediction performance.

```bash
# Run all three strategies
python components/gat.py \
    --data output/triplets_qwen2.5:14b_fnspid_JudgeLLM_metrics_computation.csv \ # it should be matched to your output file.
    --strategy all

# Run a single strategy
python components/gat.py  --data output/triplets_qwen2.5:14b_semi_cleaned_data_summary_JudgeLLM_metrics_computation.csv --strategy strict
```


**What it trains:**

For each strategy, the script trains 10 model configurations (5 loss functions × 2 edge feature settings):

| Loss | Description |
|---|---|
| `findkg` | FindKG margin loss |
| `bpr` | Bayesian Personalised Ranking |
| `ce` | Cross-Entropy |
| `infonce` | InfoNCE contrastive loss |
| `hybrid` | CE + InfoNCE combined |



**Output:**
- `output/gat_results_{strategy}.csv` — MRR, Hits@1/3/10, Seen/Unseen MRR leaderboard under 5 loss functions.
- `weights/{loss}_{edge}_{strategy}/{period}.pt` — model checkpoints per rolling window period


---

## Step 4 — Knowledge Graph Visualisation

Generates interactive monthly HTML visualisations of the knowledge graph, optionally overlaying GAT link prediction scores.

```bash
python components/run_visualizer.py \
    --triplets_path output/triplets_qwen2.5:32b_fnspid_JudgeLLM_metrics_computation.csv \
    --weights_dir weights \
    --output_dir output/kg_visualization
```


**Output structure:**
```
output/kg_visualization/
├── {strategy}/
│   ├── raw_no_prediction/
│   │   └── graph_{period}_raw.html          # Monthly KG without scores
│   └── pred_{loss}_{edge}/
│       └── graph_{period}_scored.html       # Monthly KG with GAT scores overlaid
```


> **Note:** This semester we did not focus on this part. We included samples from the last group.

---

