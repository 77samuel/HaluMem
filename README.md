# HaluMem Transition Analysis

Empirical analysis of cross-stage error transitions in memory-augmented LLMs using the HaluMem benchmark.

## Finding

Omission errors at the memory update stage predominantly transform into Hallucination errors at QA time (O→H = 72.8% of off-diagonal transitions, 95% CI [62.3%, 81.3%], p = 2.4×10⁻⁵).

## Repository Structure
halumem/

├── data/

│   ├── pilot_slices/

│   ├── generation/

│   └── eval/

├── eval/

│   ├── eval_memzero.py

│   ├── eval_tools.py

│   ├── evaluation.py

│   └── llms.py

├── analysis/

│   └── transition_analysis.py

├── README.md

└── requirements.txt

## Users

| User | Name | Sessions | Transitions |
|------|------|----------|-------------|
| User 0 | Martin Mark | 36 | 46 |
| User 3 | Sarah Garcia | 28 | 28 |
| User 12 | Sharon Brown | 20 | 30 |

## Setup

```bash
pip install -r requirements.txt
```

## Run Analysis

```bash
python analysis/transition_analysis.py
```

## Citation

Coming soon.

## Authors

Samuel Stephen, R. Vignesh  
Karunya Institute of Technology and Sciences, Coimbatore, Tamil Nadu, India
