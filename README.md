# GraphMatcher

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

GraphMatcher is a graph-based ontology matching system that aligns entities from two ontologies using a neural architecture centered on graph attention and neighborhood-aware representations.

The project combines ontology preprocessing, graph construction, and a graph attention network to predict likely alignments between concepts and properties across two RDF/OWL sources.

## Highlights

- Ontology alignment for RDF/OWL-based datasets
- Graph attention network for neighborhood-aware matching
- Support for entity and property matching
- Config-driven dataset and model paths
- Python 3.9-compatible dependency set

## Project goals

This project follows the idea that ontology alignment can benefit from structural context, not just lexical similarity. In practice, the model learns from neighboring concepts and relations to estimate whether two entities represent the same concept across ontologies.

The work was developed for the OAEI conference track and was designed to score strong matches in uncertain reference alignment settings.

## Repository structure

```text
.
├── config.ini
├── requirements.txt
├── datasets/
│   └── conference/
│       ├── ontologies/
│       └── alignments/
├── outputs/
├── saved_models/
├── src/
│   ├── model/
│   ├── preprocessing/
│   ├── train_model.py
│   ├── test_model.py
│   └── project_paths.py
├── tests/
│   └── test_project_paths.py
└── README.md
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python3.9 -m venv .venv39
source .venv39/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Configure the project in `config.ini`:

```ini
[General]
dataset = conference
K = 5
ontology_split = False
max_false_examples = 150000

[Paths]
dataset_folder = datasets
alignment_folder = /alignments/
save_model_path = saved_models/conference.pt
load_model_path = saved_models/conference.pt
output_folder = outputs/

[Parameters]
max_paths = 21
max_pathlen = 8

[Hyperparameters]
lr = 0.001
num_epochs = 5
weight_decay = 0.001
batch_size = 32
```

> The project resolves paths relative to the repository root, so local absolute paths are no longer required.

## Training

Run the training pipeline from the project root:

```bash
python src/train_model.py
```

## Testing

Evaluate alignment between two ontology files:

```bash
python src/test_model.py path/to/source.owl path/to/target.owl
```

## Example alignment output

```xml
<map>
  <Cell>
    <entity1 rdf:resource='http://conference#has_the_last_name'/>
    <entity2 rdf:resource='http://confof#hasSurname'/>
    <relation>=</relation>
    <measure rdf:datatype='http://www.w3.org/2001/XMLSchema#float'>0.972</measure>
  </Cell>
</map>
```

## Notes

- This codebase uses `rdflib` for ontology parsing instead of the legacy `ontospy` package.
- The project is validated for Python 3.9, which is the recommended runtime for the pinned dependency set.
- The path logic was centralized in `src/project_paths.py` so the project behaves consistently across machines.

## Citation

If you use GraphMatcher in your work, please cite:

```bibtex
@inproceedings{efeoglu2022graphmatcher,
  title = {GraphMatcher: A Graph Representation Learning Approach for Ontology Matching},
  author = {Efeoglu, S.},
  booktitle = {Proceedings of the 17th International Workshop on Ontology Matching (OM 2022)},
  year = {2022},
  series = {CEUR Workshop Proceedings},
  note = {Co-located with ISWC 2022}
}
```

Also, the paper is referenced as:

```text
{ISWC 2022}
{\normalfont
\textbf{S. Efeoglu}. 
\textit{GraphMatcher: A Graph Representation Learning Approach for
Ontology Matching}. 
In \textit{Proceedings of the 17th International Workshop on Ontology
Matching (OM 2022)}, co-located with ISWC 2022. 
CEUR Workshop Proceedings.}
```

## References

[1] Iyer, Vivek, Arvind Agarwal, and Harshit Kumar. "VeeAlign: Multifaceted Context Representation Using Dual Attention for Ontology Alignment." Proceedings of EMNLP 2021.

[2] Veličković, Petar, et al. "Graph Attention Networks." arXiv preprint arXiv:1710.10903 (2017).

## Acknowledgements

This project builds on the VeeAlign design and uses a graph attention approach inspired by the GAT paper above.

