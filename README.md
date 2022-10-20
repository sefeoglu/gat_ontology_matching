# GraphMatcher
The GraphMatcher aims to find the correspondes between two ontologies and outputs the possible alignments between them.

The GraphMatcher leverages Graph Attention Network[2] in its neural network structure.
The project leverages a new neighborhood aggregation algorithm, so it examines contribution of neighboring terms which have not been used in the previous matchers before.

## Set up
* 1.) install requirements
``` pip install -r requirements.txt```

* 2.) set the parameters in the config.ini
````
[General]
dataset =               ------> name of a dataset e.g., conference.
K =                     ------> the parameter for K fold cross-validation
ontology_split =        ------> True/False
max_false_examples =    ------>

[Paths]
dataset_folder =        ------> a path to the ontologies
alignment_folder =      ------> a path to the reference alignments
save_model_path =       ------> save the model to the path
load_model_path =       ------> model path
output_folder =         ------> The output folder for the alignments

[Parameters]
max_paths =             ------>
max_pathlen =           ------> ( number of neighboring concepts' types: Equivalent class, subclass of(general to specific or specific to general(2))...
[Hyperparameters] 

lr =                    ------> learning rate
num_epochs =            ------> number of epochs
weight_decay =          ------> Weight decay
batch_size =            ------> Batch Size (8/16/32)

````

* 3.) train the model 
```` 
python src/train_model.py

````
* 4.) test the model
````
python src/test_model.py ${source.rdf} ${target.rdf}
````

* 5.) evaluate the model with the MELT

## Project Definition

Note: The codes in train_model.py and test_model.py are partially based on the VeeAlign[2] project with the permission of its main author. I would like to thank the main author.

## References:
[1] 
````
@inproceedings{iyer-etal-2021-veealign,
    title = "{V}ee{A}lign: Multifaceted Context Representation Using Dual Attention for Ontology Alignment",
    author = "Iyer, Vivek  and
      Agarwal, Arvind  and
      Kumar, Harshit",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.842",
    doi = "10.18653/v1/2021.emnlp-main.842",
    pages = "10780--10792",
   }
````
[2]
````
@misc{https://doi.org/10.48550/arxiv.1710.10903,
  title = {Graph Attention Networks},
  author = {Veličković, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Liò, Pietro and Bengio, Yoshua},
  keywords = {Machine Learning (stat.ML), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Social and Information Networks (cs.SI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  url = {https://arxiv.org/abs/1710.10903},
  publisher = {arXiv},
  doi = {10.48550/ARXIV.1710.10903},
  year = {2017},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

````
