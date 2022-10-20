# GraphMatcher
The Graph Matcher leverages Graph Attention Network in its neural network structure.
The project extended the VeeAlign projects with Graph Attention Networks as well as compares both ml-based ontology matchers' performances.

## Set up
* 1.) install requirements
``` pip install -r requirements.txt```

* 2.) set the parameters in the config.ini
````
[General]
dataset =               ------>
K =                     ------>
ontology_split =        ------>
max_false_examples =    ------>

[Paths]
dataset_folder =        ------>
alignment_folder =      ------>
save_model_path =       ------>
load_model_path =       ------>
output_folder =         ------>

[Parameters]
max_paths =             ------>
max_pathlen =           ------>
[Hyperparameters] 

lr =                    ------>
num_epochs =            ------>
weight_decay =          ------>
batch_size =            ------>

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


## References:
[1] 
````

````
[2]
````

````
Note: The codes in train_model.py and test_model.py are partially based on the VeeAlign project with the permission of its main author. I would like to thank the main author.