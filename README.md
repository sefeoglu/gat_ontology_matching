# GraphMatcher
The Graph Matcher leverages Graph Attention Network in its neural network structure.
The project extended the VeeAlign projects with Graph Attention Networks as well as compares both ml-based ontology matchers' performances.

### Set up
* 1.) install requirements
``` pip install -r requirements.txt```

* 2.) set the parameters in the config.ini
````
[General]
dataset = conference(or biodiv)
quick_mode = False(True for training with pretrained embeddings)
K = 5
ontology_split = False
max_false_examples = 150000

[Paths]

dataset_folder = 

alignment_folder = /alignments/
save_model_path = congerence.pt
load_model_path = conference.pt
embedding_cache_path = saved_models/cached_conference_embeddings.pkl
output_folder = outputs/

````

* 3.) train the model 
```` 
python train_model.py

````
* 4.) test the model
````
python test_model.py ${source.rdf} ${target.rdf}
````

* 5.) evaluate the model with the MELT

Note: The codes in train_model.py and test_model.py are partially based on the VeeAlign project with the permission of its main author. I would like to thank the main author.
