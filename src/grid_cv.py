from itertools import product
import os, sys
import configparser
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from train_model import Trainer

## define hyperparameter list here 24 models
epochs = [5, 10, 15]
batch_size = [16, 32]
l_rate = [0.001, 0.01]
weight_decay = [0.001, 0.01]
param_grid = dict(learning_rate=l_rate, batch_size=batch_size, weight_decay=weight_decay, epochs=epochs)
combinations = list(product(*param_grid.values()))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
print ("Prefix path: ", PREFIX_PATH)

# Read `config.ini` and initialize parameter values
config = configparser.ConfigParser()
config.read(PREFIX_PATH + 'config.ini')

for i,row in enumerate(combinations):
    for fold in range(0,5):
        lr = row[0]
        batch_size = row[1]
        num_epochs = row[3]
        weight_decay = row[2]
        MODEL_NAME = str(i)+"_"+str(fold)+"_model.pt"
        trainer = Trainer(num_epochs, batch_size, lr, weight_decay, MODEL_NAME)
        trainer.split_data(fold)
        trainer.train()