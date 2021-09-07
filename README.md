## GNN For Predicting Energy of Known and Hypothetical Structures

This package provides the following functionalities: 
* Predict total energy of new crystal strtuctures using pre-trained models
* Train models with custom datasets

### Package Requirements

Following python packages are needed: 

* [TensorFlow 2](https://www.tensorflow.org/install) 
* [TensorFlow Addons](https://www.tensorflow.org/addons/overview) 
* [pymatgen](https://pymatgen.org/installation.html) 
* [scikit-learn](https://scikit-learn.org/stable/install.html) 
* [nfp](https://pypi.org/project/nfp/)

### Usage

* Predict total energy using pre-trained models: `energy_prediction_demo` 
* Train a model using a custom dataset: `train_model`
* Pre-trained models: `pretrained_models`

### Datasets

To reproduce the models, download the following datasets:

* *Total Energy*

  * [Total energy of ICSD structures from NRELMatDB database (materials.nrel.gov)](nrelmatdb_icsd_energies.csv) 
  * [Total energy of hypothetical structures](hypothetical_structure_energies.csv) 

* *Crystal Structures*  
  
  * ICSD crystal structure files cannot be distributed 
  * [Relaxed structures of hypothetical materials](relaxed_hypothetical_structures.tar.gz)

### Authors

The files in this repository are curated and maintained by

* [Shubham Pandey](mailto:shubhampandey[at]mines[dot]edu)
* [Peter St. John](mailto:Peter.STJohn[at]nrel[dot]gov)
* [Prashun Gorai](mailto:pgorai[at]mines[dot]edu)

### Cite
"Predicting Energy and Stability of Known and Hypothetical Crystals Using Graph Neural Networks"

S. Pandey, J. Qu, V. Stevanovic, P. St. John, and P. Gorai, *Patterns* (2021). DOI: [insert](dx.doi.org/10.0.0.0)

### License

This package is released under the MIT License.