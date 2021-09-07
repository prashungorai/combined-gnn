# Steps to Train a Model

1. **Pre-process data**

* Edit the path to training data on line 21 of [preprocess_crystals.py](preprocess_crystals.py), if using your own data 
* Edit the path to structure files (e.g., VASP POSCAR format) on line 25, if using your own crystal structures
* Execute [preprocess_crystals.py](preprocess_crystals.py)
```shell
python preprocess_crytals.py
```
* A folder /tfrecords containing following the files will be created: train.tfrecord.gz, valid.tfrecord.gz, test.csv.gz, preprocessor.json

<br>

2. **Train the model** 

* Execute [train_model.py](train_model.py)
```shell
python train_model.py
```
* A folder /trained_model containing the following files will be created: log.csv, best_model.hdf5
  * log.csv: contains information on epochs, training and validation losses
  * best_model.hdf5: contains model weights

<br>

3. **Predict with the trained model**

* Edit the path to the structure files (e.g., VASP POSCAR format) on line 28 of [run_test.py](run_test.py)
* Execute [run_test.py](run_test.py)
```shell
python run_test.py
```
* A file predicted_test.csv will be created, which will contain predicted energy of the test crystal structures 