import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error 

import pickle
import tensorflow as tf
import tensorflow_addons as tfa
import nfp
from nfp_extensions import RBFExpansion, CifPreprocessor

from pymatgen.core.structure import Structure
from tqdm import tqdm

# Initialize the preprocessor class.
preprocessor = CifPreprocessor(num_neighbors=12)
preprocessor.from_json('tfrecords/preprocessor.json')


# Load the model    
model = tf.keras.models.load_model(
    'best_model.hdf5',
    custom_objects={**nfp.custom_objects, **{'RBFExpansion': RBFExpansion}})

# Read-in the test set
test = pd.read_csv('tfrecords/test.csv.gz')

# path to POSCARs
poscar_file = lambda x: 'relaxed_hypotheticals/POSCAR_{}'.format(x)
get_crystal = lambda x: Structure.from_file(poscar_file(x), primitive=True)

# Construct features for test set structures
test_dataset = tf.data.Dataset.from_generator(
    lambda: (preprocessor.construct_feature_matrices(get_crystal(id), train=False)
             for id in tqdm(test.structure_id)),
    output_types=preprocessor.output_types,
    output_shapes=preprocessor.output_shapes)\
    .padded_batch(batch_size=32,
                  padded_shapes=preprocessor.padded_shapes(max_sites=256, max_bonds=2048),
                  padding_values=preprocessor.padding_values)


# Make predictions
predictions = model.predict(test_dataset)

test['predicted_energyperatom'] = predictions
test.to_csv('predicted_test.csv', index=False)


# MAE
f = open('mae_test.txt','w')
print(f'Test MAE: {(test.energyperatom - test.predicted_energyperatom.squeeze()).abs().mean():.3f} eV/atom', file=f)
f.close()

