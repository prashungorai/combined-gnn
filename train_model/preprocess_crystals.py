import os
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
import pickle

import nfp

from nfp_extensions import CifPreprocessor
tqdm.pandas()

      
if __name__ == '__main__':
        
    # Read energy data
    data = pd.read_csv('hypothetical_structure_energies.csv')

    # path to POSCARs
    poscar_file = lambda x: 'relaxed_hypotheticals/POSCAR_{}'.format(x)
    poscar_exists = lambda x: os.path.exists(poscar_file(x))
    data['poscar_exists'] = data.structure_id.apply(poscar_exists)
    data = data[data.poscar_exists]


    # Try to parse crystals with pymatgen
    def get_crystal(id):
        try:
            return Structure.from_file(poscar_file(id), primitive=True)
        except Exception:
            return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data['crystal'] = data.structure_id.progress_apply(get_crystal)

    # record parse issues
    data[data.crystal.isna()]['structure_id'].to_csv('problems.csv')
        
    data = data.dropna(subset=['crystal'])
    print(f'{len(data)} crystals after down-selection')


    # Split the data into training and test sets
    train, test = train_test_split(data.composition.unique(), test_size=10, random_state=1)
    train, valid = train_test_split(train, test_size=10, random_state=1)
    
    
    # Initialize the preprocessor class.
    preprocessor = CifPreprocessor(num_neighbors=12)

    def inputs_generator(df, train=True):
        """ This code encodes the preprocessor output (and prediction target) in a 
        tf.Example format that we can use with the tfrecords file format. This just
        allows a more straightforward integration with the tf.data API, and allows us
        to iterate over the entire training set to assign site tokens.
        """
        for i, row in tqdm(df.iterrows(), total=len(df)):
            input_dict = preprocessor.construct_feature_matrices(row.crystal, train=train)
            input_dict['energyperatom'] = float(row.energyperatom)

            features = {key: nfp.serialize_value(val) for key, val in input_dict.items()}
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))

            yield example_proto.SerializeToString()

    # Process the training data, and write the resulting outputs to a tfrecord file
    serialized_train_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(data[data.composition.isin(train)], train=True),
        output_types=tf.string, output_shapes=())

    os.mkdir('tfrecords')

    filename = 'tfrecords/train.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_train_dataset)
    
    # Save the preprocessor data
    preprocessor.to_json('tfrecords/preprocessor.json')

    # Process the validation data
    serialized_valid_dataset = tf.data.Dataset.from_generator(
        lambda: inputs_generator(data[data.composition.isin(valid)], train=False),
        output_types=tf.string, output_shapes=())

    filename = 'tfrecords/valid.tfrecord.gz'
    writer = tf.data.experimental.TFRecordWriter(filename, compression_type='GZIP')
    writer.write(serialized_valid_dataset)
    
    # Save train, valid, and test datasets
    data[data.composition.isin(train)][
        ['composition', 'structure_id', 'energyperatom']].to_csv(
        'tfrecords/train.csv.gz', compression='gzip', index=False)
    data[data.composition.isin(valid)][
        ['composition', 'structure_id', 'energyperatom']].to_csv(
        'tfrecords/valid.csv.gz', compression='gzip', index=False)
    data[data.composition.isin(test)][
        ['composition', 'structure_id', 'energyperatom']].to_csv(
        'tfrecords/test.csv.gz', compression='gzip', index=False)
