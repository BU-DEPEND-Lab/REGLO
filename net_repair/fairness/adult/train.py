import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = ""

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
RANDOM_SEED = 512


def load_adult_dataset():
    train_csv = tf.keras.utils.get_file(
        'adult.data',
        'https://download.mlcc.google.com/mledu-datasets/adult_census_train.csv'
    )
    test_csv = tf.keras.utils.get_file(
        'adult.test',
        'https://download.mlcc.google.com/mledu-datasets/adult_census_test.csv'
    )
    train_df = pd.read_csv(
        train_csv, names=COLUMNS, sep=r'\s*,\s*',
        engine='python', na_values="?"
    )
    test_df = pd.read_csv(
        test_csv, names=COLUMNS, sep=r'\s*,\s*', skiprows=[0],
        engine='python', na_values="?"
    )
    # Strip trailing periods mistakenly included only in UCI test dataset.
    test_df['income_bracket'] = test_df.income_bracket.str.rstrip('.')

    return train_df, test_df


def pandas_to_numpy(data):
    '''Convert a pandas DataFrame into a Numpy array'''
    # Drop empty rows.
    data = data.dropna(how="any", axis=0)

    # Separate DataFrame into two Numpy arrays.
    labels = np.array(data['income_bracket'] == ">50K")
    features = data.drop('income_bracket', axis=1)
    features = {name: np.array(value) for name, value in features.items()}

    return features, labels


def load_model_and_test(path_and_filename, features, labels):
    model = keras.models.load_model(path_and_filename)
    model.evaluate(x=features, y=labels)


def main():
    train_df, test_df = load_adult_dataset()

    # Create categorical feature columns

    # Since we don't know the full range of possible values with occupation and
    # native_country, we'll use categorical_column_with_hash_bucket() to help
    # map each feature string into an integer ID.
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000)
    native_country = tf.feature_column.categorical_column_with_hash_bucket(
            "native_country", hash_bucket_size=1000)

    # For the remaining categorical features, since we know what the possible
    # values
    # are, we can be more explicit and use
    # categorical_column_with_vocabulary_list()
    gender = tf.feature_column.categorical_column_with_vocabulary_list(
            "gender", ["Female", "Male"])
    race = tf.feature_column.categorical_column_with_vocabulary_list(
            "race", [
                "White", "Asian-Pac-Islander",
                "Amer-Indian-Eskimo", "Other", "Black"
            ])
    education = tf.feature_column.categorical_column_with_vocabulary_list(
            "education", [
                "Bachelors", "HS-grad", "11th", "Masters", "9th",
                "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
                "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
                "Preschool", "12th"
            ])
    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
            "marital_status", [
                "Married-civ-spouse", "Divorced", "Married-spouse-absent",
                "Never-married", "Separated", "Married-AF-spouse", "Widowed"
            ])
    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
            "relationship", [
                "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
                "Other-relative"
            ])
    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
            "workclass", [
                "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
                "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
            ])

    # Create numeric feature columns
    # For Numeric features, we can just call on feature_column.numeric_column()
    # to use its raw value instead of having to create a map between value and
    # ID.
    age = tf.feature_column.numeric_column("age")
    fnlwgt = tf.feature_column.numeric_column("fnlwgt")
    education_num = tf.feature_column.numeric_column("education_num")
    capital_gain = tf.feature_column.numeric_column("capital_gain")
    capital_loss = tf.feature_column.numeric_column("capital_loss")
    hours_per_week = tf.feature_column.numeric_column("hours_per_week")

    age_buckets = tf.feature_column.bucketized_column(
            age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # List of variables, with special handling for gender subgroup.
    variables = [native_country, education, occupation, workclass,
                 relationship, age_buckets]
    subgroup_variables = [gender]
    feature_columns = variables + subgroup_variables

    deep_columns = [
            tf.feature_column.indicator_column(workclass),
            tf.feature_column.indicator_column(education),
            tf.feature_column.indicator_column(age_buckets),
            tf.feature_column.indicator_column(relationship),
            tf.feature_column.embedding_column(native_country, dimension=8),
            tf.feature_column.embedding_column(occupation, dimension=8),
    ]

    # Define Deep Neural Net Model
    # Parameters from form fill-ins
    hidden_units_layer_01 = 128
    hidden_units_layer_02 = 64
    learning_rate = 0.1
    L1_regularization_strength = 0.001
    L2_regularization_strength = 0.001

    tf.random.set_seed(RANDOM_SEED)

    # List of built-in metrics that we'll need to evaluate performance.
    METRICS = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'),
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc'),
    ]

    regularizer = tf.keras.regularizers.l1_l2(
            l1=L1_regularization_strength, l2=L2_regularization_strength)

    model = tf.keras.Sequential([
          layers.DenseFeatures(deep_columns),
          layers.Dense(
              hidden_units_layer_01, activation='relu',
              kernel_regularizer=regularizer, name='fc_1'
          ),
          layers.Dense(
              hidden_units_layer_02, activation='relu',
              kernel_regularizer=regularizer, name='fc_2'
          ),
          layers.Dense(
              1, activation='sigmoid',
              kernel_regularizer=regularizer, name='otuput'
          )
    ])

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=METRICS)

    # Fit Deep Neural Net Model to the Adult Training Dataset

    epochs = 10
    batch_size = 500

    features, labels = pandas_to_numpy(train_df)
    model.fit(x=features, y=labels, epochs=epochs, batch_size=batch_size)

    # Evaluate Deep Neural Net Performance

    features, labels = pandas_to_numpy(test_df)
    model.evaluate(x=features, y=labels)

    # save the trained model
    file_name = "model_adult_{}_{}_lr_{}_epoch_{}_batch_{}".format(
        hidden_units_layer_01, hidden_units_layer_02,
        learning_rate, epochs, batch_size
    )

    model.save('./model/' + file_name)

    # test saved model
    load_model_and_test('./model/' + file_name, features, labels)


if __name__ == '__main__':
    main()
