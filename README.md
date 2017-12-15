# Year Classification

Classify which year a text was written using multiple methods

## Generate data sets

To recreate the datasets from scratch, including extracting the year and surface text from ```.xml.bz``` files downloaded from Språkbanken, removing unwanted data, shuffling, and sampling, please see the programs under the ```data``` directory. All are runnable and accept the ```--help``` argument.

## Run simple models (sk-learn)

### Requirements
```pip3 install -r requirements.txt```

### Command
To run the default Bayes model on samples with size 1000 characters, run:

```python3 sk_learn <dataset_directory_path> -t bayes -s 1000```

To see additional options, run:

```python3 sk_learn --help```

## Run complex models

### Prerequisites
TensorFlow and Keras has to be installed
See [TensorFlow installation page](https://www.tensorflow.org/install/) and [Keras installation page](https://keras.io/#installation)

### Command
```python3 runeberg/classifier.py [conv|multiconv|lstm]```
