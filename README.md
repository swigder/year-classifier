# Year Classification

Classify which year a text was written using multiple methods

## Run simple models (sk-learn)

### Requirements
```pip3 install -r requirements.txt```

### Command
```python3 sk_learn/__main__.py <dataset_directory_path> -t bayes -s 1000```

For additional options, run:

```python3 sk_learn/__main__.py --help```

## Run complex models

### Prerequisites
TensorFlow and Keras has to be installed
See [TensorFlow installation page](https://www.tensorflow.org/install/) and [Keras installation page](https://keras.io/#installation)

### Command
```python3 runeberg/classifier.py [conv|multiconv|lstm]```
