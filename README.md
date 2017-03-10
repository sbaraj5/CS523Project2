# MadLibs-RNN

Generate answers to MadLibs-ish styled templates using a RNN/LSTM

Mostly reused code from https://github.com/hunkim/word-rnn-tensorflow 

...which is mostly reused code from https://github.com/sherjilozair/char-rnn-tensorflow which was inspired from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- [Tensorflow](http://www.tensorflow.org)

# Basic Usage
To train with default parameters on the tinyshakespeare corpus, run:
```bash
python3 train.py
```

To start the GUI and sample from a trained model
```bash
python3 ui_madlibs.py
```

# Sample output

## MadLibs-RNN

see proj2/sample output/
