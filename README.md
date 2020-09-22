# covidbot(Contextual Chatbot for covid-19)

here you can find the data and the compete training and impelementation in pytorch 

Simple chatbot implementation with PyTorch.

- The implementation should be easy to follow for beginners and provide a basic understanding of chatbots.
- The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.
- Customization for your own use case is super easy. Just modify intents.json with possible patterns and responses and re-run the training .

## Install PyTorch and dependencies
For Installation of PyTorch see official website.
- https://pytorch.org/

You also need nltk:
 ```console
pip install nltk
 ```
 If you get an error during the first run, you also need to install `nltk.tokenize.punkt`:
Run this once in your terminal:
 ```console
$ python
>>> import nltk
>>> nltk.download('punkt')
```

## Usage
Run
```console
python train.py
```
This will dump `data.pth` file. And then run
```console
python chatup.py
```

 
