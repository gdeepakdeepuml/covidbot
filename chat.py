import random 
import json
import torch 
from model import NeuralNet
from nltk_util import tokenization,stemm,bag_of_word

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('intentscovid.json','r') as f:
    intents = json.load(f)

File = 'datacovid.pth'
data = torch.load(File)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
model_state = data['model_state']
all_words = data['all_words']
tags = data ['tags']

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = 'covidbot'
#print("do you help to protect from covid!! let's chat! type 'quit' to get out !! ")

def chatpost(msg):
#while True:
    #sentence = input("You: ")
    print(msg)
    sentence = msg
    if sentence.lower() == "quit":
        print("its work's") 

    sentence = tokenization(sentence)
    x = bag_of_word(sentence,all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output,dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.65:
        for intent in intents['intents']:
            if tag == intent['tag']:
                responses = intent['responses']
                
        return (random.choice(responses))
    else:
        answer="I don't understand can you be more clear please!"
        return (answer)


