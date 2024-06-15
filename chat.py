# Importing required libraries
import random
import json
import torch
from model import NeuraNet
from nltk_utils import bag_of_words, tokenize

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# opening the file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# loading the saved data
FILE = "data.pth"
data = torch.load(FILE)

# Our model will learn the following parameters given earlier in the training file
# parameters required to create the model
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# creating the model variable
model = NeuraNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)
model.load_state_dict(model_state)
model.eval()  # evaluate the model

# Creating the chat
bot_name = "Jarvis"


# function that get message as a parameters and return response
def get_response(msg):
    # calculate the bag of words
    sentence = tokenize(msg)
    x = bag_of_words(sentence, all_words)

    # reshape
    x = x.reshape(1, x.shape[0])
    # create torch tensor
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Applying softmax and getting probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        # check the intents if they matches
        for intent in intents["intents"]:
            if tag == intent['tag']:
                return random.choice(intent['responses'])  # random choice from the responses array

    return "I do not understand..."



