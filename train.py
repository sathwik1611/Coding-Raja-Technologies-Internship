# importing required libraries
import json
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuraNet

# Open the json file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# creating list for words, tags and xy to hold patterns and tags
all_words = []
tags = []
xy = []

# get the tags from json file
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    # to get the patterns form the json file
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Training data
x_train = []
y_train = []

# loop over xy array
for (pattern_sentence, tag) in xy:
    # x data
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    # y data
    label = tags.index(tag)
    y_train.append(label)

# done with training data
x_train = np.array(x_train)
x_train = np.array(x_train)

# creating hyperparameter
batch_size = 8
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


# creating new dataset
class Chatdataset(Dataset):
    # x and y data
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # length of x data
    def __len__(self):
        return self.n_samples


# Creating dataset variable
dataset = Chatdataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating the model variable
model = NeuraNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()  # empty the gradients
        loss.backward()  # calculate backpropagation
        optimizer.step()

    # done with training loop
    # epochs and loss
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, loss = {loss.item()}')

# final loss
print(f'Final Loss: {loss.item()}')

# save the data
data  = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

# file to save the data
FILE = "data.pth"
torch.save(data, FILE)


print(f'Training Complete and file saved to {FILE}')
