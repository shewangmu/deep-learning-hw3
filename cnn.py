import torch
import time
import torch.nn as nn
import matplotlib.pyplot as plt
from loading_data import *
    
class CNN(nn.Module):
    """
    the structure of embedding class is:
        word embeddings --> average pooling --> Linear --> sigmoid
    reveives the dimension of D of the input words features
    """
    def __init__(self, input_dim, C, H, F, H_dot=5):
        """
        - input_dim is the length of dictionary
        - C is embedding size.
        - H is sentence dimension.
        - F is number of feature maps.
        - H_dot is convolution kernel size.
        """
        super(CNN, self).__init__()
        
        self.C = C
        self.H = H
        self.F = F
        self.H_dot = H_dot
        
        self.embed = nn.Embedding(input_dim, C, padding_idx=0)
        
        self.conv = nn.Sequential(
                nn.Conv1d(C, F, H_dot),
                #nn.AvgPool1d(H-H_dot+1)
                nn.MaxPool1d(H-H_dot+1)
                )
        
        self.linear = nn.Sequential(
               nn.ReLU(),
               nn.Linear(F, 2),
               nn.Sigmoid()
        )
    
    def forward(self, x):
        N, H = x.shape
        embed_word = self.embed(x)
        embed_word = embed_word.transpose(1,2)
        
        out = self.conv(embed_word)
        out = out.view(N, self.F)
        out = self.linear(out)
        
        return out

def load_word_dic():
    #loading data
    train_label = []
    train_txt = []
    max_length = 0
    with open('./data/train.txt', 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            train_label.append(int(line[0]))
            max_length = max(max_length, len(line[1:]))
            train_txt.append(' '.join(line[1:]))
    
    dev_label = []
    dev_txt = []
    with open('./data/dev.txt', 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            dev_label.append(int(line[0]))
            max_length = max(max_length, len(line[1:]))
            dev_txt.append(' '.join(line[1:]))
    
    test_label = []
    test_txt = []
    with open('./data/test.txt', 'r') as file:
        for line in file:
            line = line.strip().split(' ')
            test_label.append(int(line[0]))
            max_length = max(max_length, len(line[1:]))
            test_txt.append(' '.join(line[1:]))
    
    #getting vocaburary
    word_dic = {}
    for line in train_txt+dev_txt+test_txt:
        line = line.split(' ')
        for word in line:
            if word not in word_dic:
                word_dic[word] = len(word_dic)
    
    train = [train_txt, train_label]
    dev = [dev_txt, dev_label]
    test = [test_txt, test_label]
    
    return word_dic, train, dev, test, max_length

def loading_tensor():    
    word_dic, train, dev, test, max_length = load_word_dic()
    train_txt, train_label = train
    dev_txt, dev_label = dev
    test_txt, test_label = test
    #forming tensor
    train_label = torch.tensor(train_label)
    dev_label = torch.tensor(dev_label)
    test_label = torch.tensor(test_label)
        
    train_data = torch.zeros([len(train_label),max_length], dtype=torch.long, device='cpu')
    dev_data = torch.zeros([len(dev_label),max_length], dtype=torch.long, device='cpu')
    test_data = torch.zeros([len(test_label),max_length], dtype=torch.long, device='cpu')
        
    for i in range(len(train_label)):
        line = train_txt[i].strip().split(' ')
        j = 0
        for word in line:
            try:
                train_data[i, j] = word_dic[word]+1
                j += 1
            except KeyError:
                j += 1
        
    for i in range(len(dev_label)):
        line = dev_txt[i].strip().split(' ')
        j = 0
        for word in line:
            try:
                dev_data[i][j] = word_dic[word]+1
                j += 1
            except:
                j += 1
    
    for i in range(len(test_label)):
        line = test_txt[i].strip().split(' ')
        j = 0
        for word in line:
            try:
                test_data[i][j] = word_dic[word]+1
                j += 1
            except:
                j += 1
    
    return  word_dic, train_label, train_data, dev_label, dev_data, test_label, test_data


    
def train(traindata, trainlabel, batchsize, net, criterion, optimizer, device):
    N, V = traindata.shape
    train_data = traindata.reshape([int(N/batchsize), batchsize, V])
    train_label = trainlabel.reshape([int(N/batchsize), batchsize])
    
    history = []
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i in range(len(train_label)):
            txt = train_data[i]
            labels = train_label[i]
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            outputs = net(txt)
            #outputs = net(txt, length, mask)
            running_loss = criterion(outputs, labels)
            history.append(running_loss)
            running_loss.backward()
            optimizer.step()
            
            # print statistics
            # running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')
    return history

def test(testlabel, testdata, style):
    correct = 0
    total = len(testlabel)
    with torch.no_grad():
        txt = testdata
        labels = testlabel
        outputs = net(txt)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d %s images: %f %%' % (len(testlabel),style,
        100 * correct / total))
    
if __name__ == '__main__': 
    word_dic, train_label, train_data, dev_label, dev_data, test_label, test_data = loading_tensor()
    index = torch.randperm(len(train_label))
    train_data = train_data[index]
    train_label = train_label[index]
    
    input_dim = len(word_dic) + 1
    H = len(train_data[0])
    #when kernel size = 5, global_avg_pooling
    net = CNN(input_dim, 128, H, 128, 5)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    batchsize = 100
    device = 'cpu'
    
    history = train(train_data, train_label, batchsize, net, criterion, optimizer, device)
    
    test(train_label, train_data, 'train')
    test(dev_label, dev_data, 'dev')
    test(test_label, test_data, 'test')
    
    #when kernel size = 7, global_avg_pooling
    net = CNN(input_dim, 128, H, 128, 7)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    batchsize = 100
    device = 'cpu'
    
    history = train(train_data, train_label, batchsize, net, criterion, optimizer, device)
    
    test(train_label, train_data, 'train')
    test(dev_label, dev_data, 'dev')
    test(test_label, test_data, 'test')
    
    
    
    
    
    
    