import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from PIL import Image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

train_iters = 5000
zoom = 1 #i.e. GPT is not a blurry jpeg of the internet


class config:
    input_size=2
    hidden_size=100
    output_size=3
    lr = 0.001
    nonlinearity = nn.ReLU()

class MLP(nn.Module): 
  def __init__(self, config):
    super().__init__()
    self.nonlinearity = config.nonlinearity

    self.fc_in = nn.Linear(config.input_size, config.hidden_size)

    self.hidden = nn.Linear(config.hidden_size, config.hidden_size)

    self.fc_out = nn.Linear(config.hidden_size, config.output_size)

    self.lossf = nn.MSELoss()

    self.loss = 0
    self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)

  def forward(self, x):

    out = self.fc_in(x)
    out = self.nonlinearity(out)

    out = self.hidden(out)
    out = self.nonlinearity(out)

    return self.fc_out(out) 

  def L(self):
    self.loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    return self.loss

im = torch.tensor(cv2.cvtColor(cv2.imread('input.png'), cv2.COLOR_BGR2RGB)[np.newaxis,...]/255).cuda()-0.5

coords = torch.tensor([[(i/im.shape[1],j/im.shape[2]) for j in range(im.shape[2])] for i in range(im.shape[1])]).to(device).float()-0.5

net = MLP(config).to(device)

for i in tqdm(range(train_iters)):
    pred = net(coords)   
    net.loss = net.lossf(pred,im[0].float())
    net.L()


zoom = 1 #i.e. GPT is not a blurry jpeg of the internet

pred = net(coords/zoom)  
pred = ((pred+0.5)*255)
im = Image.fromarray(pred.cpu().detach().numpy().astype('uint8'))
im.show()