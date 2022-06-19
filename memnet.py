import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import convert_image, adjust_learning_rate,load_image_to_tensor
import wandb


name ="MEMNET_MSE_LR0.01_NEP255_LOSSWT1._WAND"
wandb.init(project='debugging-nn',name=name)

torch.manual_seed(123) 


# Network
class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i+1) for i in range(num_memblock)]
        )

    def forward(self, x):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out1 = self.reconstructor(out)
        out2 = out1 + residual
        
        return {'output':out2,'out1':out1,'out2':out2,'out':out}


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""
    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock+num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)
        
        gate_out = self.gate_unit(torch.cat(xs+ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    x - Relu - Conv - Relu - Conv - x
    """

    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)
        
    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))




#load images
images = load_image_to_tensor()



#training
net = MemNet(1,32,8,8)
print(net)

# loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()

wandb.watch(net,log="all",log_freq=1)

label = images['hr']
input_model = images['lr_2']
lr = 0.0001

optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=lr)

epochs = 255
n_freq = 50
loss_wt = 1.

for i in range(epochs):
  lr = adjust_learning_rate(optimizer,i,lr)
  result = net(input_model)
  output = result['output']
  if i==0:
    for item in result:
        print(item + ' shape',result[item].shape)

  for item in result:
    conv = convert_image(result[item])
    conv_image = wandb.Image(conv, caption=item)      
    wandb.log({item: conv_image})

  loss = loss_fn(label,output)*loss_wt
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  wandb.log({"loss":loss})
  wandb.log({"learning-rate":lr})

  if i % n_freq == 0:
    print(f'range output: {output.min()},{output.max()}')
    print('loss:', loss.item())
    # for p in net.parameters():
    #   print(p.grad*lr)# or whatever other operation
wandb.unwatch(net)
wandb.finish()

torch.save(net, name+'.pth')
# model = torch.load(PATH)
# model.eval()