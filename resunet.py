import cv2
import torch
import torch
import torch.nn as nn
from utils import convert_image, adjust_learning_rate,load_image_to_tensor
import wandb


name ="RESUNET_MSE_LR0.01_NEP255_LOSSWT1._WAND"
wandb.init(project='debugging-nn',name=name)

torch.manual_seed(123) 


# Network
class batchnorm_relu(nn.Module):
  def __init__(self,in_ch):
    super().__init__()

    self.bn = nn.BatchNorm2d(in_ch)
    self.relu = nn.LeakyReLU(0.1)

  def forward(self,input):
    x=self.bn(input)
    x=self.relu(x)
    return x


class residual_block(nn.Module):
  def __init__(self,in_ch=1,out_ch=1,stride=1):
    super().__init__()

    '''Convolutional layer'''
    self.b1 = batchnorm_relu(in_ch)
    self.c1= nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1,stride = stride)
    self.b2 = batchnorm_relu(out_ch)
    self.c2 = nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,stride=1)

    """Shortcut Connection (Identity Mapping)"""
    self.s = nn.Conv2d(in_ch,out_ch,kernel_size=1,padding=0,stride=stride)

  def forward(self,input):
    x = self.b1(input)
    x = self.c1(x)
    x = self.b2(x)
    x = self.c2(x)
    s = self.s(input)

    skip = x+s
    return skip

class decoder_block(nn.Module):
  def __init__(self,in_ch,out_ch):
    super().__init__()

    # self.upsample = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
    self.upsample = nn.ConvTranspose2d(in_ch,in_ch,kernel_size=2,stride=2,padding=0)
    self.r = residual_block(in_ch+out_ch,out_ch)

  def forward(self,inputs,skip):
    x = self.upsample(inputs)
    x = torch.cat([x,skip],axis=1)
    x = self.r(x)
    return x


class build_resunet(nn.Module):
  def __init__(self):
    super().__init__()

    '''Encoder 1'''
    self.c11 = nn.Conv2d(1,32,kernel_size=3,padding=1)
    self.br1 = batchnorm_relu(32)
    self.c12 = nn.Conv2d(32,32,kernel_size=3, padding=1)
    self.c13 = nn.Conv2d(1,32,kernel_size=1,padding=0)

    '''Encoder 2 and 3'''
    self.r2 = residual_block(32,64,stride=2)
    self.r3 = residual_block(64,128,stride=2)

    '''Bridge'''
    self.r4 = residual_block(128,256,stride=2)

    '''Decoder'''
    self.d1 = decoder_block(256,128)
    self.d2 = decoder_block(128,64)
    self.d3 = decoder_block(64,32)

    '''Output'''
    self.output = nn.Conv2d(32,1,kernel_size=1,padding=0)
    self.leakyrelu =nn.LeakyReLU(0.1)


  def forward(self,input):
    """Encoder 1"""
    x = self.c11(input)
    x=self.br1(x)
    x=self.c12(x)
    s=self.c13(input)
    skip1 = x+s

    '''Encoder 2 and 3'''
    skip2 = self.r2(skip1)
    skip3 = self.r3(skip2)

    '''Bridge '''
    b = self.r4(skip3)

    '''Decoder'''
    d1 = self.d1(b,skip3)
    d2 = self.d2(d1,skip2)
    d3 = self.d3(d2,skip1)

    '''Output'''
    output = self.output(d3)
    output = self.leakyrelu(output)

    # return output
    return {'x':x,
        'skip1':skip1,
        'skip2':skip2,
        'skip3':skip3,
        'bridge':b,
        'd1':d1,
        'd2':d2,
        'd3':d3,
        'output':output
        }


images = load_image_to_tensor()



#training
net = build_resunet()
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