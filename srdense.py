import cv2
import torch
import torch
import torch.nn as nn


from utils import convert_image, adjust_learning_rate,load_image_to_tensor

import wandb

name ="DENSE_MSE_LR0.01_NEP255_LOSSWT1._WAND"
wandb.init(project='debugging-nn',name=name)

torch.manual_seed(123) 


# Network

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(self.conv(x))


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.block = [ConvLayer(in_channels, growth_rate, kernel_size=3)]
        for i in range(num_layers - 1):
            self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, kernel_size=3))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class SRDenseNet(nn.Module):
    def __init__(self, num_channels=1, growth_rate=2, num_blocks=7, num_layers=5):
        super(SRDenseNet, self).__init__()

        # low level features
        self.conv = ConvLayer(num_channels, growth_rate * num_layers, 3)

        # high level features
        self.dense_blocks = []
        for i in range(num_blocks):
            self.dense_blocks.append(DenseBlock(growth_rate * num_layers * (i + 1), growth_rate, num_layers))
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(growth_rate * num_layers + growth_rate * num_layers * num_blocks, 256, kernel_size=1),
            nn.LeakyReLU(0.1)
        )

        # reconstruction layer
        self.reconstruction = nn.Conv2d(256, num_channels, kernel_size=3, padding=3 // 2)
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(0.1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.dense_blocks(x1)
        x3 = self.bottleneck(x2)
        x4 = self.tanh(self.reconstruction(x3))
        return {'conv':x1,'dense':x2,'bottleneck':x3,'output':x4}


images = load_image_to_tensor()



#training
net = SRDenseNet()
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
    print('conv shape',result['conv'].shape)
    print('dense shape',result['dense'].shape)
    print('bottleneck',result['bottleneck'].shape)

  conv = convert_image(result['conv'])
  dense = convert_image(result['dense'])
  bottleneck = convert_image(result['bottleneck'])
  # output = convert_image(result['output'])

  conv_image = wandb.Image(conv, caption="conv feature map")      
  wandb.log({"conv_image": conv_image})

  dense_image = wandb.Image(dense, caption="densenet feature map")      
  wandb.log({"dense_image": dense_image})

  bottleneck_image = wandb.Image(bottleneck, caption="bottleneck feature map")      
  wandb.log({"bottleneck_image": bottleneck_image})

  # output = (output-output.min())/(output.max()-output.min())
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