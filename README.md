# Convolution 3D cuDNN API 实现样例

### 对照的Pytorch源码
```python
class Conv3dNet(nn.Module):
    def __init__(self):
        super(Conv3dNet, self).__init__()

        self.conv = nn.Conv3d(
                in_channels=3,
                out_channels=64,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                dilation=(1, 1, 1),
                groups=1,
                bias=False)
        self.conv.weight.data = torch.ones(64, 3, 1, 7, 7)

    def forward(self, input):
        return self.conv(input)


def load_state_dict(pth=None):
    conv_net = Conv3dNet()
    conv_net.eval()
    sample_input = torch.ones(1, 3, 8, 350, 640) * 0.5
    return conv_net.cuda(), sample_input.cuda()


if __name__ == '__main__':
    net, sample_input = load_state_dict()
    w = net(sample_input)
    print(w)
    print(w.shape)
```


