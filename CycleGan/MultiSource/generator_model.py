import torch
import torch.nn as nn


class ConvGenBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs) if down else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.block = nn.Sequential(
            ConvGenBlock(channels, channels, kernel_size=3, padding=1),
            ConvGenBlock(channels, channels, use_act=False, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return x + self.block(x)
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        return self.gamma * out + x
    

class Generator(nn.Module):
    def __init__(self, image_channels:int, num_features:int=64, num_residuals:int=9): # num_residuals=9 for 256*256 or larger, num_residuals=6 for 128*128 or smaller
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(image_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvGenBlock(num_features, num_features*2, down=True, kernel_size=3, stride=2, padding=1),
                ConvGenBlock(num_features*2, num_features*4, down=True, kernel_size=3, stride=2, padding=1)
            ]
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for i in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvGenBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvGenBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        self.last = nn.Conv2d(num_features, image_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

class GeneratorWithAttentionBlock(nn.Module):
    def __init__(self, image_channels:int, num_features:int=64, num_residuals:int=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(image_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvGenBlock(num_features, num_features*2, down=True, kernel_size=3, stride=2, padding=1),
                ConvGenBlock(num_features*2, num_features*4, down=True, kernel_size=3, stride=2, padding=1)
            ]
        )
        self.attention = AttentionBlock(num_features*4)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for i in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvGenBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvGenBlock(num_features*2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )
        self.last = nn.Conv2d(num_features, image_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.attention(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return self.tanh(self.last(x))
    

def test():
    image_channels = 3
    image_size = 256
    x = torch.rand((2, image_channels, image_size, image_size))
    generator = Generator(image_channels, 9)
    print(generator(x).shape)

if __name__ == "__main__":
    test()