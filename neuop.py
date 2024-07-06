from packages import *

class NeuralOperatorBlock(nn.Module):
    def __init__(self, in_chs, out_chs, modes1, modes2, modes3):
        super(NeuralOperatorBlock, self).__init__()
        """
        This class perform transforming the input into Fourier space using FFT,
        then multiplied by a matrix that contains learnable parameters,
        and finally projecting the results to the real space using inverse FFT.
        
        Fourier modes must be less or equal than: N//2 + 1, where N = [x.size, y.size, t.size]
        """
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1, where N is the minimum of [x.size, y.size, t.size]
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (self.out_channels * self.out_channels))
        self.weights0 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def complex_mul(self, input, weights):
        # (batch,in_channel,x,y,t ), (in_channel,out_channel,x,y,t) --> (batch,out_channel,x,y,t)
        # Summation is performed on the in_channel axis
        return torch.einsum("bixyt,ioxyt->boxyt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Transform to Fourier space
        x_fourier = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply Fourier modes
        out_fourier = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_fourier[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.complex_mul(x_fourier[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights0)
        
        out_fourier[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.complex_mul(x_fourier[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights1)
        
        out_fourier[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.complex_mul(x_fourier[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights2)
        
        out_fourier[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.complex_mul(x_fourier[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights3)

        # inverse FFT to real space
        x = torch.fft.irfftn(out_fourier, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x
    

class U_Net(nn.Module):
    def __init__(self, input_channels, output_channels, conv_kernel_size, dropout_rate):
    
        """this architecture works only if input_channels=output_channels
        Note: if your grid is not (40,40,40) you need to adjust the kernel_size in the deconv layers"""
        
        super(U_Net, self).__init__()

        assert input_channels == output_channels, "input_channels must be equal to output_channels"

        self.input_channels = input_channels

        self.conv0 = self.conv(input_channels, output_channels, kernel_size=conv_kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=conv_kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv1_0 = self.conv(input_channels, output_channels, kernel_size=conv_kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=conv_kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=conv_kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels, kernel_size=(4,4,4))
        self.deconv1 = self.deconv(input_channels*2, output_channels, kernel_size=(4,4,4))
        self.deconv0 = self.deconv(input_channels*2, output_channels, kernel_size=(4,4,4))
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=conv_kernel_size, stride=1)


    def forward(self, x):

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_0(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        
        out_deconv2 = self.deconv2(out_conv2)
        cat2 = torch.cat((out_conv1, out_deconv2), 1)
        
        out_deconv1 = self.deconv1(cat2)
        cat1 = torch.cat((out_conv0, out_deconv1), 1)
        
        out_deconv0 = self.deconv0(cat1)
        cat0 = torch.cat((x, out_deconv0), 1)
        
        y = self.output_layer(cat0)

        return y

    def conv(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        """for an input of size (N, input_channels, D, H, W)
        the output is of size (N, output_channels, Dout, Hout, Wout), where
        
        Dout = { D + 2*padding[0] - dilation[0]*(kernel_size[0]-1) -1 } // stride[0] - 1
        Hout = { H + 2*padding[1] - dilation[1]*(kernel_size[1]-1) -1 } // stride[1] - 1
        Wout = { W + 2*padding[2] - dilation[2]*(kernel_size[2]-1) -1 } // stride[2] - 1
        """
        return nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels, kernel_size):
        """for an input of size (N, input_channels, D, H, W)
        the output is of size (N, output_channels, Dout, Hout, Wout), where
        
        Dout = (D-1)*stride[0] - 2*padding[0] + dilation[0]*(kernel_size[0]-1) + output_padding[0] + 1
        Hout = (H-1)*stride[1] - 2*padding[1] + dilation[1]*(kernel_size[1]-1) + output_padding[1] + 1
        Wout = (W-1)*stride[2] - 2*padding[2] + dilation[2]*(kernel_size[2]-1) + output_padding[2] + 1

        output_padding default: 0
        dilation default: 1
        padding default: 0
        bias default: True
        stride default: 1
        """
        return nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels, kernel_size=kernel_size,
                               stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride):
        return nn.Conv3d(input_channels, output_channels, kernel_size=kernel_size,
                         stride=stride, padding=(kernel_size - 1) // 2)
    

class NeuralOperatorAll(nn.Module):
    def __init__(self, input_channels, output_channels, modes1, modes2, modes3, width):
        super(NeuralOperatorAll, self).__init__()
        """       
        input shape: (batchsize, x=40, y=40, t=40, c=6)
        output shape: (batchsize, x=40, y=40, t=40, c=2)
        
        Input channels are: (permeability, producer_controls, injectors_rates, well_locations, pressure_initial_condition, saturation_initial_condition)
        Ouptput channels are: (pressures, saturations)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        
        self.input_channels = input_channels

        
        self.fc0 = nn.Linear(self.input_channels, self.width)
        
        self.neu_op0 = NeuralOperatorBlock(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.neu_op1 = NeuralOperatorBlock(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.neu_op2 = NeuralOperatorBlock(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.neu_op3 = NeuralOperatorBlock(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.neu_op4 = NeuralOperatorBlock(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.neu_op5 = NeuralOperatorBlock(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.unet3 = U_Net(self.width, self.width, conv_kernel_size=3, dropout_rate=0)
        self.unet4 = U_Net(self.width, self.width, conv_kernel_size=3, dropout_rate=0)
        self.unet5 = U_Net(self.width, self.width, conv_kernel_size=3, dropout_rate=0)
        
        self.w0 = nn.Conv1d(self.width, self.width, kernel_size=1)
        self.w1 = nn.Conv1d(self.width, self.width, kernel_size=1)
        self.w2 = nn.Conv1d(self.width, self.width, kernel_size=1)
        self.w3 = nn.Conv1d(self.width, self.width, kernel_size=1)
        self.w4 = nn.Conv1d(self.width, self.width, kernel_size=1)
        self.w5 = nn.Conv1d(self.width, self.width, kernel_size=1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, output_channels)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_t = x.shape[1], x.shape[2], x.shape[3]
        
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.neu_op0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_t)
        x = x1 + x2
        x = F.relu(x)
        
        x1 = self.neu_op1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_t)
        x = x1 + x2
        x = F.relu(x)
        
        x1 = self.neu_op2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_t)
        x = x1 + x2 
        x = F.relu(x)
        
        x1 = self.neu_op3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_t)
        x3 = self.unet3(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.neu_op4(x)
        x2 = self.w4(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_t)
        x3 = self.unet4(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x1 = self.neu_op5(x)
        x2 = self.w5(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_t)
        x3 = self.unet5(x)
        x = x1 + x2 + x3
        x = F.relu(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    

class NeuralOperator(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, width):
        super(NeuralOperator, self).__init__()

        """
        A wrapper class
        """

        self.model = NeuralOperatorAll(in_channels, out_channels, modes1, modes2, modes3, width)
        self.out_channels = out_channels


    def forward(self, x):

        batchsize = x.shape[0]
        size_x, size_y, size_t = x.shape[1], x.shape[2], x.shape[3]
        
        x = self.model(x)

        x = x.view(batchsize, size_x, size_y, size_t, self.out_channels)

        return x.squeeze()