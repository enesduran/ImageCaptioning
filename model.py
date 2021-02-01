import torch.nn as nn
import torch


class InceptionA(nn.Module):
    def __init__(self, input_size, pool_features):
        super().__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64, eps=0.001),
                                     nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(96, eps=0.001),
                                     nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(96, 96, bias=False, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(96, eps=0.001),
                                     nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=input_size, out_channels=48, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(48, eps=0.001),
                                     nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(64, eps=0.001),
                                     nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(in_channels=input_size, out_channels=pool_features, kernel_size=1,
                                               bias=False),
                                     nn.BatchNorm2d(pool_features, eps=0.001),
                                     nn.ReLU(inplace=True))
        self.conv4_1 = nn.Sequential(nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(64, eps=0.001),
                                     nn.ReLU(inplace=True))

    def forward(self, input_x):
        out_1 = self.conv1_1(input_x)
        out_1 = self.conv1_2(out_1)
        out_1 = self.conv1_3(out_1)

        out_2 = self.conv2_1(input_x)
        out_2 = self.conv2_2(out_2)

        out_3 = self.conv3_1(input_x)

        out_4 = self.conv4_1(input_x)

        outputs = [out_1, out_2, out_3, out_4]
        concatenated_output = torch.cat(outputs, 1)

        return concatenated_output


class InceptionB(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels = input_size, out_channels=448, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(448, eps=0.001),
                                     nn.ReLU(inplace=False))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=448, out_channels=384, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(384, eps= 0.001),
                                     nn.ReLU(inplace=False))
        self.conv1_3_1 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1,3), padding=(0,1), bias=False),
                                       nn.BatchNorm2d(384, eps=0.001),
                                       nn.ReLU(inplace=False))
        self.conv1_3_2 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,1), padding=(1,0), bias=False),
                                       nn.BatchNorm2d(384, eps=0.001),
                                       nn.ReLU(inplace=False))

        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channels=input_size, out_channels=384, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(384, eps=0.001),
                                     nn.ReLU(inplace=True))
        self.conv2_2_1 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(1,3), padding=(0,1), bias=False),
                                       nn.BatchNorm2d(384, eps=0.001),
                                       nn.ReLU(inplace=True))
        self.conv2_2_2 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,1), padding=(1,0), bias=False),
                                       nn.BatchNorm2d(384, eps = 0.001),
                                      nn.ReLU(inplace=False))

        self.conv3_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1 ),
                                     nn.Conv2d(in_channels=input_size, out_channels=192, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(192, eps=0.001),
                                     nn.ReLU(inplace=True))

        self.conv4_1 = nn.Sequential(nn.Conv2d(in_channels=input_size, out_channels=320, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(320, eps = 0.001),
                                     nn.ReLU(inplace=False))

    def forward(self, input_x):
        out_1 = self.conv1_1(input_x)
        out_1 = self.conv1_2(out_1)
        out_1_list = [self.conv1_3_1(out_1), self.conv1_3_2(out_1)]
        out_1 = torch.cat(out_1_list, 1)

        out_2 = self.conv2_1(input_x)
        out_2_list = [self.conv2_2_1(out_2), self.conv2_2_2(out_2)]
        out_2 = torch.cat(out_2_list, 1)

        out_3 = self.conv3_1(input_x)

        out_4 = self.conv4_1(input_x)

        out_list = [out_1, out_2, out_3, out_4]
        out = torch.cat(out_list, 1)
        return out


class Model(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden, lstm_output, word_length, lr):
        super().__init__()
        self.word_length = word_length
        # CNN part
        self.cnn_layer_1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, bias=False),
                                         nn.BatchNorm2d(32, eps=0.001),
                                         nn.ReLU(inplace=True))
        self.cnn_layer_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, bias=False),
                                         nn.BatchNorm2d(32, eps=0.001),
                                         nn.ReLU(inplace=True))
        self.cnn_layer_3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(64, eps=0.001),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2))
        self.cnn_layer_4 = nn.Sequential(nn.Conv2d(64, 80, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(80, eps=0.001),
                                         nn.ReLU(inplace=True))
        self.cnn_layer_5 = nn.Sequential(nn.Conv2d(80, 192, kernel_size=3, bias=False),
                                         nn.BatchNorm2d(192, eps=0.001),
                                         nn.ReLU(inplace=True),
                                         nn.MaxPool2d(kernel_size=3, stride=2))
        self.inception_1 = InceptionA(input_size=192, pool_features=32)
        self.inception_2 = InceptionA(input_size=256, pool_features=64)
        self.hidden_conv = nn.Sequential(nn.Conv2d(in_channels=288, out_channels=1280, kernel_size=4, stride=3, padding=0, bias=False),
                                         nn.BatchNorm2d(1280, eps=0.001),
                                         nn.ReLU(inplace=False))
        self.inception_3 = InceptionB(input_size=1280)

        # LSTM part
        self.hidden_1 = nn.Linear(8 * 8 * 2048, lstm_input_dim)

        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_hidden, num_layers=1, bias=True,
                            batch_first=True, dropout=0, bidirectional=False)
        self.lstm_output_layer = nn.Linear(lstm_hidden, lstm_output)
        # Adam
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward_train(self, input_x, words, coming_batch_size):
        out = self.cnn_layer_1(input_x)
        out = self.cnn_layer_2(out)
        out = self.cnn_layer_3(out)
        out = self.cnn_layer_4(out)
        out = self.cnn_layer_5(out)
        out = self.inception_1(out)
        out = self.inception_2(out)
        out = self.hidden_conv(out)
        out = self.inception_3(out)

        out = out.view(-1, 8 * 8 * 2048)
        out = self.hidden_1(out)
        out = torch.cat((out.view((coming_batch_size, 1004, 1)), words[:, :, 1:]), dim=2).view((coming_batch_size, 17, 1004))

        out, hidden = self.lstm(out.float())
        out = self.lstm_output_layer(out)
        out = torch.softmax(out, dim=2)
        return out

    def forward_test(self, input_x):
        out = self.cnn_layer_1(input_x)
        out = self.cnn_layer_2(out)
        out = self.cnn_layer_3(out)
        out = self.cnn_layer_4(out)
        out = self.cnn_layer_5(out)
        out = self.inception_1(out)
        out = self.inception_2(out)
        out = self.hidden_conv(out)
        out = self.inception_3(out)

        out = out.view(-1, 8 * 8 * 2048)
        out = self.hidden_1(out)

        out = out.unsqueeze(0)

        output_probabilities = []
        hidden = torch.zeros_like(out)

        for i in range(16):
            if i == 0:
                out_lstm, hidden = self.lstm(out, hidden)
                out_lstm = self.lstm_output_layer(out_lstm)
            else:
                out, hidden = self.lstm(out, hidden)
                out = self.lstm_output_layer(out)
                out = torch.softmax(out, dim=1)
                output_probabilities.append(out)
        # we may add X_end if the word count is less than 17
        return output_probabilities
