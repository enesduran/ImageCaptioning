import torchvision.models as models
import torch.nn as nn

import torch


class pretrained_Resnet_LSTM_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, decoder_last2,decoder_last1, embedding_matrix):
        super(pretrained_Resnet_LSTM_Decoder, self).__init__()

        resnet = models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False

        num_ftrs = resnet.fc.in_features

        #layers = list(resnet.children())[:-1]
        #self.resnet = nn.Sequential(*layers)
        self.resnet = resnet
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.batch_normalizer1 = nn.BatchNorm1d(num_ftrs)
        self.batch_normalizer2 = nn.BatchNorm1d(512)
        self.encoder_last = nn.Linear(num_ftrs, embed_size)

        self.embedding_layer = nn.Embedding(vocab_size, embed_size) # _weight=torch.from_numpy(embedding_matrix))
        # self.embedding_layer.weight.requires_grad = False
        # self.embedding_layer.requires_grad_(False)

        #self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embedding_matrix))

        #self.embedding_layer = nn.Embedding(1004,300)

        # self.lstm_decoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.lstmcell = nn.LSTMCell(input_size=embed_size , hidden_size= hidden_size)

        self.decoder_last2 = nn.Linear(hidden_size, decoder_last2)
        # self.decoder_last1 = nn.Linear(decoder_last2 , decoder_last1)
        self.last = nn.Linear(decoder_last2, vocab_size)

        self.out_activation = nn.Softmax(dim=2)

    def forward(self, images, captions):
        # hidden = torch.zeros((1, images.size(0), 512)).cuda()
        # memory =  torch.zeros((1, images.size(0), 512)).cuda()
        # lstm_hidden = (hidden, memory)

        features = self.resnet(images)
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        #features = self.batch_normalizer1(features)

        hx = torch.zeros((batch_size, 512), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        cx = torch.zeros((batch_size, 512), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # features = self.batch_normalizer(features)
        # features = features / features.max()

        features = self.encoder_last(features)
        print(f"Max1: {torch.max(features)}")
        # features = torch.tanh(features)


        captions = captions[:, :-1]
        embed = self.embedding_layer(captions.long())
        embed = torch.cat((features.unsqueeze(1), embed.float()), dim=1)

        lstm_output = torch.zeros((batch_size, 17, 512) , device=torch.device('cuda'))
        for i in range(17):
            hx, cx = self.lstmcell(embed[:, i,:], (hx, cx))
            lstm_output[:, i, :] = hx

        # lstm_decoder_outputs, self.hidden = self.lstm_decoder(embed , self.hidden)
        #lstm_decoder_outputs = torch.tanh(lstm_decoder_outputs)

        out = self.decoder_last2(lstm_output)
        print(f"Max2: {torch.max(out)}")
        out = torch.nn.functional.relu(out)

        out = self.last(out)
        print(f"Max3: {torch.max(out)}")
        out = self.out_activation(out)

        return out
