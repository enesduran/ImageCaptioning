from abc import ABC

import torch
from torch import nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder_Decoder_attention(nn.Module, ABC):
    def __init__(self, use_pretrained, attention_dim, decoder_dim, vocab_size, embedding_dimension=200,
                 resnet_type='resnet34'):
        super(Encoder_Decoder_attention, self).__init__()
        if resnet_type == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=use_pretrained)
            self.encoder_dim = 512
        elif resnet_type == 'resnet34':
            resnet = torchvision.models.resnet34(pretrained=use_pretrained)
            self.encoder_dim = 512
        elif resnet_type == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=use_pretrained)
            self.encoder_dim = 2048
        elif resnet_type == 'resnet101':
            resnet = torchvision.models.resnet101(pretrained=use_pretrained)
            self.encoder_dim = 2048

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.pooling2d = nn.AdaptiveAvgPool2d((14, 14))

        for layer in self.resnet.parameters():
            layer.requires_grad = not use_pretrained
        for layers in list(self.resnet.children())[5:]:
            for layer in layers.parameters():
                layer.requires_grad = True
        ########## --------------------------------------------------------------------- ##########

        self.attention_layer = Attention(self.encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding_layers = nn.Embedding(vocab_size, embedding_dimension)  # embedding layer
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTMCell(embedding_dimension + self.encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(self.encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.embedding_layers.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, image_batch, encoded_captions, caption_lengths):
        out = self.resnet(image_batch)  # (32, 2048, 8, 8)
        out = self.pooling2d(out)  # (32, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (32, encoded_image_size, encoded_image_size, 2048)

        batch_size = out.size(0)
        encoder_dim = out.size(-1)
        out = out.view(batch_size, -1, encoder_dim)
        num_pixels = out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        wvs = self.embedding_layers(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h = self.init_h(out.mean(dim=1))  # (batch_size, decoder_dim)
        c = self.init_c(out.mean(dim=1))

        # remove end
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), 1004).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention_layer(out[:batch_size_t],
                                                                      h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.lstm(
                torch.cat([wvs[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def test_forward(self, image_batch):
        out = self.resnet(image_batch)  # (32, 2048, 8, 8)
        out = self.pooling2d(out)  # (32, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (32, encoded_image_size, encoded_image_size, 2048)

        batch_size = out.size(0)
        encoder_dim = out.size(-1)
        out = out.view(batch_size, -1, encoder_dim)
        num_pixels = out.size(1)

        # Initialize LSTM state
        h = self.init_h(out.mean(dim=1))  # (batch_size, decoder_dim)
        c = self.init_c(out.mean(dim=1))

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, 17, 1004).to(device)
        alphas = torch.zeros(batch_size, 17, num_pixels).to(device)

        start_vector = torch.zeros((batch_size, 17), device="cuda").long()
        start_vector[:,0] = 1
        current_word = self.embedding_layers(start_vector)[:,0,:]

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(17):
            a, alpha = self.attention_layer(out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            a = gate * a
            h, c = self.lstm(torch.cat([current_word, a], dim=1), (h, c))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            current_word = self.embedding_layers(torch.argmax(preds,1))
            alphas[:, t, :] = alpha

        return predictions


class Encoder(nn.Module, ABC):
    def __init__(self, use_pretrained, resnet_type='resnet34'):
        super(Encoder, self).__init__()
        if resnet_type == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=use_pretrained)
        elif resnet_type == 'resnet34':
            resnet = torchvision.models.resnet34(pretrained=use_pretrained)
        elif resnet_type == 'resnet50':
            resnet = torchvision.models.resnet50(pretrained=use_pretrained)
        elif resnet_type == 'resnet101':
            resnet = torchvision.models.resnet101(pretrained=use_pretrained)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.pooling2d = nn.AdaptiveAvgPool2d((14, 14))

        for layer in self.resnet.parameters():
            layer.requires_grad = not use_pretrained
        for layers in list(self.resnet.children())[5:]:
            for layer in layers.parameters():
                layer.requires_grad = True

    def forward(self, image_batch):
        out = self.resnet(image_batch)  # (32, 2048, 8, 8)
        out = self.pooling2d(out)  # (32, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (32, encoded_image_size, encoded_image_size, 2048)
        return out


class Attention(nn.Module, ABC):
    def __init__(self, encoder_dimension, decoder_dimension, attention_dimension):
        super(Attention, self).__init__()
        self.enc_2_attention = nn.Linear(encoder_dimension, attention_dimension)
        self.dec_2_attention = nn.Linear(decoder_dimension, attention_dimension)
        self.alphas = nn.Linear(attention_dimension, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.enc_2_attention(encoder_out)
        att2 = self.dec_2_attention(decoder_hidden)
        att = self.alphas(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha_scores = self.softmax(att)
        a = (encoder_out * alpha_scores.unsqueeze(2)).sum(dim=1)
        return a, alpha_scores


class DecoderWithAttention(nn.Module, ABC):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, out, encoded_captions, caption_lengths):
        batch_size = out.size(0)
        encoder_dim = out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        out = out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # encoder_out = encoder_out[sort_ind]
        # encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        mean_encoder_out = out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
