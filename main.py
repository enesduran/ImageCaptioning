import cv2
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import *
from utils import *
from models import Encoder_Decoder_attention


def train(train_loader, model, criterion, model_optimizer, epoch):
    global image

    def threading_func():
        while 1:
            shh = (np.flip(np.swapaxes(image, 0, 2), axis=2) + 0.48) * 0.25
            shh = cv2.resize(shh, (640, 640))
            cv2.imshow("figure", shh)
            cv2.waitKey(50)

    th = threading.Thread(target=threading_func)
    th.start()

    model.train()

    batch_time = avgValsTracker()  # forward prop. + back prop. time
    data_time = avgValsTracker()  # data loading time
    losses = avgValsTracker()  # loss (per word decoded)
    top5accs = avgValsTracker()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imagem = imgs[0].cpu()
        imagem = imagem.detach().numpy()
        scores, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs, caps, caplens)
        if i % 100 * 2 == 0:
            print("Sample best prediction: ")
            print(mNN.words_string[torch.argmax(scores[0], dim=1).cpu()])
            image = imagem


        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        model_optimizer.zero_grad()
        loss.backward()

        if i % 100 * 2 == 0:
            model.eval()
            scores_test = model.test_forward(imgs)
            print("Sample test prediction: ")
            print(mNN.words_string[torch.argmax(scores_test[0], dim=1).cpu()])
            model.train()

        # Clip gradients
        if grad_clip is not None:
            for group in model_optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        # Update weights
        model_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.value:.3f} ({batch_time.average:.3f})\t'
                  'Data Load Time {data_time.value:.3f} ({data_time.average:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.average:.4f})\t'
                  'Top-5 Accuracy {top5.value:.3f} ({top5.average:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(validation_loader, model_network, criterion_func):
    model_network.eval()

    batch_time = avgValsTracker()
    losses = avgValsTracker()
    top5accs = avgValsTracker()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(validation_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forwardla
            scores, caps_sorted, decode_lengths, alphas, sort_ind = model(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove time-steps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion_func(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.value:.3f} ({batch_time.average:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.average:.4f})\t'
                      'Top-5 Accuracy {top5.value:.3f} ({top5.average:.3f})\t'.format(i, len(validation_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                for idxx in range(len(img_caps) - 1, -1, -1):
                    if img_caps[idxx][0] == -1:
                        del img_caps[idxx]
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['x_START_']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print('\n * LOSS: {loss.avg:.3f}, TOP-5 ACCURACY: {top5.avg:.3f}, BLEU-4: {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("skdlşalş")
    filename_train_data = "eee443_project_dataset_train.h5"
    filename_test_data = "eee443_project_dataset_test.h5"

    # init network
    mNN = final_network(filename_train_data, filename_test_data, download_images_train=False,
                        download_images_test=False)

    word_map = {mNN.words_string[i]: i for i in range(1004)}

    # Model parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Training parameters
    epochs = 25  # number of epochs to train for (if early stopping is not triggered)
    batch_size = 32
    workers = 12  # for data-loading; right now, only 1 works with h5py

    learning_rate = 2e-4  # learning rate for decoder
    grad_clip = 5.  # clip gradients at an absolute value of
    alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
    image = np.ones((3, 256, 256))

    model = Encoder_Decoder_attention(use_pretrained=True,
                                      attention_dim=512,
                                      decoder_dim=512,
                                      vocab_size=len(word_map),
                                      embedding_dimension=512,
                                      resnet_type='resnet34')

    model_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    model = model.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    mDataSet_training = CaptionDataset("data_images_training/", mNN.train_cap, mNN.train_imid, 'TRAIN',
                                       transform=transforms.Compose([normalize]))
    train_data_len = int(len(mDataSet_training) * 0.9)
    val_data_len = len(mDataSet_training) - train_data_len
    mDataSet_validation = CaptionDataset("data_images_training/", mNN.train_cap, mNN.train_imid, 'VAL',
                                         transform=transforms.Compose([normalize]))
    mDataSet_validation = torch.utils.data.random_split(mDataSet_validation, [train_data_len, val_data_len])[1]
    train_loader = torch.utils.data.DataLoader(mDataSet_training, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(mDataSet_validation, batch_size=batch_size, shuffle=True,
                                             num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              model_optimizer=model_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(validation_loader=val_loader,
                                model_network=model,
                                criterion_func=criterion)
