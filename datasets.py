import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CaptionDataset(Dataset):
    def __init__(self, data_folder, captions, imid, split, transform=None):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        self.root_dir = data_folder
        self.transform = transform
        self.image_ids = imid
        self.image_captions = captions
        self.transform = transform

    def __getitem__(self, i):
        idx = self.image_ids[i]
        m_path = self.root_dir + str(idx) + ".jpg"
        img = Image.open(m_path).convert('RGB')
        img = img.resize((256, 256))
        img = torch.FloatTensor(np.array(img) / 255.0)
        img = img.transpose(0, 2)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.image_captions[i])
        caplen = torch.LongTensor([17])

        if self.split == 'TRAIN':
            return img, caption, caplen
        elif self.split == 'VAL':
            all_caps = self.image_captions[np.where(self.image_ids == self.image_ids[i])]
            if len(all_caps) < 6:
                missing = 6 - len(all_caps)
                all_caps = np.append(all_caps, np.ones((missing, 17)) * -1, axis=0)
            return img, caption, caplen, torch.Tensor(all_caps)

    def __len__(self):
        return len(self.image_ids)
