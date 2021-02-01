import numpy as np
import h5py
import time
import requests
import threading
from os import path
from torch.utils import data



class mDataSet(data.Dataset):
    def __init__(self, data_file_path, captions, imid, transform_func=None):
        self.root_dir = data_file_path
        self.transform = transform_func
        self.image_ids = imid
        self.image_captions = captions

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        idx = self.image_ids[index]
        m_path = self.root_dir + str(idx) + ".jpg"
        img = Image.open(m_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        caption = self.image_captions[index]
        return img, caption



class final_network:
    def __init__(self, train_data_file_path, test_data_file_path, download_images_train=True,
                 download_images_test=True):

        self.filename_train_data = train_data_file_path
        self.filename_test_data = test_data_file_path

        self.train_cap = None
        self.train_imid = None
        self.train_ims = None
        self.train_url = None
        self.train_images = None
        self.test_images = None
        self.test_caps = None
        self.test_imid = None
        self.test_ims = None
        self.test_url = None
        self.existing_indexes_train_data = None  # some images' urls are broken therefore we need the working ones...
        self.existing_indexes_test_data = None  # some images' urls are broken therefore we need the working ones...
        self.words_indexes = None
        self.words_string = None
        self.embedding_matrix = None
        self.read_data(download_images_train, download_images_test)
        # self.process_embedding_matrix()

    def read_data(self, download_flag_train_data, download_flag_test_data):
        # open the data file, get the data
        with h5py.File(self.filename_train_data, "r+") as f:
            dummy = list(f.keys())
            # Get the data
            self.train_url = np.array(f[dummy[3]])
            dummy_tuple = tuple(f[dummy[4]])
            self.words_indexes = np.zeros((len(dummy_tuple[0])), dtype=np.int)
            for idx in range(len(dummy_tuple[0])):
                self.words_indexes[idx] = int(dummy_tuple[0][idx])

            self.words_string = np.array(dummy_tuple[0].dtype.names)
            indexes = np.argsort(self.words_indexes)
            self.words_string = self.words_string[indexes]
            self.words_indexes = self.words_indexes[indexes]

            self.train_ims = np.array(f[dummy[2]])
            self.train_cap = np.array(f[dummy[0]])
            self.train_imid = np.array(f[dummy[1]]).reshape(len(np.array(f[dummy[1]])))

            indexes = np.argsort(self.train_imid)
            self.train_cap = self.train_cap[indexes]
            self.train_imid = self.train_imid[indexes]
            self.train_imid -= 1

        with h5py.File(self.filename_test_data, "r+") as f:
            dummy = list(f.keys())
            # Get the data
            self.test_caps = np.array(f[dummy[0]])
            self.test_imid = np.array(f[dummy[1]]).reshape(len(np.array(f[dummy[1]])))
            self.test_ims = np.array(f[dummy[2]])
            self.test_url = np.array(f[dummy[3]])
            indexes = np.argsort(self.test_imid)
            self.test_caps = self.test_caps[indexes]
            self.test_imid = self.test_imid[indexes]

        if download_flag_train_data:
            print(f'MESSAGE: Download started for training set... Running in {6} threads')
            t1 = time.time()

            def dummy_downloader_training_data(arr, num):
                for index in range(int(len(arr) / 6)):
                    image_name = index * 6 + num
                    curr_url = arr[image_name]
                    response = requests.get(curr_url)
                    if image_name % 500 == 0 and image_name != 0:
                        t3 = time.time()
                        gh = t3 - t1
                        remaining_time = (len(arr) * gh / image_name) / 60
                        print(
                            f"Image {image_name} is downloaded, Time passed is {(gh / 60) :.2f} mins and remaining "
                            f"time is {(remaining_time - gh / 60):.2f} mins")
                    if response.ok:
                        file = open("data_images_training/" + str(image_name) + ".jpg", "wb")
                        # file = open(self.filename_train_data + str(image_name) + ".jpg", "wb")
                        file.write(response.content)
                        file.close()

            thread_pool = []
            for idx in range(6):
                print(f"Thread {idx} started")
                t = threading.Thread(target=dummy_downloader_training_data, args=(self.train_url, idx))
                thread_pool.append(t)

            for t in thread_pool:
                t.start()

            thread_pool[5].join()

            t2 = time.time()
            print(f"Download finished in {(t2 - t1) / 60}  mins")

        if download_flag_test_data:
            print(f'MESSAGE: Download started for test set... Running in {6} threads')
            t1 = time.time()

            def dummy_downloader_training_data(arr, num):
                for index in range(int(len(arr) / 6)):
                    image_name = index * 6 + num
                    curr_url = arr[image_name]
                    response = requests.get(curr_url)
                    if image_name % 500 == 0 and image_name != 0:
                        t3 = time.time()
                        gh = t3 - t1
                        remaining_time = (len(arr) * gh / image_name) / 60
                        print(
                            f"{image_name} image is downloaded from test set, time passed is {(gh / 60) :.2f} mins "
                            f"and remaining time is {(remaining_time - gh / 60):.2f} mins")
                    if response.ok:
                        file = open("data_images_test/" + str(image_name) + ".jpg", "wb")
                        # file = open(self.filename_test_data + str(image_name) + ".jpg", "wb")
                        file.write(response.content)
                        file.close()

            thread_pool = []
            for idx in range(6):
                print(f"Thread {idx} started")
                t = threading.Thread(target=dummy_downloader_training_data, args=(self.test_url, idx))
                thread_pool.append(t)

            for t in thread_pool:
                t.start()

            thread_pool[4].join()

            t2 = time.time()
            print(f"Download finished in {(t2 - t1) / 60}  mins")

        """
            burdan sonrası broken urllerdeki resimlerin captionlarını datasetten silmek için... train_imid den 
            idlerini ve train_cap den captionları sırayla siliyo
        """
        if path.isfile("trainimid.npy") and path.isfile("train_cap.npy") and path.isfile("existing_indexes_train.npy"):
            self.train_imid = np.load("trainimid.npy")
            self.train_cap = np.load("train_cap.npy")
            self.existing_indexes_train_data = np.load("existing_indexes_train.npy")
        else:
            print(f"Sorry, you will wait!!!")
            self.existing_indexes_train_data = np.arange(len(self.train_url))
            print(f"Scanning training data for missing images...")
            dummy = 0
            for idx in range(len(self.train_url)):
                if idx % 10000 == 0:
                    print(f"---Remaining images: {len(self.train_url) - idx}  Missing images: {dummy}")
                filepath = "data_images_training/" + str(idx) + ".jpg"
                if not path.isfile(filepath):
                    index_to_delete = np.where(self.train_imid == idx)
                    self.train_imid = np.delete(self.train_imid, index_to_delete, 0)
                    self.train_cap = np.delete(self.train_cap, index_to_delete, 0)
                    self.existing_indexes_train_data = np.delete(self.existing_indexes_train_data,
                                                                 np.where(self.existing_indexes_train_data == idx), 0)
                    dummy += 1
            np.save('trainimid', self.train_imid)
            np.save('train_cap', self.train_cap)
            np.save("existing_indexes_train", self.existing_indexes_train_data)

        if path.isfile("testimid.npy") and path.isfile("test_cap.npy") and path.isfile("existing_indexes_test.npy"):
            self.test_imid = np.load("testimid.npy")
            self.test_caps = np.load("test_cap.npy")
            self.existing_indexes_test_data = np.load("existing_indexes_test.npy")
        else:
            self.existing_indexes_test_data = np.arange(len(self.test_url))
            print(f"Scanning test data for missing images...")
            dummy = 0
            for idx in range(len(self.test_url)):
                if idx % 5000 == 0:
                    print(f"---Remaining images: {len(self.test_url) - idx}  Missing images: {dummy}")
                filepath = "data_images_test/" + str(idx) + ".jpg"
                if not path.isfile(filepath):
                    index_to_delete = np.where(self.test_imid == idx)
                    self.test_imid = np.delete(self.test_imid, index_to_delete, 0)
                    self.test_caps = np.delete(self.test_caps, index_to_delete, 0)
                    self.existing_indexes_test_data = np.delete(self.existing_indexes_test_data,
                                                                np.where(self.existing_indexes_test_data == idx), 0)
                    dummy += 1
            np.save('testimid', self.test_imid)
            np.save('test_cap', self.test_caps)
            np.save("existing_indexes_test", self.existing_indexes_test_data)

        # self.train_images = np.zeros((len(self.train_url), 240, 320, 3), dtype=np.int8)  # burası 17 kusur gb ram
        # self.test_images = np.zeros((len(self.test_url) , 240, 320, 3 ) , dtype=np.int8) burayı da commentten
        # çıkarırsanız toplam 26 gb ram gerekiyor!!! pytorch dataloader kullanıcaz mecburen


class avgValsTracker(object):
    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


def accuracy(scores, targets, k):
    words, indexes = scores.topk(k, 1, True, True)
    acc = indexes.eq(targets.view(-1, 1).expand_as(indexes))
    total_acc = acc.view(-1).float().sum()  # 0D tensor
    return total_acc.item() * (100.0 / targets.size(0))
