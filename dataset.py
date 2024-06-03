import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
#import random
import torchvision.datasets as datasets
class ImageData(object):

    def __init__(self, dataset, path="/mnt/workspace/colla_group/data/"):
        dataset = os.path.join(path, dataset)
        data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
        labels = data.classes
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        image=self._TRANSFORM(Image.open(image))
        return image, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



class DatasetObject:
    def __init__(self, dataset, n_client, seed, rule, unbalanced_sgm=0, rule_arg='', data_path=''):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        self.rule_arg = rule_arg
        self.seed     = seed
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        # self.name = "{:s}_{:s}_{:s}_{:.0f}%-{:d}".format(dataset, rule, str(rule_arg), args.active_ratio*args.total_client, args.total_client)
        self.name = "%s_%d_%d_%s_%s" %(self.dataset, self.n_client, self.seed, self.rule, rule_arg_str)
        self.name += '_%f' %unbalanced_sgm if unbalanced_sgm!=0 else ''
        self.unbalanced_sgm = unbalanced_sgm
        self.data_path = data_path
        if rule not in ["Pathological","Dirichlet"]:
            self.set_data_dg()
        else:
            self.set_data()
    
    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('%sData/%s' %(self.data_path, self.name)):
            # Get Raw data                
            if self.dataset == 'mnist':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                trainset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=True , download=True, transform=transform)
                testset = torchvision.datasets.MNIST(root='%sData/Raw' %self.data_path, 
                                                    train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=1)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            if self.dataset == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

                trainset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR10(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
                
            if self.dataset == 'CIFAR100':
                print(self.dataset)
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                                                                     std=[0.2675, 0.2565, 0.2761])])
                trainset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=True , download=True, transform=transform)
                testset = torchvision.datasets.CIFAR100(root='%sData/Raw' %self.data_path,
                                                      train=False, download=True, transform=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            
            if self.dataset == 'tinyimagenet':
                print(self.dataset)
                transform = transforms.Compose([# transforms.Resize(224),
                                                transforms.ToTensor(),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406], #pre-train
                                                #                      std=[0.229, 0.224, 0.225])])
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                                     std=[0.5, 0.5, 0.5])])

                root_dir = "./Data/Raw/tiny-imagenet-200/"
                trn_img_list, trn_lbl_list, tst_img_list, tst_lbl_list = [], [], [], []
                trn_file = os.path.join(root_dir, 'train_list.txt')
                tst_file = os.path.join(root_dir, 'val_list.txt')
                with open(trn_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        trn_img_list.append(img)
                        trn_lbl_list.append(int(lbl))
                with open(tst_file) as f:
                    line_list = f.readlines()
                    for line in line_list:
                        img, lbl = line.strip().split()
                        tst_img_list.append(img)
                        tst_lbl_list.append(int(lbl))
                trainset = DatasetFromDir(img_root=root_dir, img_list=trn_img_list, label_list=trn_lbl_list, transformer=transform)
                testset = DatasetFromDir(img_root=root_dir, img_list=tst_img_list, label_list=tst_lbl_list, transformer=transform)
                train_load = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
                test_load = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=0)
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            
            if self.dataset != 'emnist':
                train_itr = train_load.__iter__(); test_itr = test_load.__iter__() 
                # labels are of shape (n_data,)
                train_x, train_y = train_itr.__next__()
                test_x, test_y = test_itr.__next__()

                train_x = train_x.numpy(); train_y = train_y.numpy().reshape(-1,1)
                test_x = test_x.numpy(); test_y = test_y.numpy().reshape(-1,1)
            
            
            if self.dataset == 'emnist':
                emnist = io.loadmat(self.data_path + "Data/Raw/matlab/emnist-letters.mat")
                # load training dataset
                x_train = emnist["dataset"][0][0][0][0][0][0]
                x_train = x_train.astype(np.float32)

                # load training labels
                y_train = emnist["dataset"][0][0][0][0][0][1] - 1 # make first class 0

                # take first 10 classes of letters
                train_idx = np.where(y_train < 10)[0]

                y_train = y_train[train_idx]
                x_train = x_train[train_idx]

                mean_x = np.mean(x_train)
                std_x = np.std(x_train)

                # load test dataset
                x_test = emnist["dataset"][0][0][1][0][0][0]
                x_test = x_test.astype(np.float32)

                # load test labels
                y_test = emnist["dataset"][0][0][1][0][0][1] - 1 # make first class 0

                test_idx = np.where(y_test < 10)[0]

                y_test = y_test[test_idx]
                x_test = x_test[test_idx]
                
                x_train = x_train.reshape((-1, 1, 28, 28))
                x_test  = x_test.reshape((-1, 1, 28, 28))
                
                # normalise train and test features

                train_x = (x_train - mean_x) / std_x
                train_y = y_train
                
                test_x = (x_test  - mean_x) / std_x
                test_y = y_test
                
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            
            # Shuffle Data
            np.random.seed(self.seed)
            rand_perm = np.random.permutation(len(train_y))
            train_x = train_x[rand_perm]
            train_y = train_y[rand_perm]
            
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
            
            
            ###
            n_data_per_client = int((len(train_y)) / self.n_client)
            # Draw from lognormal distribution
            # client_data_list = (np.random.lognormal(mean=np.log(n_data_per_client), sigma=self.unbalanced_sgm, size=self.n_client))
            # client_data_list = (client_data_list/(np.sum(client_data_list)/len(train_y)))
            client_data_list = np.ones(self.n_client, dtype=int)*n_data_per_client
            diff = np.sum(client_data_list) - len(train_y)
            
            # Add/Subtract the excess number starting from first client
            if diff!= 0:
                for client_i in range(self.n_client):
                    if client_data_list[client_i] > diff:
                        client_data_list[client_i] -= diff
                        break
            ###     
            
            if self.rule == 'Dirichlet' or self.rule == 'Pathological':
                if self.rule == 'Dirichlet':
                    cls_priors = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=self.n_client)
                    # np.save("results/heterogeneity_distribution_{:s}.npy".format(self.dataset), cls_priors)
                    prior_cumsum = np.cumsum(cls_priors, axis=1)
                elif self.rule == 'Pathological':
                    c = int(self.rule_arg)
                    a = np.ones([self.n_client,self.n_cls])
                    a[:,c::] = 0
                    [np.random.shuffle(i) for i in a]
                    # np.save("results/heterogeneity_distribution_{:s}_{:s}.npy".format(self.dataset, self.rule), a/c)
                    prior_cumsum = a.copy()
                    for i in range(prior_cumsum.shape[0]):
                        for j in range(prior_cumsum.shape[1]):
                            if prior_cumsum[i,j] != 0:
                                prior_cumsum[i,j] = a[i,0:j+1].sum()/c*1.0

                idx_list = [np.where(train_y==i)[0] for i in range(self.n_cls)]
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
                true_sample = [0 for i in range(self.n_cls)]
                # print(cls_amount)
                client_x = [ np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32) for client__ in range(self.n_client) ]
                client_y = [ np.zeros((client_data_list[client__], 1)).astype(np.int64) for client__ in range(self.n_client) ]
    
                while(np.sum(client_data_list)!=0):
                    curr_client = np.random.randint(self.n_client)
                    # If current node is full resample a client
                    # print('Remaining Data: %d' %np.sum(client_data_list))
                    if client_data_list[curr_client] <= 0:
                        continue
                    client_data_list[curr_client] -= 1
                    curr_prior = prior_cumsum[curr_client]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if train_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            cls_amount [cls_label] = len(idx_list[cls_label]) 
                            continue
                        cls_amount[cls_label] -= 1
                        true_sample[cls_label] += 1
                        
                        client_x[curr_client][client_data_list[curr_client]] = train_x[idx_list[cls_label][cls_amount[cls_label]]]
                        client_y[curr_client][client_data_list[curr_client]] = train_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break
                print(true_sample)
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)
                

            
            elif self.rule == 'iid' and self.dataset == 'CIFAR100' and self.unbalanced_sgm==0:
                assert len(train_y)//100 % self.n_client == 0 
                
                # create perfect IID partitions for cifar100 instead of shuffling
                idx = np.argsort(train_y[:, 0])
                n_data_per_client = len(train_y) // self.n_client
                # client_x dtype needs to be float32, the same as weights
                client_x = np.zeros((self.n_client, n_data_per_client, 3, 32, 32), dtype=np.float32)
                client_y = np.zeros((self.n_client, n_data_per_client, 1), dtype=np.float32)
                train_x = train_x[idx] # 50000*3*32*32
                train_y = train_y[idx]
                n_cls_sample_per_device = n_data_per_client // 100
                for i in range(self.n_client): # devices
                    for j in range(100): # class
                        client_x[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :, :, :] = train_x[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :, :, :]
                        client_y[i, n_cls_sample_per_device*j:n_cls_sample_per_device*(j+1), :] = train_y[500*j+n_cls_sample_per_device*i:500*j+n_cls_sample_per_device*(i+1), :] 
            
            
            elif self.rule == 'iid':
                
                client_x = [ np.zeros((client_data_list[client__], self.channels, self.height, self.width)).astype(np.float32) for client__ in range(self.n_client) ]
                client_y = [ np.zeros((client_data_list[client__], 1)).astype(np.int64) for client__ in range(self.n_client) ]
            
                client_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
                for client_idx_ in range(self.n_client):
                    client_x[client_idx_] = train_x[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                    client_y[client_idx_] = train_y[client_data_list_cum_sum[client_idx_]:client_data_list_cum_sum[client_idx_+1]]
                
                client_x = np.asarray(client_x)
                client_y = np.asarray(client_y)

            
            self.client_x = client_x; self.client_y = client_y

            self.test_x  = test_x;  self.test_y  = test_y
            
            # Save data
            print('begin to save data...')

            os.mkdir('%sData/%s' %(self.data_path, self.name))
            
            np.save('%sData/%s/client_x.npy' %(self.data_path, self.name), client_x)
            np.save('%sData/%s/client_y.npy' %(self.data_path, self.name), client_y)

            np.save('%sData/%s/test_x.npy'  %(self.data_path, self.name),  test_x)
            np.save('%sData/%s/test_y.npy'  %(self.data_path, self.name),  test_y)

            print('data loading finished.')

        else:
            print("Data is already downloaded")
            self.client_x = np.load('%sData/%s/client_x.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.client_y = np.load('%sData/%s/client_y.npy' %(self.data_path, self.name), mmap_mode = 'r')
            self.n_client = len(self.client_x)

            self.test_x  = np.load('%sData/%s/test_x.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            self.test_y  = np.load('%sData/%s/test_y.npy'  %(self.data_path, self.name), mmap_mode = 'r')
            
            if self.dataset == 'mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;
            if self.dataset == 'fashion_mnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'emnist':
                self.channels = 1; self.width = 28; self.height = 28; self.n_cls = 10;
            if self.dataset == 'tinyimagenet':
                self.channels = 3; self.width = 64; self.height = 64; self.n_cls = 200;
            
            print('data loading finished.')

        
def generate_syn_logistic(dimension, n_client, n_cls, avg_data=4, alpha=1.0, beta=0.0, theta=0.0, iid_sol=False, iid_dat=False):
    
    # alpha is for minimizer of each client
    # beta  is for distirbution of points
    # theta is for number of data points
    
    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)
    
    samples_per_user = (np.random.lognormal(mean=np.log(avg_data + 1e-3), sigma=theta, size=n_client)).astype(int)
    print('samples per user')
    print(samples_per_user)
    print('sum %d' %np.sum(samples_per_user))
    
    num_samples = np.sum(samples_per_user)

    data_x = list(range(n_client))
    data_y = list(range(n_client))

    mean_W = np.random.normal(0, alpha, n_client)
    B = np.random.normal(0, beta, n_client)

    mean_x = np.zeros((n_client, dimension))

    if not iid_dat: # If IID then make all 0s.
        for i in range(n_client):
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    sol_W = np.random.normal(mean_W[0], 1, (dimension, n_cls))
    sol_B = np.random.normal(mean_W[0], 1, (1, n_cls))
    
    if iid_sol: # Then make vectors come from 0 mean distribution
        sol_W = np.random.normal(0, 1, (dimension, n_cls))
        sol_B = np.random.normal(0, 1, (1, n_cls))
    
    for i in range(n_client):
        if not iid_sol:
            sol_W = np.random.normal(mean_W[i], 1, (dimension, n_cls))
            sol_B = np.random.normal(mean_W[i], 1, (1, n_cls))

        data_x[i] = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        data_y[i] = np.argmax((np.matmul(data_x[i], sol_W) + sol_B), axis=1).reshape(-1,1)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    return data_x, data_y
        
class DatasetSynthetic:
    def __init__(self, alpha, beta, theta, iid_sol, iid_data, n_dim, n_client, n_cls, avg_data, data_path, name_prefix):
        self.dataset = 'synt'
        self.name  = name_prefix + '_'
        self.name += '%d_%d_%d_%d_%f_%f_%f_%s_%s' %(n_dim, n_client, n_cls, avg_data,
                alpha, beta, theta, iid_sol, iid_data)

        if (not os.path.exists('%sData/%s/' %(data_path, self.name))):
            # Generate data
            print('Sythetize')
            data_x, data_y = generate_syn_logistic(dimension=n_dim, n_client=n_client, n_cls=n_cls, avg_data=avg_data, 
                                        alpha=alpha, beta=beta, theta=theta, 
                                        iid_sol=iid_sol, iid_dat=iid_data)
            os.mkdir('%sData/%s/' %(data_path, self.name))
            os.mkdir('%sModel/%s/' %(data_path, self.name))
            np.save('%sData/%s/data_x.npy' %(data_path, self.name), data_x)
            np.save('%sData/%s/data_y.npy' %(data_path, self.name), data_y)
        else:
            # Load data
            print('Load')
            data_x = np.load('%sData/%s/data_x.npy' %(data_path, self.name))
            data_y = np.load('%sData/%s/data_y.npy' %(data_path, self.name))

        for client in range(n_client):
            print(', '.join(['%.4f' %np.mean(data_y[client]==t) for t in range(n_cls)]))

        self.client_x = data_x
        self.client_y = data_y

        self.test_x = np.concatenate(self.client_x, axis=0)
        self.test_y = np.concatenate(self.client_y, axis=0)
        self.n_client = len(data_x)
        print(self.client_x.shape)

# Original prepration is from LEAF paper...
# This loads Shakespeare dataset only.
# data_path/train and data_path/test are assumed to be processed
# To make the dataset smaller,
# We take 2000 datapoints for each client in the train_set

class ShakespeareObjectCrop:
    def __init__(self, data_path, dataset_prefix, crop_amount=2000, test_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')
        
        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        
        # Ignore groups information, combine test cases for different clients into one test data
        # Change structure to DatasetObject structure
        
        self.users = users
        
        self.n_client = len(users)
        self.user_idx = np.asarray(list(range(self.n_client)))
        self.client_x = list(range(self.n_client))
        self.client_y = list(range(self.n_client))

        print(train_data)
        print(test_data)
        
        test_data_count = 0
        
        for client in range(self.n_client):
            np.random.seed(rand_seed + client)
            start = np.random.randint(len(train_data[users[client]]['x'])-crop_amount)
            self.client_x[client] = np.asarray(train_data[users[client]]['x'])[start:start+crop_amount]
            self.client_y[client] = np.asarray(train_data[users[client]]['y'])[start:start+crop_amount]
            
        test_data_count = (crop_amount//test_ratio) * self.n_client
        self.test_x = list(range(test_data_count))
        self.test_y = list(range(test_data_count))
        
        test_data_count = 0
        for client in range(self.n_client):
            curr_amount = (crop_amount//test_ratio)
            np.random.seed(rand_seed + client)
            start = np.random.randint(len(test_data[users[client]]['x'])-curr_amount)
            self.test_x[test_data_count: test_data_count+ curr_amount] = np.asarray(test_data[users[client]]['x'])[start:start+curr_amount]
            self.test_y[test_data_count: test_data_count+ curr_amount] = np.asarray(test_data[users[client]]['y'])[start:start+curr_amount]
            
            test_data_count += curr_amount
        
        self.client_x = np.asarray(self.client_x)
        self.client_y = np.asarray(self.client_y)
        
        self.test_x = np.asarray(self.test_x)
        self.test_y = np.asarray(self.test_y)
        
        # Convert characters to numbers
        
        self.client_x_char = np.copy(self.client_x)
        self.client_y_char = np.copy(self.client_y)
        
        self.test_x_char = np.copy(self.test_x)
        self.test_y_char = np.copy(self.test_y)
        
        self.client_x = list(range(len(self.client_x_char)))
        self.client_y = list(range(len(self.client_x_char)))
        

        for client in range(len(self.client_x_char)):
            client_list_x = list(range(len(self.client_x_char[client])))
            client_list_y = list(range(len(self.client_x_char[client])))
            
            for idx in range(len(self.client_x_char[client])):
                client_list_x[idx] = np.asarray(word_to_indices(self.client_x_char[client][idx]))
                client_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.client_y_char[client][idx]))).reshape(-1)

            self.client_x[client] = np.asarray(client_list_x)
            self.client_y[client] = np.asarray(client_list_y)
                
        self.client_x = np.asarray(self.client_x)
        self.client_y = np.asarray(self.client_y)
        
        
        self.test_x = list(range(len(self.test_x_char)))
        self.test_y = list(range(len(self.test_x_char)))
                
        for idx in range(len(self.test_x_char)):
            self.test_x[idx] = np.asarray(word_to_indices(self.test_x_char[idx]))
            self.test_y[idx] = np.argmax(np.asarray(letter_to_vec(self.test_y_char[idx]))).reshape(-1)
        
        self.test_x = np.asarray(self.test_x)
        self.test_y = np.asarray(self.test_y)
        
class ShakespeareObjectCrop_noniid:
    def __init__(self, data_path, dataset_prefix, n_client=100, crop_amount=2000, test_ratio=5, rand_seed=0):
        self.dataset = 'shakespeare'
        self.name    = dataset_prefix
        users, groups, train_data, test_data = read_data(data_path+'train/', data_path+'test/')

        # train_data is a dictionary whose keys are users list elements
        # the value of each key is another dictionary.
        # This dictionary consists of key value pairs as 
        # (x, features - list of input 80 lenght long words) and (y, target - list one letter)
        # test_data has the same strucute.
        # Why do we have different test for different clients?
        
        # Change structure to DatasetObject structure
        
        self.users = users

        test_data_count_per_client = (crop_amount//test_ratio)
        # Group clients that have at least crop_amount datapoints
        arr = []
        for client in range(len(users)):
            if (len(np.asarray(train_data[users[client]]['y'])) > crop_amount 
                and len(np.asarray(test_data[users[client]]['y'])) > test_data_count_per_client):
                arr.append(client)

        # choose n_client clients randomly
        self.n_client = n_client
        np.random.seed(rand_seed)
        np.random.shuffle(arr)
        self.user_idx = arr[:self.n_client]
          
        self.client_x = list(range(self.n_client))
        self.client_y = list(range(self.n_client))
        
        test_data_count = 0

        for client, idx in enumerate(self.user_idx):
            np.random.seed(rand_seed + client)
            start = np.random.randint(len(train_data[users[idx]]['x'])-crop_amount)
            self.client_x[client] = np.asarray(train_data[users[idx]]['x'])[start:start+crop_amount]
            self.client_y[client] = np.asarray(train_data[users[idx]]['y'])[start:start+crop_amount]

        test_data_count = (crop_amount//test_ratio) * self.n_client
        self.test_x = list(range(test_data_count))
        self.test_y = list(range(test_data_count))
        
        test_data_count = 0

        for client, idx in enumerate(self.user_idx):
            
            curr_amount = (crop_amount//test_ratio)
            np.random.seed(rand_seed + client)
            start = np.random.randint(len(test_data[users[idx]]['x'])-curr_amount)
            self.test_x[test_data_count: test_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['x'])[start:start+curr_amount]
            self.test_y[test_data_count: test_data_count+ curr_amount] = np.asarray(test_data[users[idx]]['y'])[start:start+curr_amount]
            test_data_count += curr_amount

        self.client_x = np.asarray(self.client_x)
        self.client_y = np.asarray(self.client_y)
        
        self.test_x = np.asarray(self.test_x)
        self.test_y = np.asarray(self.test_y)
        
        # Convert characters to numbers
        
        self.client_x_char = np.copy(self.client_x)
        self.client_y_char = np.copy(self.client_y)
        
        self.test_x_char = np.copy(self.test_x)
        self.test_y_char = np.copy(self.test_y)
        
        self.client_x = list(range(len(self.client_x_char)))
        self.client_y = list(range(len(self.client_x_char)))

        for client in range(len(self.client_x_char)):
            client_list_x = list(range(len(self.client_x_char[client])))
            client_list_y = list(range(len(self.client_x_char[client])))
            
            for idx in range(len(self.client_x_char[client])):
                client_list_x[idx] = np.asarray(word_to_indices(self.client_x_char[client][idx]))
                client_list_y[idx] = np.argmax(np.asarray(letter_to_vec(self.client_y_char[client][idx]))).reshape(-1)

            self.client_x[client] = np.asarray(client_list_x)
            self.client_y[client] = np.asarray(client_list_y)
                
        self.client_x = np.asarray(self.client_x)
        self.client_y = np.asarray(self.client_y)
        
        
        self.test_x = list(range(len(self.test_x_char)))
        self.test_y = list(range(len(self.test_x_char)))
                
        for idx in range(len(self.test_x_char)):
            self.test_x[idx] = np.asarray(word_to_indices(self.test_x_char[idx]))
            self.test_y[idx] = np.argmax(np.asarray(letter_to_vec(self.test_y_char[idx]))).reshape(-1)
        
        self.test_x = np.asarray(self.test_x)
        self.test_y = np.asarray(self.test_y)
    
class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist':
            self.X_data = torch.tensor(data_x).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
            
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100' or self.name == "tinyimagenet":
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
            # self.transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])
        
            self.X_data = data_x
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = data_y.astype('float32')
                
        elif self.name == 'shakespeare':  
            self.X_data = data_x
            self.y_data = data_y
                
            self.X_data = torch.tensor(self.X_data).long()
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(self.y_data).float()
            
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'mnist' or self.name == 'synt' or self.name == 'emnist':
            X = self.X_data[idx, :]
            if isinstance(self.y_data, bool):
                return X
            else:
                y = self.y_data[idx]
                return X, y
        
        elif self.name == 'CIFAR10' or self.name == 'CIFAR100':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img # Horizontal flip
                if (np.random.rand() > .5):
                # Random cropping 
                    pad = 4
                    extended_img = np.zeros((3,32 + pad *2, 32 + pad *2)).astype(np.float32)
                    extended_img[:,pad:-pad,pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:,dim_1:dim_1+32,dim_2:dim_2+32]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y
                
        elif self.name == 'tinyimagenet':
            img = self.X_data[idx]
            if self.train:
                img = np.flip(img, axis=2).copy() if (np.random.rand() > .5) else img  # Horizontal flip
                if np.random.rand() > .5:
                    # Random cropping
                    pad = 8
                    extended_img = np.zeros((3, 64 + pad * 2, 64 + pad * 2)).astype(np.float32)
                    extended_img[:, pad:-pad, pad:-pad] = img
                    dim_1, dim_2 = np.random.randint(pad * 2 + 1, size=2)
                    img = extended_img[:, dim_1:dim_1 + 64, dim_2:dim_2 + 64]
            img = np.moveaxis(img, 0, -1)
            img = self.transform(img)
            if isinstance(self.y_data, bool):
                return img
            else:
                y = self.y_data[idx]
                return img, y

        elif self.name == 'shakespeare':
            x = self.X_data[idx]
            y = self.y_data[idx] 
            return x, y
            
class DatasetFromDir(data.Dataset):

    def __init__(self, img_root, img_list, label_list, transformer):
        super(DatasetFromDir, self).__init__()
        self.root_dir = img_root
        self.img_list = img_list
        self.label_list = label_list
        self.size = len(self.img_list)
        self.transform = transformer

    def __getitem__(self, index):
        img_name = self.img_list[index % self.size]
        # ********************
        img_path = os.path.join(self.root_dir, img_name)
        img_id = self.label_list[index % self.size]

        img_raw = Image.open(img_path).convert('RGB')
        img = self.transform(img_raw)
        return img, img_id

    def __len__(self):
        return len(self.img_list) 
