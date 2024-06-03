import torch
from utils import *
from dataset import Dataset
from torch.utils import data
from tqdm import tqdm
from time import time

class Client():
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        self.args = args
        self.device = device
        self.model_func = model_func
        self.received_vecs = received_vecs
        self.comm_vecs = {
            'local_update_list': None,
            'local_model_param_list': None,
        }
        
        if self.received_vecs['Params_list'] is None:
            raise Exception("CommError: invalid vectors Params_list received")
        self.model = set_client_from_params(device=self.device, model=self.model_func(), params=self.received_vecs['Params_list'])

        
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        self.dataset = dataset
        self.max_norm = 10
    
    def train(self):
        # local training
        self.model.train()
        #print('client train')
        #t_start=time()
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                #print(self.device)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                predictions = self.model(inputs)
                loss = self.loss(predictions, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                self.optimizer.step()
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs