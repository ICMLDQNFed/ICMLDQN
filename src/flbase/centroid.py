import os
import torch
import torch.nn.functional as F
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, SubsetRandomSampler

class Centroid():
    def __init__(self, n_classes, samples_data, samples_tar=None, Ctemp=1, CdecayFactor=0.9999):
        self.n_classes = n_classes
        self.centroids = torch.ones((n_classes, n_classes)) / n_classes
        self.CdecayFactor = CdecayFactor
        self.Ctemp = Ctemp
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.samples_data = samples_data
        # self.samples_tar = samples_tar.to(self.device)
        # self.dataset = dataset
        # self.label_list = torch.tensor(self.dataset.targets)

    def update_batch(self, model, targets, sample_size=8):
        # target_cpu = targets.cpu()
        # Classes =  target_cpu.unique()
        # device = next(model.parameters()).device
        # model.train()
        # output = F.softmax(logit.float(), 1)

        # with torch.inference_mode():

        # idx_list = [np.random.choice(self.class_loaders[i], sample_size, replace = False) for i in range(100)]
        # idx_list = np.concatenate(idx_list, 0)
        # sample_loader = DataLoader(self.dataset, batch_size = len(idx_list), sampler = SubsetRandomSampler(idx_list), pin_memory = True)
        # img, tar = next(iter(sample_loader))
        # img = img.to(device)
        # logits = model(img).detach()
        # output = F.softmax(logits.float(), 1)
        # for Class in range(100):
        #     self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
        #         (1- self.CdecayFactor) * torch.mean(output[tar.cpu() == Class], axis = 0).detach().cpu()

        # start = time.time()       
        # with torch.no_grad():
            # start = time.time()
        # img = self.samples_data[:, idx].view(-1,3,32,32)
        # tar_idx = torch.cat([5000 * i + idx for i in range(10)], 0)
        # tar = torch.index_select(self.samples_tar, 0, tar_idx)
        # end = time.time()
        # print('Sample time: {:.2f}'.format(end - start))
        # breakpoint()
        # start = time.time()
        with torch.no_grad():
            for Class in range(self.n_classes):
                if len(self.samples_data[Class]) <= 0:
                    continue
                idx = torch.randperm(len(self.samples_data[Class]))[:sample_size].to(self.device)
                img = torch.index_select(self.samples_data[Class].to(self.device), 0, idx).view(-1,3,32,32)
                logits = model(img).detach()
                output = F.softmax(logits.float(), 1)
                try:
                    self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
                        (1- self.CdecayFactor) * torch.mean(output, axis = 0).detach().cpu()
                except Exception:
                    breakpoint()
                    # break         
            # idx = (self.label_list == Class).nonzero().squeeze()
            # sample_idx = np.random.choice(idx, sample_size, replace = False)
            # end = time.time()
            # print('Inference time: {:.2f}'.format(end - start))

            # p_dist = 1 / len(idx) * torch.ones(len(idx))
            # sample_idx = idx[p_dist.multinomial(sample_size)]
            # tmp_dataset = Subset(self.dataset, sample_idx)
            # tmp_loader = DataLoader(tmp_dataset, batch_size = sample_size)
            # # breakpoint()

            # img = torch.zeros((sample_size, 3, 32, 32))
            # for i, elem in enumerate(sample_idx):
            #     img[i], _ = self.dataset[elem]
            # # img = torch.tensor(self.dataset.data[sample_idx]).to(device)
            # img = img.to(device)
            # logits = model(img).detach()
            # output = F.softmax(logits.float(), 1)
            # self.centroids[Class]/home/multicompc15/Documents/cifar-opl-sample-correct = self.CdecayFactor * self.centroids[Class] + \
            #     (1- self.CdecayFactor) * torch.mean(output, axis = 0).detach().cpu()
            
            # for img, tar in tmp_loader:
            #     if tar.sum() / sample_size != Class:
            #         print('WRONG!!!!!')
            #         breakpoint()
            #     img, tar = img.to(device), tar.to(device)
            #     # img = img.to(device)
            #     logits = model(img).detach()
            #     output = F.softmax(logits.float(), 1)
            #     self.centroids[Class] = self.CdecayFactor * self.centroids[Class] + \
            #         (1- self.CdecayFactor) * torch.mean(output, axis = 0).detach().cpu()
            

        self.centroids =  self.centroids/(self.centroids.sum(1)[:,None])
        # breakpoint()

    def update_epoch(self, model, data_loader):
        self.centroids = torch.zeros_like(self.centroids)
        model.train()
        device = next(model.parameters()).device
        with torch.inference_mode():
            for image,target in data_loader:
                image,target = image.to(device), target.to(device)
                logit = model(image).detach()

                Classes =  target.cpu().unique()
                logit = logit.cpu()
                output = F.softmax(logit.float(), 1)
                
                for Class in Classes:
                    self.centroids[Class] += torch.sum(output[target.cpu() == Class], axis = 0)

            self.centroids =  self.centroids/(self.centroids.sum(1)[:,None])
    
    def get_centroids(self, target):
        return torch.index_select(self.centroids, 0, target.cpu()).to(target.device)
