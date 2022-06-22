#!/usr/bin/env python
# coding: utf-8

# In[2]:


from comet_ml import Experiment

experiment = Experiment(
    api_key="Qta9GieqkZxXGHSrnJrg55z3O",
    project_name="cnn2d-kappa-apnea-regression",
    workspace="martaferpoy",
)
experiment.set_name("Kappa: 128 canales, 0.3 dropout, 16 kernel, 8 capas")


# In[1]:


import numpy as np 
import pandas as pd
from statistics import mean
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import time
import copy
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support


# In[2]:


hiperparametros = {
    'epocas' : 200,
    'lr' : 0.001,
    'batch_size' : 32,
    'seed' : 56389856,
    'dropout' : 0.3
}


# In[3]:


experiment.log_parameters(hiperparametros)


# In[3]:


import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary


# In[4]:


torch.cuda.is_available()


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


# In[6]:


device


# # Open data

# In[8]:


x_file_t = open('C:/Users/GIB/Documents/Marta/x_train.pkl', 'rb')
X_train = pickle.load(x_file_t)

y_events_file_t = open('C:/Users/GIB/Documents/Marta/y_events_train.pkl', 'rb')
y_events_train = pickle.load(y_events_file_t)

pacient_name_file_t = open('C:/Users/GIB/Documents/Marta/pacient_name_train.pkl', 'rb')
names_train = pickle.load(pacient_name_file_t)


x_file_v = open('C:/Users/GIB/Documents/Marta/x_validation.pkl', 'rb')
X_validation = pickle.load(x_file_v)

y_events_file_v = open('C:/Users/GIB/Documents/Marta/y_events_validation.pkl', 'rb')
y_events_validation = pickle.load(y_events_file_v)

pacient_name_file_v = open('C:/Users/GIB/Documents/Marta/pacient_name_validation.pkl', 'rb')
names_validation = pickle.load(pacient_name_file_v)


# In[14]:


len(X_train), len(X_validation), len(y_events_train), len(y_events_validation), len(names_train), len(names_validation)


# In[15]:


X_train[0].shape, X_validation[0].shape


# # Dataset class

# In[16]:


class MyDataset(Dataset):
    def __init__(self, x, y, names) -> None:
        self.x = x
        self.y = y
        self.names = names

    def __getitem__(self, index):
        x_ind = self.x[index]
        y_ind = self.y[index]
        names_ind = self.names[index]

        x = torch.from_numpy(x_ind).float()
        y = torch.Tensor([y_ind]).float()

        return x.reshape(1, x.shape[0], x.shape[1]), y, names_ind

    def __len__(self):
        return len(self.x)


# In[17]:


dataset = MyDataset(X_train, y_events_train, names_train)


# In[18]:


dataset_val = MyDataset(X_validation, y_events_validation, names_validation)


# # Dataloader

# In[19]:


dataloader = DataLoader(dataset, batch_size = hiperparametros['batch_size'], shuffle = True) 


# In[20]:


dataloader_val = DataLoader(dataset_val, batch_size = hiperparametros['batch_size'], shuffle = True)


# In[21]:


print(len(dataloader.dataset))


# # Model

# In[7]:


class MyModel(nn.Module): 
    def __init__(self):
        
        super().__init__() 

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (1, 16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (1, 16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 48, kernel_size = (1,16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size = (1, 16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 80, kernel_size = (1, 16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels = 80, out_channels = 96, kernel_size = (1, 16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = 112, kernel_size = (1, 16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 112),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels = 112, out_channels = 128, kernel_size = (1, 16), stride= (1,2)) ,
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1,2), stride = (1,1)),
            nn.Dropout(p = hiperparametros['dropout'])
        )

        self.lin = nn.Linear(in_features = 128* 2* 30, out_features = 1)
    

    def forward(self,x): 
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        # Flatten
        out = out.reshape(out.shape[0], out.shape[1]*out.shape[2]*out.shape[3])
        out = self.lin(out)


        return out 


# In[8]:


torch.manual_seed(hiperparametros['seed'])
model = MyModel().to(device)


# In[9]:


summary(model, input_size = (1, 2, 12000))


# In[25]:


criterion = torch.nn.HuberLoss(reduction= "mean")
optimizer = torch.optim.Adam(model.parameters(), lr=hiperparametros['lr'])


# # Bucle de entrenamiento

# In[26]:


def validate_loss_kappa_acc(val_loss, val_kappa, val_accuracy):
    model.eval()
    loss_medio_val = 0
    pasos_val = 0
  
    dict_patients_val = {}
    outputs_list_val = torch.empty(0).to(device)
    labels_list_val = torch.empty(0).to(device)
    names_list_val = ()

    with torch.no_grad():
        for data in dataloader_val:
            inputs, labels, names = data[0].to(device), data[1].to(device), data[2]
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            loss_medio_val += loss_val.item()
            outputs_list_val = torch.cat((outputs_list_val, outputs))
            labels_list_val = torch.cat((labels_list_val, labels))
            names_list_val = names_list_val + names

            pasos_val += 1
      
    val_loss.append(loss_medio_val/pasos_val)
    experiment.log_metric('Loss validation',loss_medio_val/pasos_val)

    dict_patients_val = group_pacients(outputs_list_val.detach().to('cpu').numpy(), labels_list_val.detach().to('cpu').numpy(), names_list_val, dict_patients_val)

    kappa, accuracy = get_AHI_kappa_acc(dict_patients_val)

    val_kappa.append(kappa)
    val_accuracy.append(accuracy)
    
    experiment.log_metric('Kappa validation',kappa)
    experiment.log_metric('Accuracy validation',accuracy)
    
    return(val_loss, val_kappa, val_accuracy, loss_medio_val/pasos_val, kappa, accuracy)


# In[27]:


def group_pacients(output_batch, labels_batch, names_batch, dict_p):
    for i in range(output_batch.shape[0]):
        output = output_batch[i]; label = labels_batch[i]; name = names_batch[i]

        if name in [*dict_p]:
            dict_p[name].append([output, label])
        else:
            dict_p[name] = [[output, label]]

    return dict_p


# In[28]:


def get_AHI_kappa_acc(dictionary):
    AHI_pred = []
    AHI_real = []
  
    for key in [*dictionary]:
        individual = dictionary[key]
        outputs_ind = [item[0] for item in individual]
        labels_ind = [item[1] for item in individual]

        AHI_pred.append(sum(outputs_ind)/len(outputs_ind)*3)
        AHI_real.append(sum(labels_ind)/len(labels_ind)*3)

    AHI_pred_disc = np.digitize(AHI_pred, bins = np.array([5,15,30]))
    AHI_real_disc = np.digitize(AHI_real, bins = np.array([5,15,30]))

    kappa = cohen_kappa_score(AHI_pred_disc, AHI_real_disc)

    accuracy = sum(AHI_pred_disc == AHI_real_disc).item()/len(AHI_real_disc)
    
    return kappa, accuracy


# In[29]:


loss_list =[]; validation_loss_list = []
kappa_list = []; validation_kappa_list = []
accuracy_list = []; validation_accuracy_list = []


loss_min = 10000000
kappa_max = -100000
accuracy_max = -1000
best_model = None
best_outputs = torch.empty(0).to(device)
best_labels = torch.empty(0).to(device)
best_names = ()

tiempo = time.time()

for epoch in range(hiperparametros['epocas']):
    experiment.set_epoch(epoch)

    loss_medio = 0
    pasos = 0

    dict_patients = {}
    outputs_list = torch.empty(0).to(device)
    labels_list = torch.empty(0).to(device)
    names_list = ()

    # Bucle de entrenamiento
    for i,data in enumerate(dataloader, 0): 
        inputs, labels, names = data[0].to(device), data[1].to(device), data[2]
        optimizer.zero_grad() 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_medio += loss.item()
        loss.backward() 
        optimizer.step() 

        outputs_list = torch.cat((outputs_list, outputs))
        labels_list = torch.cat((labels_list, labels))
        names_list = names_list + names

        pasos += 1

    print("Epoch = ", epoch)
    loss_list.append(loss_medio/pasos)
    experiment.log_metric('Loss train',loss_medio/pasos)

    dict_patients = group_pacients(outputs_list.detach().to('cpu').numpy(), labels_list.detach().to('cpu').numpy(), names_list, dict_patients)
    
    kappa, accuracy = get_AHI_kappa_acc(dict_patients)

    kappa_list.append(kappa)
    accuracy_list.append(accuracy)
    experiment.log_metric('Kappa train',kappa)
    experiment.log_metric('Accuracy train',accuracy)

    validation_loss_list, validation_kappa_list, validation_accuracy_list, loss_actual, kappa_actual, accuracy_actual = validate_loss_kappa_acc(validation_loss_list, validation_kappa_list, validation_accuracy_list)

    # chequeo la metrica que tengo en esta epoca, y si es mejor que en epocas anteriores, guardo el modelo
    if(kappa_actual > kappa_max):
        loss_min = loss_actual
        kappa_max = kappa_actual
        accuracy_max = accuracy_actual
        best_model = copy.deepcopy(model)
        best_outputs = outputs_list
        best_labels = labels_list
        best_names = names_list
    
    model.train()
        
experiment.log_metric('Best loss validation', loss_min)
experiment.log_metric('Best kappa validation', kappa_max)
experiment.log_metric('Best accuracy validation', accuracy_max)

print("Time required: ", time.time() - tiempo)

# In[41]:


print('Loss en entrenamiento = ', loss_list[len(loss_list)-1])
print('Loss en validacion = ', validation_loss_list[len(validation_loss_list)-1])
print('Kappa en entrenamiento = ', kappa_list[len(kappa_list)-1])
print('Kappa en validacion = ', validation_kappa_list[len(validation_kappa_list)-1])
print('Accuracy en entrenamiento = ', accuracy_list[len(accuracy_list)-1])
print('Accuracy en validacion = ', validation_accuracy_list[len(validation_accuracy_list)-1])


# In[64]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(loss_list, color = "coral", label = "train")
plt.plot(validation_loss_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n', color = 'black')
plt.ylabel('\n \n Loss', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
# plt.show()
experiment.log_figure(figure_name='Loss_plot', figure = plt)
plt.clf()
plt.close()


# In[68]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(kappa_list, color = "coral", label = "train")
plt.plot(validation_kappa_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n', color = 'black')
plt.ylabel('\n \n Kappa', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
# plt.show()
experiment.log_figure(figure_name='Kappa_plot', figure = plt)
plt.clf()
plt.close()


# In[67]:


plt.rcdefaults()
plt.figure(figsize = (5,3))
plt.plot(accuracy_list, color = "coral", label = "train")
plt.plot(validation_accuracy_list, color = "green", label = "validation")
plt.xlabel('Epoch \n \n', color = 'black')
plt.ylabel('\n \n Accuracy', color = 'black')
plt.grid(linestyle = 'dotted')
plt.legend()
# plt.show()
experiment.log_figure(figure_name='Accuracy_plot', figure = plt)
plt.clf()
plt.close()


# In[36]:


torch.save(best_model.state_dict(), 'C:/Users/GIB/Documents/Marta/trained_models/kappa2d/kappa2d_03drop128canales16kernel8capas.pth')


# In[104]:


experiment.log_notebook('kappa2d_128canales_03drop_16kernel_8capas.ipynb')
experiment.end()

