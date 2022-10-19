import requests
import json
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

days = ['0', '31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']
months = ['0', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

in_data_ori = []
out_data_ori = []

for year in range(2015, 2023):
    for month in range(1, 13):
        url = "https://webapi.sporttery.cn/gateway/jc/football/getMatchResultV1.qry?matchPage=1&matchBeginDate=" + str(year) + "-" + months[month] + "-01&matchEndDate=" + str(year) + "-" + months[month] + "-" + days[month] + "&leagueId=&pageSize=999999&pageNo=1&isFix=0&pcOrWap=1"
        res = json.loads(requests.get(url).text)['value']['matchResult']
        for item in res:
            if item['a'] != '' and item['d'] != '' and item['h'] != '' and item['winFlag'] != '':
                print(item['a'], item['d'], item['h'], item['winFlag'])
                in_data_ori.append([float(item['a']), float(item['d']), float(item['h'])]) 
                h = 0
                d = 0
                a = 0
                if str(item['winFlag']) == 'H':
                    h = 1
                elif str(item['winFlag']) == 'D':
                    d = 1
                elif str(item['winFlag']) == 'A':
                    a = 1
                out_data_ori.append([h, d, a])

testnum = 100
train_in_data = in_data_ori[:len(in_data_ori) - testnum]
test_in_data = in_data_ori[len(in_data_ori) - testnum:]
train_out_data = out_data_ori[:len(out_data_ori) - testnum]
test_out_data = out_data_ori[len(out_data_ori) - testnum:]

train_in_data_tensor = torch.from_numpy(np.array(train_in_data)).float()
train_out_data_tensor = torch.from_numpy(np.array(train_out_data)).float()
test_in_data_tensor = torch.from_numpy(np.array(test_in_data)).float()
test_out_data_tensor = torch.from_numpy(np.array(test_out_data)).float()

train_dataset = TensorDataset(train_in_data_tensor, train_out_data_tensor)
test_dataset = TensorDataset(test_in_data_tensor, test_out_data_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)