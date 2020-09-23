# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 14:40:48 2020

@author: h5904
"""
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datetime import date, timedelta, datetime
 
from PIL import Image 
import matplotlib.pyplot as plt
import pandas as pd
 
from sklearn.manifold import TSNE
import os

class BuyerAmount(object):
 
  def __init__(self):
 
    #Default data structure:
    ########################################################################################
    #    {'日期1' : {
    #         分點1 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
    #         分點2 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
    #         分點3 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
    #            .
    #            .
    #              }
    #
    #    '日期2' : {
    #         分點1 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
    #         分點2 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
    #         分點3 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
    #            .
    #            .
    #              }
    ########################################################################################
 
     #New data structure:
    ########################################################################################
    #    {'日期1' : [
    #           [ 股票1 , 股票2 , 股票3 ] <= 分點1
    #           [ 股票1 , 股票2 , 股票3 ] <= 分點2
    #           [ 股票1 , 股票2 , 股票3 ] <= 分點3
    #             .
    #             .
    #              ]
    #
    #    '日期2' : {
    #           [ 股票1 , 股票2 , 股票3 ] <= 分點1
    #           [ 股票1 , 股票2 , 股票3 ] <= 分點2
    #           [ 股票1 , 股票2 , 股票3 ] <= 分點3
    #             .
    #             .
    #              ]
    ########################################################################################
    #Google drive path
    self.BuyAmountList_path = "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock Data"
    self.test_case = self.PklToData( "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock_Information/Buyer_dic")  # 所有券商 { (6802,6832):"凱基-虎尾" ,....} 
    self.buyer_index = self.PklToData( "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock_Information/Buyer_index" ) # 券商 index 對照表 { "凱基-虎尾" : 0 , "凱基-後甲" : 1 ,....} 
    self.stock_index = self.PklToData( "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock_Information/Stock_List" ) # Stock index 對照表 => {"1101":0 , "1102": .....}
 
    #init process 
    self.BuyAmountList = None # BuyAmountList(Dict)
    self.dates = None
    self.rawdata = None
    self.load_pkl_data() # Laoding PKL data
    self.all_dates() #show & collect all keys(dates)
    
 
 
  def load_pkl_data(self):
 
    self.BuyAmountList = {}
    print("loading data.....")
    for path in os.listdir(self.BuyAmountList_path):
      with open( os.path.join( self.BuyAmountList_path, path) ,'rb') as pkl :
        self.BuyAmountList.update(pickle.load(pkl))
      
    print("loading data sucess!")
 
  def all_dates(self):
 
    return [key for key in self.BuyAmountList.keys()]
 
  def Data_date_select(self ,dic ,dates = None ,merge = True):
    # merge request 
    if dates == None:
      dates = self.all_dates()
   
    if merge: 
      date_list = [ dic[date] for date in dates ]
      if len(date_list) == 1: #dicts only 1 element , return directly
        return date_list[0] 
      
      date_list = data_merge(date_list)  
      return date_list
    
    dicts = { date : dic[date] for date in dates  }
    return dicts #dicts > 1 element , no merge request, return dicts [note that this is list with dicts!] 
 
    
  @staticmethod
  def drop_cln_row(df_buy ,drop_portion = "Buyer" ,Buyer_threshold = None ,Upper_bound = None , Lower_bound = None):
    
    sum_axis = 0 if drop_portion == "Buyer" else 1 
    drop_axis = 1 if drop_portion == "Buyer" else 0 
 
    if drop_portion == "Buyer":
      #drop_idx_buyer
      df_sum_0 = df_buy.abs().sum(sum_axis)
      #df_sum_0_re = df_sum_0.groupby(pd.cut(df_sum_0,np.arange(0,180000,100))).count() #See the bar-bin distribution of amount 
      drop_idx = df_sum_0[df_sum_0 < Buyer_threshold].index # 分點買賣<800張的濾掉
      df_buy.drop( drop_idx ,axis = drop_axis,inplace=True)
 
    if drop_portion == "Stock":
      #drop_idx_stock
      df_sum_1 = df_buy.abs().sum(axis=1) 
      drop_idx_stock_1 = df_sum_1[df_sum_1 > Upper_bound].index
      drop_idx_stock_2 = df_sum_1[df_sum_1 < Lower_bound].index
      df_buy.drop(drop_idx_stock_1,axis = 0,inplace=True)
      df_buy.drop(drop_idx_stock_2,axis = 0,inplace=True)
 
    return df_buy #dataframe
  
  @staticmethod
  def buyer_seller_filter(df_buy,filter = "buy"):
 
    df_filter = df_buy.copy()
    if filter == "buy":
      df_filter[df_buy <= 0] = 0 
    # else == "seller"
    if filter == "sell": 
      df_filter[df_buy >= 0] = 0 
    
    return df_filter
  
  def DataToPkl(self , path , data ):
 
    with open( path ,'wb') as pkl :
          pickle.dump( data , pkl )       
          
  def PklToData(self , path ):
      
    with open( path ,'rb') as pkl :
        data = pickle.load( pkl )
 
    return data
 
# Merge and add dicts element and return the added dicts
def data_merge(data_list):
 
  mat_sum_np = 0
  for data in data_list :
    mat = np.array(data) 
    mat_sum_np += mat
  
  return mat_sum_np.tolist()
 
 
def plot(df_buy,size,return_ = True):
 
  Resize = size
  Transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size = (Resize,Resize)),
    transforms.ToTensor(),
  ])
  df_nplist = np.array(df_buy.abs().values.tolist()) # change to np 
  df_nplist_ts = torch.Tensor((df_nplist)) # change to torch.Tensor
  df_nplist_ts = Transformer(df_nplist_ts) # Resize 
  plt.imshow(df_nplist_ts[0]) 
 
  if return_ == True:
    return df_nplist_ts.numpy()
 
 
#Normalize pic 
def normalize(image):
  return (image - image.min()) / (image.max() - image.min())
 
# reducing form of SVD decomposition
def reduce_svd_mat(mat_u ,mat_s ,mat_vt ,top = 30):
  
  mat_s = np.diag(mat_s[:top])
  mat_u = mat_u[:,:top]
  mat_vt = mat_vt[:top,:]
  return mat_s , mat_u , mat_vt
 
 
 
 
 
####################### Swap the key of dict ####################
 
# the depth of dict
def dict_depth(dic):
    if isinstance(dic, dict): # if is dict type 
        return 1 + (max(map(dict_depth, dic.values())) if dic else 0) # layer+1 and do recursion on the dict.values , and catch the max value in the recursion of dict.value
    return 0
 
# permute lists
def permutation(lists ,permute ):
 
      lists = np.array(lists)
      lists = np.transpose(lists, permute) 
      lists = lists.tolist()  
      return lists 
# transform dict to list 
def DicToMat(dic  , layer_index  , layer_number  , mat_elemet = None  , mat_keyvaluemap = None ):
  
    # Try : re-change the index-keymap to the following flat-form : {"1":[9802,2230],"2":[10/2,10/3,10/4,10/5],"3":[1,2,3]} which is easy to search ?!  
 
    for i , key in enumerate(dic):
 
        # create the layer-key dict : mat_keyvaluemap.
        if layer_index not in mat_keyvaluemap:
            mat_keyvaluemap.update({layer_index:[key]})  
        elif key not in mat_keyvaluemap[layer_index]:
            mat_keyvaluemap[layer_index].append(key)  
            
        if layer_index > 1 :
            mat_elemet.append([]) # if layer_index 不等於 1 (bottom condition) , 則持續加入空 list
            DicToMat(dic = dic[key] , layer_index = layer_index-1 ,layer_number = layer_number ,mat_elemet = mat_elemet[i] ,mat_keyvaluemap = mat_keyvaluemap )  # 利用 recursion 來處理不定layer數目的 dict , layer = layer_index-1 , mat_elemet = mat_elemet[i] 
        else :
            mat_elemet.append(dic[key]) #如果已在最深一層裡, 直接寫入 data  
    return mat_elemet , mat_keyvaluemap
 
# transform list to dict
def MatToDic(mat , maps , layer_index  , layer_number  , dic = None, permute = None ):
    
    # Try : re-change the index-keymap to the following flat-form : {"3":[9802,2230],"2":[10/2,10/3,10/4,10/5],"1":[1,2,3]} which is easy to search ?!
      
    #創造 permutation - index_layer 比對 pairs
    idx_layer_pairs = {}
    for idx ,keys in enumerate(maps.keys()):
        idx_layer_pairs.update({idx:keys})
 
    
    NOW_IN_LAYER = idx_layer_pairs[permute[layer_number-layer_index]] # 從 permute 順序 0 開始 , 逐 layers construct keys in dict  
    if layer_index > 1 :     
        for i , row in enumerate(mat):
            
            key = maps[ NOW_IN_LAYER ][ i ]  # 從 permute 順序 0 開始 , 逐 layers construct keys in dict   
            dic.update({key:{}})  # if layer_index 不等於 1 (not bottom condition) , 則持續加入空 dict (並 construct keys in dict ) 
            MatToDic(mat = mat[i], maps = maps , layer_index = layer_index - 1 , layer_number = layer_number ,  dic = dic[key] , permute = permute  )  # 利用 recursion 來處理不定layer數目的 dict , layer = layer_index-1 , mat = mat[i] , dic = dic[key] 
          
    else :
        for k , ele in enumerate(mat):
            
            key , value = maps[ NOW_IN_LAYER ][ k ] , mat[k]         
            if key not in dic:
                dic.update( { key :  value }  ) #如果已在最深一層裡, 直接寫入 data  
                         
    return dic 
 
#swap keys 
def Swap_Key(dic ,permute ):    
 
    dic_depth = dict_depth(dic) #get the depth of dict
 
    mat_elemet , mat_keyvaluemap = DicToMat( dic  , layer_index = dic_depth , layer_number = dic_depth , mat_elemet = [] , mat_keyvaluemap = {}) # transform dict to matrix
    mat_elemet = permutation(mat_elemet , permute) # matrix permutation 
    new_dic = MatToDic( mat = mat_elemet , maps = mat_keyvaluemap , layer_index = dic_depth , layer_number = dic_depth , permute = permute , dic = {}) # transform matrix to dict
 
    return new_dic 
 
#######################################################
# remove holiday
def delete_key(dicts):
  
  delete_keys = []
  dict_keys = [key for key in dicts.keys()]
  for key in dict_keys:  
    mat = np.array(dicts[key])
    value = np.sum(np.sum(mat,axis = 0),axis = 0)
    if value == 0:
      delete_keys.append(key)
      del dicts[key]
  
  return dicts , delete_keys


def visualize_plot(data , reduced_dim = 30 , dates = None , perplexity = 30, Buyer_threshold = 5000 , Upper_bound=100000 , Lower_bound=5000 , buy_or_sell = "buy" , embed = "buyer"):
 
  B = BuyerAmount()
  #initialize data
  df = pd.DataFrame(data)
 
  #filter data
  df_buy = B.buyer_seller_filter(df ,filter = buy_or_sell )
  df_buy = B.drop_cln_row(df_buy ,drop_portion="Buyer" ,Buyer_threshold = Buyer_threshold )
  df_buy = B.drop_cln_row(df_buy ,drop_portion="Stock" ,Upper_bound=Upper_bound ,Lower_bound=Lower_bound )
 
  #get SVD decomposition
  np_buy = np.array(df_buy.values.tolist()) # transform df to np.array
  u ,s ,vt = np.linalg.svd(np_buy ,full_matrices=True)
  mat_s , mat_u , mat_vt = reduce_svd_mat(u ,s ,vt , top = reduced_dim)
  print("u shape:",mat_u.shape , "s shape:",mat_s.shape , "vt shape:",mat_vt.shape)
 
  #get embedding vector
  latent_vec = mat_vt.T if embed == "buyer" else mat_s
  embed = TSNE(n_components=2,perplexity=perplexity).fit_transform(latent_vec)
  
  # filter outliers #
  # 3169 7/11~16 no data bugs!!! 來自 stocklist_url 裡面 3169 減資上櫃被刪除 bugs !!! #
 
  data = embed
  x, y = data.T
  plt.scatter(x,y)
  plt.show()