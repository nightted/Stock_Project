# -*- coding: utf-8 -*-
# 買賣超總資料
import requests
from datetime import date, timedelta, datetime
from bs4 import BeautifulSoup
import time
import numpy as np 
import pickle


# Following: 接下來直接將以下抓取函示寫成一包 Class:
# Dict 內新增股票欄位function & 抓取券商function獨立出來 
# => (初始化Dict = {} 放 __init__ 內)  
# => (新增股票欄位func獨立出來 class.new('Stocknumber')) 
# => (抓取券商func獨立出來 class.grab('Stocknumber,start,end,) , 另判斷 if 股票in Dict or not? ) 

class Stock_Crawler(object):

    def __init__(self):
        
        # Main list        
        self.BuyAmountList = {}      
        # URl and cookies 
        self.stocklist_url = 'https://stock.wespai.com/pick/choice'
        self.data_header = [
            {'qry[]': 'dv', 'id[]': 'dv', 'val[]': '0;12000'},
            {'cookie': '__utmz=140509424.1585664337.12.6.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utma=140509424.970758813.1562249232.1585762091.1586178222.15; __utmc=140509424; __utmt=1; __utmb=140509424.4.10.1586178222',
             'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
            ]   
        self.buyerlist_url = 'http://5850web.moneydj.com/z/js/zbrokerjs.djjs'
        self.crawler_url = 'http://5850web.moneydj.com/z/zg/zgb/zgb0.djhtm?a={l}&b={m}&c=E&e={i}-{j}-{k}&f={i}-{j}-{k}'
        
        
        try:
            self.PklToData() # Loading the pickel data to self.BuyAmountList first
            
        except FileNotFoundError as e:
            print("this is the first time to create the dict!")
            
    def BuyerAmount_Crawler(self,Date): #這可以是classmethod or staticmethod

        CrawlDay = str(datetime.strptime(Date, "%y/%m/%d")) # 要抓取的起始點
        test_case = self.All_the_Buyer_List()  # 所有券商

        try:  # 檢查一下新增的"日期"有沒有在BuyAmountList裡,有就 pass 繼續新增其他天 data, 沒的話就新增這支股票 .
            if self.BuyAmountList[CrawlDay]:
                pass

        except KeyError:
            self.Generator_Specified_Date(CrawlDay)
        
        start_time = time.time()
        count = 1
        for point in test_case:
            for sub_point in test_case[point]:
                
                dayarray = CrawlDay.split('-')
                # 分點買賣超的網頁爬蟲
                url = self.crawler_url.format(i=dayarray[0], j=dayarray[1], k=dayarray[2][:2], l=point, m=sub_point)
                res = requests.get(url)
                soup = BeautifulSoup(res.text, 'html.parser')
                lists = [str(x) for x in soup.findAll('td', class_=['t4t1'])]  # 特定券商&買賣超統計
                    
                #Question : 如何在這裡一次把單一分點的data全部餵進所有分點dict裡面!!!
                #發想2 : 可產生空的股票清單 , 並比對 lists 內 element 做填入 , 後續再把清單丟到 self.BuyAmountList 就好  
                
                All_stock_member = self.Generator_StockAllList() #產生所有股票清單
                StockListDummy = self.FindStockNumber(lists) #找到目前爬取券商前五十名的買賣超股票清單
                # 將前五十名的買賣超股票清單買賣超data , update 進 All_stock_member dict 裡
                for idx , stock in enumerate(StockListDummy):              
                    if stock in All_stock_member:                    
                        Target_amount = self.StrToDigit(soup.select('.t3n1')[idx * 3 + 2].string) # 總買賣超格數為3的倍數格
                        All_stock_member[stock] = Target_amount 
                        
                self.BuyAmountList[CrawlDay][test_case[point][sub_point]].update(All_stock_member) # Update All_stock_member 進 BuyAmountList 裡
                print("Now crawling date:{}, ".format(CrawlDay),"Finish [{a}/{b}] = {c}%".format(a=count,b=len(self.BuyAmountList[CrawlDay]),c=count/len(self.BuyAmountList[CrawlDay])*100),", Waste {} sec".format(time.time()-start_time),"and now in:",test_case[point][sub_point])
                count+=1
                # 分點資料找不到券商,開始往各股買賣超前十五名找(補償分點買賣超的不足, EX:摩根史丹利大量買超只列到前五十,股本比較小的股票就不會被顯示在上面)
        
        self.DataToPkl() # Save to pkl while finishing grabbing data 
        
        print('Finish!!!')
            
        
    # Transform str digit with ',' to int
    def StrToDigit(self, Number):

        Number = ''.join(Number.split(','))
        return int(Number)

    def FindStockNumber(self,lists):
        List = []
        for ele in lists:
            dummy = ele.split("'")[1][2:]
            List.append(dummy) 
            
        return List

    def Generator_Specified_Date(self, Date):
    
########################################################################################
#           # data structure
#           # (Now change to) Data Structure : 
#                      {'日期1' : {
#                               分點1 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                               分點2 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                               分點3 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                                .
#                                .
#                                .
#                            }
#
#                      '日期2' : {
#                               分點1 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                               分點2 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                               分點3 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                                .
#                                .
#                                .
#                            }
########################################################################################
            Date = str(Date)
            self.BuyAmountList.update({Date: {}})
            for mem in self.All_the_Buyer_List().values():
                for mem2 in mem.values():
                    self.BuyAmountList[Date].update({mem2:{}})
                
    # Return所有券商跟分點成 Dict
    def All_the_Buyer_List(self):
        
        def List_to_Dic(List):
            All_list_number = list(map(lambda x: x.split(',')[0] if (',' in x) else '', List))[:len(List)]
            All_list_name = list(map(lambda x: x.split(',')[1] if (',' in x) else '', List))[:len(List)]
            Res = dict(zip(All_list_number, All_list_name))
            return Res

        # 這邊之後可以把List存在SQL要用再呼叫
        url = self.buyerlist_url
        res = requests.get(url)
        res.encoding = 'big5'
        res = res.text[21:17964].split(';')  # 暴力去除雜訊 XDD

        All_list, AllList = [], {}
        for ele in res:
            All_list += [ele.split('!')]
        for i in range(len(All_list)-11): # 暴力去除雜訊 XDD
            AllList[All_list[i][0].split(',')[0]] = List_to_Dic(All_list[i])
        return AllList
 
   # 產生所有Stcok List(之後會把它存到SQL)
    def Generator_StockAllList(self):       
     
        ######################
        #初始畫表格部分要修改成(Now change to)樣子
        ######################
        url = self.stocklist_url
        data = self.data_header[0]
        headers = self.data_header[1]
        res = requests.post(url, data=data, headers=headers)
        res = res.text.split("],[")

        Stock_List = {}
        for ele in res:
            ele = ele.split(',')[0][1:len(ele.split(',')[0]) - 1]
            Stock_List.update({ele: 0}) #set default buyamount as 0
        Stock_List['1101'] = Stock_List['["1101']
        del Stock_List['["1101']  # 暴力修亂碼法XDD

        return Stock_List   
    
    #Transform 2keys dict to list(2D)
    def DicToMat_2D(self, dic):
        
        mat_elemet = []
        mat_keyvaluemap = []
        for i , key in enumerate(dic):           
            mat_elemet.append([])
            mat_keyvaluemap.append([])
            
            for j , subkey in enumerate(dic[key]):             
                mat_elemet[i].append(dic[key][subkey])
                mat_keyvaluemap[i].append((key,subkey))
        
        return mat_elemet , mat_keyvaluemap
                
    #Transform 2D list to dict(2keys)       
    def MatToDic_2D(self, mat, maps , swap = False):
        
        def transpose(lists):        
            lists = np.array(lists)
            lists = np.swapaxes(lists,1,0).tolist()
            return lists    
        
        dic = {}
        key_idx,subkey_idx = 0 , 1 
        if swap:   
            #if want to swap the dict key ,value 
            maps , mat = transpose(maps) ,transpose(mat)
            key_idx,subkey_idx = 1 , 0   # need reverse the priority of key&subkey due to transpose of maps matrix  
                               
        for i , row_ele in enumerate(mat):      
            for j , col_ele in enumerate(mat[i]):             
                key ,subkey , value = maps[i][j][key_idx], maps[i][j][subkey_idx], mat[i][j]          
                if key not in dic:
                    dic.update( { key : { subkey : value } } )
                else:
                    dic[key].update({subkey : value})                                   
        return dic
    
     #Swap the priority of key - subkey 
    def Swap_Key_Subkey(self,dic, swap = True):
    
        mat ,maps = self.DicToMat_2D(dic)
        dic = self.MatToDic_2D(mat, maps, swap)
        return dic 
          
 
    def DataToPkl(self):
        print("Writing data.....")
        with open('Stock_crawler.pickle' ,'wb') as pkl :
            pickle.dump(self.BuyAmountList ,pkl)
        print("Writing finishing!")
            
    def PklToData(self):
        print("loading data.....")
        with open('Stock_crawler.pickle' ,'rb') as pkl :
            self.BuyAmountList.update(pickle.load(pkl))
        print("loading data sucess!")

   
A = Stock_Crawler()
for keys in A.BuyAmountList.keys():
    print(keys)
for data in range(13,18,1):
    A.BuyerAmount_Crawler('20/07/{}'.format(data))
