# -*- coding: utf-8 -*-

# 買賣超總資料

import requests
from datetime import date, timedelta, datetime
from bs4 import BeautifulSoup
import time
import numpy as np 
import pickle
import json  
 
 
# Following: 接下來直接將以下抓取函示寫成一包 Class:
# Dict 內新增股票欄位function & 抓取券商function獨立出來 
# => (初始化Dict = {} 放 __init__ 內)  
# => (新增股票欄位func獨立出來 class.new('Stocknumber'))     
# => (抓取券商func獨立出來 class.grab('Stocknumber,start,end,) , 另判斷 if 股票in Dict or not? ) 
 
#################################################################################################################################
#     Data Structure of BuyAmountList : 
#                {'日期1' : {
#                        分點1 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        分點2 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        分點3 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                          .
#                          .
#                          .
#                    }
#
#                '日期2' : {
#                        分點1 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        分點2 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        分點3 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                          .
#                          .
#                          .
#                    }
#
#     Data Structure of Stock_Price :
#                 # 這邊只抓有被買賣的股票?? (因為沒被買賣,成本就沒意義,收盤價也就不重要了)
#                {'日期1' : {
#                        開盤價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        收盤價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        最高價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        最低價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }  
#                    }
#
#                {'日期2' : {
#                        開盤價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        收盤價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        最高價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }
#                        最低價 : {股票名1 : 張數1 , 股票名2 : 張數2 . . . . . }  
#                    }
#
###################################################################################################################################
 
class Stock_Crawler(object):
 
    def __init__( self ):
        
                
        # URl and cookies 
        self.stocklist_url = 'https://stock.wespai.com/pick/choice'
        self.data_header = [
            {'qry[]': 'dv', 'id[]': 'dv', 'val[]': '0;12000'},
            {'cookie': '__utmz=140509424.1585664337.12.6.utmcsr=google|utmccn=(organic)|utmcmd=organic|utmctr=(not%20provided); __utma=140509424.970758813.1562249232.1585762091.1586178222.15; __utmc=140509424; __utmt=1; __utmb=140509424.4.10.1586178222',
             'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36'}
            ]   
        self.buyerlist_url = 'http://5850web.moneydj.com/z/js/zbrokerjs.djjs'
        self.crawler_url = 'http://5850web.moneydj.com/z/zg/zgb/zgb0.djhtm?a={l}&b={m}&c=E&e={i}-{j}-{k}&f={i}-{j}-{k}'
        self.K_line_url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date={i}{j}{k}&stockNo={l}&_=1598347181135"
        
        #Google drive path
        self.BuyAmountList_path = "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock Data/Stock_BuyAmountList_{}.pickle"
        self.holidays_path = "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock_Information/holidays"
        test_case_path = "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock_Information/Buyer_dic"
        buyer_index_path = "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock_Information/Buyer_index"
        stock_index_path = "/content/drive/My Drive/Colab Notebooks/Stock_Project/Stock_Information/Stock_List"

        #Load parameter
        self.test_case = self.PklToData(test_case_path)  # 所有券商 { (6802,6832):"凱基-虎尾" ,....} 
        self.buyer_index = self.PklToData(buyer_index_path) # 券商 index 對照表 { "凱基-虎尾" : 0 , "凱基-後甲" : 1 ,....} 
        self.stock_index = self.PklToData(stock_index_path) # Stock index 對照表 => {"1101":0 , "1102": .....}
        self.holidays = self.PklToData(self.holidays_path) # Stock index 對照表 => {"1101":0 , "1102": .....}
              
    def loading_data(self, year_month_name):

        try:
            print("loading data...")
            self.BuyAmountList = self.PklToData(self.BuyAmountList_path.format(year_month_name)) # Loading the pickel data to self.BuyAmountList first
            print("loading data success !!!")
                    
        except FileNotFoundError as e:
            self.BuyAmountList = { } 
            print("this is the first time to create the dict!")

    def saving_data(self, year_month_name):

        self.DataToPkl( self.BuyAmountList_path.format(year_month_name) , self.BuyAmountList ) # Save to pkl while finishing grabbing data
        
    def Iter_between_interval(self , start_day , end_day , function = None ):
        
        # day format :　"20/01/30" 
        # datetime format : "2020-01-30 00:00:00"
 
        # loading data 
        year_month_name = "-".join([start_day.split("/")[0],start_day.split("/")[1]])
        self.loading_data(year_month_name)  # loading data in year_month
        # transfer the time to datetime format
        start_datetime = datetime.strptime(start_day ,"%y/%m/%d" ) 
        end_datetime = datetime.strptime(end_day ,"%y/%m/%d" ) 
        
        # main iteration process
        crawl_datetime = start_datetime 
        for i in range((end_datetime - start_datetime).days + 1):
          if crawl_datetime.isoweekday() not in [6,7]:

            day_array = [str(crawl_datetime).split('-')[0][2:] , str(crawl_datetime).split('-')[1] , str(crawl_datetime).split('-')[2][:2]] # ["20" , "01" , "30"]
            day = '/'.join(day_array)
            
            # judge the month transfer , if month increases , loading or creating new month
            if year_month_name != "-".join([day_array[0] , day_array[1]]):
              year_month_name = "-".join([day_array[0] , day_array[1]])
              self.loading_data(year_month_name) 

            if str(crawl_datetime) in self.BuyAmountList and len(self.BuyAmountList[ str(crawl_datetime) ]) == len(self.buyer_index): 
              crawl_datetime += timedelta(days=1)
              print(f"{day} has been crawled !")
              continue

            if str(crawl_datetime) in self.holidays:
              crawl_datetime += timedelta(days=1)
              print(f"{day} was holiday !")
              continue

            self.BuyAmountCrawler(day) # do something in this interval

            self.saving_data(year_month_name) # saving data in year_month

          crawl_datetime += timedelta(days=1)
          

            
    def BuyAmountCrawler(self , Date ): #這可以是classmethod or staticmethod
        
        start_time = time.time()
        count = 1  
 
        CrawlDay = str(datetime.strptime(Date, "%y/%m/%d")) # 要抓取的起始點
        dayarray = CrawlDay.split('-')
        self.Generator_Specified_Date(CrawlDay) # generate new key of date in Buyamountlist
 
        for point , sub_point in self.test_case.keys():
             
            Stock_raw = self.BuyAmount_SubProcess( dayarray , point , sub_point ) 
            self.BuyAmountList[CrawlDay].append(Stock_raw) # Update Stock_raw 進 BuyAmountList 裡
            print("Now crawling date:{}, ".format(CrawlDay),"Finish [{a}/{b}] = {c}%".format(a=count,b=len(self.test_case),c=count/len(self.test_case)*100),", Waste {} sec".format(time.time()-start_time),"and now in:",self.test_case[(point,sub_point)])
            count+=1
            # 分點資料找不到券商,開始往各股買賣超前十五名找(補償分點買賣超的不足, EX:摩根史丹利大量買超只列到前五十,股本比較小的股票就不會被顯示在上面)
        
        self.delete_holidays(CrawlDay) # check whether it's no trading date , if so delete this date  
        print('BuyAmount Finish!!!')
 
    def BuyAmount_SubProcess(self, dayarray , point , sub_point ):
 
        Stock_raw = [0]*len(self.stock_index) # this is a list likes => [0,0,0,0,.....,0,0,0]
 
        url = self.crawler_url.format(i=dayarray[0], j=dayarray[1], k=dayarray[2][:2], l=point, m=sub_point) # 分點買賣超的網頁爬蟲
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        lists = [str(x) for x in soup.findAll('td', class_=['t4t1'])]  # 特定券商&買賣超統計  
        StockListDummy = self.FindStockNumber(lists) #找到目前爬取券商前五十名的買賣超股票清單
 
        for idx , stock in enumerate(StockListDummy):
 
            #這邊利用 KLineCrawler 對照 StockListDummy 來爬前五十名的收盤價!                              
            Target_amount = self.StrToDigit(soup.select('.t3n1')[idx * 3 + 2].string) # 總買賣超格數為3的倍數格         
            if stock in self.stock_index:
              Stock_raw[ self.stock_index[stock] ] = Target_amount  # 將前五十名的買賣超股票清單買賣超data , update 進 Stock_raw 裡
            
        return Stock_raw # this is a list likes => [0,0,12,366,0,0,.....,0,25,0,0,259]
 
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
    
        Date = str(Date)
        self.BuyAmountList.update({ Date: [] } )
        
                
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
 
   # 產生所有Stock List(之後會把它存到SQL)
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
            Stock_List.update({ ele : 0 }) #set default buyamount as 0
 
        Stock_List['1101'] = Stock_List['["1101']
        del Stock_List['["1101']  # 暴力修亂碼法XDD
 
        return Stock_List   

    def delete_holidays( self , date ):
  
        mat = np.array(self.BuyAmountList[date])
        value = np.sum(np.sum(mat,axis = 0),axis = 0)
        if value == 0:
          self.holidays.append(date)
          self.DataToPkl(self.holidays_path,self.holidays)
          del self.BuyAmountList[date]  
 
    def DataToPkl(self , path , data ):
  
        with open( path ,'wb') as pkl :
            pickle.dump( data , pkl )       
            
    def PklToData(self , path ):
        
        with open( path ,'rb') as pkl :
            data = pickle.load( pkl )
 
        return data