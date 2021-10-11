# Stock-Market-Trend-Analysis-Using-HMM-LSTM

Update: There is new version of this project, see more details on https://github.com/Yikiwi13/HMM-GMM-Timing-Strategy.git

## Introduction

The hidden Markov model (HMM) is a signal prediction model which has been used to predict economic regimes and stock prices. This project intends to achieve the goal of applying machine learning algrithms into stock market. The long short term memory model (LSTM) ensures that the previous information can continue to propagate backwards without disappearing as the hidden layer continuously superimposes the input sequence under the new time state.Our main purpose is to predict the ups and downs of one stock by using HMM-LSTM.<br> 
See details in our paper: [PAPER](https://arxiv.org/abs/2104.09700)

## Process
 
Using data from 2007-2018 in China's A share stock market, including daily price and trade volume and over 200 types of feature, we divided them into 8 types of features and make 8 HMMs. Then combined them together to predict ups and downs of stock price the next day. During which, we used GMM and XGBoost to fit the emission matrix B of continuous HMMs and used LSTM to find a better connection of X and Y. Moreover, an useful method of labeling called the reiple barrier method is well used to find relationship between hidden states and the trends of stock price.<br>
 
 ```
   #行情因子
   feature_col = ['closePrice', 'turnoverVol', 'highestPrice', 'lowestPrice']
   
   #质量类因子，描述资产负债，周转，运营，盈利，成本费用等指标
   type_zhiliang = ['AccountsPayablesTDays','AccountsPayablesTRate','AccountsPayablesTRate','ARTDays','ARTDays','ARTDays','BLEV',',BondsPayableToAsset','BondsPayableToAsset','CashRateOfSales','CashToCurrentLiability','CurrentAssetsRatio','CurrentRatio','DebtEquityRatio','DebtEquityRatio','DebtsAssetRatio','EBITToTOR','EquityFixedAssetRatio','EquityToAsset','EquityTRate','FinancialExpenseRate','FixAssetRatio','FixedAssetsTRate','GrossIncomeRatio','IntangibleAssetRatio','InventoryTDays','InventoryTRate','LongDebtToAsset','LongDebtToWorkingCapital','LongTermDebtToAsset','MLEV','NetProfitRatio','NOCFToOperatingNI','NonCurrentAssetsRatio','NPToTOR','OperatingExpenseRate','OperatingProfitRatio','OperatingProfitToTOR','OperCashInToCurrentLiability','QuickRatio','ROA','ROA5','ROE','ROE5','SalesCostRatio','SaleServiceCashToOR','TaxRatio','TotalAssetsTRate','TotalProfitCostRatio','CFO2EV','ACCA','DEGM']
    
   # 描述收益与风险
   type_shouyifengxian = ['CMRA','DDNBT','DDNCR','DDNSR','DVRAT','HBETA','HSIGMA','TOBT','Skewness','BackwardADJ']
    
   # 描述市值市盈市净
   type_jiazhi = ['CTOP','CTP5','ETOP','ETP5','LCAP','LFLO','PB','PCF','PE','PS','FY12P','SFY12P','TA2EV','ASSI']
    
   #情绪类，描述心理，换手率，动态买卖，成交量，人气，意愿，大盘趋势
   type_qingxu = ['DAVOL10','DAVOL20','DAVOL5','MAWVAD','PSY','RSI','VOL10','VOL120','VOL20','VOL240','VOL5','VOL60','WVAD','ADTM','ATR14','QTR6','SBM','STM','OBV','OBV6','TVMA20','TVMA6','TVSTD20','TVSTD6','VDEA','VDIFF','VEMA10','WEMA12','VEMA26','VEMA5','VMACD','VOSC','VR','VROC12','VROC6','VSTD10','VSTD20','ACD6','ACD20','AR','BR','ARBR','NVI','PVI','JDQS20','KlingerOscillator','MoneyFlow20','Volatility']
    
   #技术指标类，平均移动线，计算周期，动态移动，差异
   type_zhibiao = ['MassIndex','SwingIndex','minusDI','plusDI','ChaikinVolatility','ChaikinOscillator','DownRVI','BollUp','BollDown','DHILO','EMA10','EMA120','EMA20','EMA5','EMA60','EA10','EA120','EA20','EA5','EA60','MFI','ILLIQUIDITY','MACD','KDJ_K','KDJ_D','KDJ_J','UpRVI','RVI','DBCD','ASI','EMV12','EMV6','ADX','ADXR','MTM','MTMMA','UOS','EMA12','EMA26','BBI','TEMA10','Ulcer10','Hurst','Ulcer5','TEMA5','CR20','Elder','DilutedEPS','EPS']
    
   #动量类因子，描述平均移动，圆滑曲线，收益，增长率，未来趋势预测
   type_dongliang = ['REVS10','REVS10','REVS5','RSTR12','RSTR24','DAREC','GREC','DAREV','GREV','DASREV','GSREV','EARNMOM','FiftyTwoWeekHigh','BIAS10','BIAS20','BIAS5','BIAS60','CCI10''CCI20','CCI5','CCI88','ROC6','ROC20','SRMI','ChandeSD','ChandeSU','CMO','ARC','AD','AD20','AD6','CoppockCurve','Aroon','AroonDown','AroonUp','DEA','DIFF','DDI','DIZ','DIF','PVT','PCT6','PVT12','TRIX5','TRIX10','MA10RegressCoeff12','MA10RegressCoeff6','PLRC6','PLRC12','APBMA','BBIC','MA10Close','BearPower','RC12','RC24']
    
   #增长类，计算增长率
   type_zengzhang = ['EGRO','FinancingCashGrowRate','InvestCashGrowRate','NetAssetGrowRate','NetProfitGrowRate','NPParentCompanyGrowRate','OperatingProfitGrowRate','OperatingRevenueGrowRate','OperCashGrowRate','SUE','TotalAssetGrowRate','TotalProfitGrowRate','REC','FEARNG','FSALESG','SUOI']
 ```
 
### Experiment with 4 different models: <br>
 
 * #### GMM-HMM <br>
 * #### XGB-HMM <br>
 * #### GMM-HMM-LSTM <br>
 * #### XGB-HMM-LSTM <br>
 
 ### Compared with the results: <br>

* #### train_set

![](https://github.com/JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM/raw/master/FIGURE/train1.jpg)  

* #### test_set

![](https://github.com/JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM/raw/master/FIGURE/test1.jpg)  

* #### iteration_process

![](https://github.com/JINGEWU/Stock-Market-Trend-Analysis-Using-HMM-LSTM/raw/master/FIGURE/best_iter.png)  

* #### Accuracy
   GMM-HMM-LSTM performs 76.1612738% <br>
   XGB-HMM-LSTM performs 80.6991611% <br>

## Contribution

### Contributors

* #### Junbang Huo
* #### Yulin Wu
* #### Jinge Wu

### Institutions

* #### AI&FintechLab of Likelihood Technology
* #### Sun Yat-sen University
* #### Xi'an Jiaotong-Liverpool University

## Acknowledgement

We would like to say thanks to Maxwell Liu from ShingingMidas Private Fund, Jiahui Wu and Xingyu Fu from Sun Yat-sen University for their generous guidance throughout the project

## Set up

### Python Version

* #### 3.6

### Modules needed

* #### numpy
* #### pandas
* #### hmmlearn
* #### xgboost
* #### ...

## Contact
Please contact Mingwen Liu(刘铭文) liumwen@shiningmidas.com for more information about data.
* liumwen@shiningmidas.com
* huojb3@mail2.sysu.edu.cn
* wuylin6@mail2.sysu.edu.cn
* jinge.wu@outlook.com
