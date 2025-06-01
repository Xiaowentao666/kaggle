#%%库
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', palette='muted')

#%%读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
#保存Id列，并删除
train_Id = train_data['Id']
test_Id = test_data['Id']
train_data.drop(columns='Id')
test_data.drop(columns='Id')
#合并数据集
saleprice = train_data['SalePrice']
all_data = pd.concat([train_data,test_data],axis=0, ignore_index=True)
all_data.drop(columns='SalePrice', inplace=True)
#%%查看数据
all_data.info()
#%%相关性
plt.figure(figsize=(15, 12))
train_corr = train_data.select_dtypes(include=[np.number]).corr()
sns.heatmap(train_corr,square=True, vmax=0.8, cmap='RdBu')
#%%
miss_val_cols = (all_data.isnull().sum())
print(miss_val_cols[miss_val_cols > 0])

#%%处理缺失值
def process_missing(df):
    #Functional中Typ相当于空值
    df['Functional'] = df['Functional'].fillna('Typ')
    #用众数替代
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['Utilities'] = df['Utilities'].fillna('AllPub')
    df['Electrical'] = df['Electrical'].fillna('SBrkr')
    df['KitchenQual'] = df['KitchenQual'].fillna('TA')
    df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
    df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])
    df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])
    # MSZoning和MSSubClass相关，用分组后的众数替代
    df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x:x.fillna(x.mode()[0]))
    
    # 删除缺失值占比80%以上的特征
    df.drop(columns=['PoolQC','MiscFeature','Alley','Fence'],inplace=True)
    # 用0替代车库类的数值型特征
    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
        df[col] = df[col].fillna(0)
    # 用None替代车库类的文本型特征
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        df[col] =df[col].fillna('None')
    # 用None替代地下室类的文本型特征 
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        df[col] = df[col].fillna('None')
    
    # LotFrontage与Neighborhood相关，用分组后的中位数替代
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
    
    # 剩余文本型特征的空值没有明确的意义，用None替代
    object = []
    for i in df.columns:
        if df[i].dtype == object:
            object.append(i)
    df.update(df[object].fillna('None'))

    #同样，对数值型特征，用0替代
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numeric.append(i)
    df.update(df[numeric].fillna(0))

    return df

all_data = process_missing(all_data)

#%% 确认是否还有缺失值
miss_val_cols = (all_data.isnull().sum())
print(miss_val_cols[miss_val_cols > 0])
all_data.isnull().sum().value_counts()

# %%异常值处理
#选取OverallQual,TotalBsmtSF,GrLivArea,YearBuilt这四个特征来查看相关性
fig = plt.figure(figsize=(15,12))
cols = ['OverallQual','TotalBsmtSF','GrLivArea','YearBuilt']
for col in cols:
    ax = fig.add_subplot(3,2,cols.index(col)+1)
    ax.scatter(train_data[col],train_data['SalePrice'])
    ax.set_xlabel(col)
plt.show()
#%%删除离群点
outlier1 = train_data[(train_data['OverallQual']==4) & (train_data['SalePrice'] > 200000)].index.tolist()
outlier2 = train_data[(train_data['TotalBsmtSF'] > 6000) & (train_data['SalePrice'] < 300000)].index.tolist()
outlier3 = train_data[(train_data['GrLivArea'] > 4500) & (train_data['SalePrice'] < 300000)].index.tolist()
outlier4 = train_data[(train_data['YearBuilt'] < 1900) & (train_data['SalePrice'] > 400000)].index.tolist()
outliers = outlier1 +outlier2 + outlier3 +outlier4
#去重
outliers = list(set(outliers))
print(f'离群点个数为：{len(outliers)},其索引为{outliers}')

# 特征矩阵以及标签都需要删除
all_data.drop(index=outliers, inplace=True)
saleprice.drop(index=outliers, inplace=True)

# 重置索引
all_data.reset_index(drop=True, inplace=True)
saleprice.reset_index(drop=True, inplace=True)


#%%EDA（探索性数据分析）
"""
EDA的主要目的有以下几点：
①标签（房价）的分布是怎样的？若不符合正态分布，需要通过数学变换来改善。
②重要特征与房价的相关性？相关系数是多少，能否创建新的特征？
③特征与特征之间的相关性？是否有一些特征，它们之间息息相关，从而产生多重共线性？这些特征要如何处理？
"""
plt.figure(figsize=(10,5))
sns.distplot(train_data['SalePrice'])
plt.xticks(rotation=30)

#%%由图可得，房价偏离正态分布，并且有明显的正偏度，后续需要通过对数变换处理
#重要特征相关性
plt.figure(figsize=(10,10))
top_corr = train_corr['SalePrice'].nlargest(10).index
train_top_corr = train_data.loc[:,top_corr].select_dtypes(include=[np.number]).corr()
sns.heatmap(train_top_corr,annot=True,square=True,fmt='.2f',cmap='RdBu',vmax=0.8)
"""
部分特征之间存在很强的相关性,存在多重共线性
GarageCars和GarageArea
TotalBsmtSF和1stFlrSF
TotRmsAbvGrd和GrLivArea
"""
# %%绘制网格图，查看相关性
cols = ['SalePrice','OverallQual','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(train_data[cols], size=2.5)

#%%特征工程
#创造交互特征
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['YrSinceRemod'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd'].astype(int)
all_data['YrSinceBuilt'] = all_data['YrSold'].astype(int) - all_data['YearBuilt'].astype(int)
all_data['OverallEval'] = all_data['OverallQual'] + all_data['OverallCond']
all_data['LowQualPct'] = all_data['LowQualFinSF'] / all_data['TotalSF']
all_data['funcSF'] = all_data['WoodDeckSF'] + all_data['OpenPorchSF'] + all_data['EnclosedPorch'] + all_data['3SsnPorch'] + all_data['ScreenPorch'] + all_data['PoolArea']

#%%有序变量编码
qual_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
order_cols = ['BsmtCond','BsmtQual','ExterCond','ExterQual','FireplaceQu',
              'GarageCond','GarageQual','HeatingQC','KitchenQual']

# 为了防止被get_dummies独热编码，数据类型改为int
for order_col in order_cols:
    all_data[order_col] = all_data[order_col].map(qual_mapping).astype(int)

# 一些数值型特征实际属于类别变量，修改其数据类型为文本型
all_data['MSSubClass'] = all_data['MSSubClass'].astype(object)
all_data['YrSold'] = all_data['YrSold'].astype(object)
all_data['MoSold'] = all_data['MoSold'].astype(object)

#时间序列标签编码
from sklearn.preprocessing import LabelEncoder
time_cols = ['GarageYrBlt','YearBuilt','YearRemodAdd','YrSold']
for time_col in time_cols:
    all_data[time_col] = LabelEncoder().fit_transform(all_data[time_col])

#%%数值型特征对数变换
# 数值型特征的数据变换：改变数据分布
all_data.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
#%%计算峰度
from scipy.stats import skew
numeric_df = all_data.select_dtypes(['float64','int32','int64'])
numeric_cols = numeric_df.columns.tolist()
skewed_cols = all_data[numeric_cols].apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_df = pd.DataFrame({'skew':skewed_cols})
skewed_df

#%%对偏度绝对值大于1的特征进行对数变换
skew_cols = skewed_df[skewed_df['skew'].abs()>1].index.tolist()
for col in skew_cols:
    all_data[col] = np.log1p(all_data[col])
#%%处理SalePrice
from scipy.stats import norm
from scipy import stats
sns.distplot(saleprice, fit=norm)
fig = plt.figure()
res = stats.probplot(saleprice, plot=plt)

#%%对数变换
saleprice = np.log1p(saleprice)
sns.distplot(saleprice, fit=norm)
fig = plt.figure()
res = stats.probplot(saleprice, plot=plt)

#%%进行独热编码
all_data = pd.get_dummies(all_data)
all_data.info()

#%%还原训练集和测试集
clean_train = all_data.iloc[:1456, :]
clean_test = all_data.iloc[1456:, :]

# 加上去除离群点后的标签列
clean_train = pd.concat([clean_train, saleprice], axis=1)

miss_val_cols = (clean_train.isnull().sum())
print(miss_val_cols[miss_val_cols > 0])
all_data.isnull().sum().value_counts()
# clean_train.select_dtypes(include=['object']).columns
# print('处理后的训练集大小：', clean_train.shape)
# print('处理后的测试集大小：', clean_test.shape)

#%%建模
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

#%%划分数据集
X = clean_train.drop(columns='SalePrice')
y = clean_train['SalePrice']

#%% 定义交叉验证模式

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=10)
kf = KFold(n_splits=10, random_state=50, shuffle=True)

# 定义衡量指标函数
def rmse(y,y_pred):
    rmse = np.sqrt(mean_squared_error(y,y_pred))
    return rmse

def cv_rmse(model,X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf))
    return rmse

#%% 建立基线模型
lbg = LGBMRegressor(objective = 'regression', random_state = 50)
xgb = XGBRegressor(objective = 'reg:linear',random_state = 50,shuffle=True)
ridge = make_pipeline(RobustScaler(), RidgeCV(cv=kf))
svr = make_pipeline(RobustScaler(),SVR())
gbr = GradientBoostingRegressor(random_state=50)
rf = RandomForestRegressor(random_state=50)

#%%# 基线模型评估
models = [lbg,xgb,ridge,svr,gbr,rf]
models_name = ['lbg','xgb','ridge','svr','gbr','rf']
scores = {}
for i,model in enumerate(models):
    score = cv_rmse(model)
    print(f'{models_name[i]}rmse score:{score.mean()}')
    scores[models_name[i]] = (score.mean(),score.std())

rmse_df = pd.DataFrame(scores,index=['rmse_score','rmse_std'])
rmse_df.sort_values('rmse_score',axis=1,inplace=True)
rmse_df


#%% 参数优化后的模型
lgb = LGBMRegressor(objeactive='regression'
                    ,n_estimators=1200
                    ,max_depth=8
                    ,num_leaves=10
                    ,min_data_in_leaf=3
                    ,max_bin=25
                    ,bagging_fraction=0.6
                    ,bagging_freq=11
                    ,feature_fraction=0.6
                    ,lambda_l1=0.004641588833612777
                    ,lambda_l2=4.641588833612782e-05
                    ,learning_rate=0.01
                    ,random_state=50
                    ,n_jobs=-1)

xgb = XGBRegressor(objective='reg:linear'
                   ,n_estimators=2550
                   ,learning_rate=0.02
                   ,max_depth=3
                   ,subsample=0.6
                   ,min_child_weight=3
                   ,colsample_bytree=0.5
                   ,random_state=50
                   ,n_jobs=-1
                   ,silent=True)

# ridge直接使用默认参数，效果反而更好
ridge = make_pipeline(RobustScaler(), RidgeCV(cv=kf))

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

gbr = GradientBoostingRegressor(n_estimators=6000
                                ,learning_rate=0.01
                                ,max_depth=4
                                ,max_features='sqrt'
                                ,min_samples_leaf=15
                                ,min_samples_split=10
                                ,loss='huber'
                                ,random_state=50)

rf = RandomForestRegressor(n_estimators=2000
                           ,max_depth=15
                           ,random_state=50
                           ,n_jobs=-1)

#%%同样的代码，评估参数优化后的模型
# 参数优化后的模型评估
models = [lgb, xgb, ridge, svr, gbr, rf]
model_names = ['lgb','xgb','ridge','svr','gbr','rf']
scores = {}

for i, model in enumerate(models):
    score = cv_rmse(model)
    print('{} rmse score: {:.4f}, rmse std: {:.4f}'.format(model_names[i], score.mean(), score.std()))
    scores[model_names[i]] = (score.mean(), score.std())
    
rmse_df = pd.DataFrame(scores, index=['rmse_score','rmse_std'])
rmse_df.sort_values('rmse_score', axis=1, inplace=True)
rmse_df

# %%模型融合
"""
1、stacking
① 第一层：选取6个模型，进行10折交叉验证。在每一折交叉验证中，将数据集（带有标签）划分为训练集和验证集。用训练集训练模型后，用
模型预测验证集的结果，并将结果保存到表格stacked_train；用模型预测测试集（未知标签）的结果，并将结果保存到表格stacked_test。
② 第一层结束后，stacked_train一共有6列，即每个模型交叉验证的结果，我们将这6列作为特征，然后在后面加上数据集的标签，这样不就
创建了一个“特征+标签”的训练集了嘛！
③ stacked_test一共有 6*10 列，即每个模型在10次交叉验证中，对测试集的预测结果。对于每个模型而言，我们取10列的均值，得到
6列，这不就是只有“特征”而不知道“标签”的测试集了嘛！④ 第二层，我们选用1个模型xgboost，用第②步获得的训练集训练，然后预测第③步的
测试集，得到stacking的预测结果

"""
# 定义StackingRegressor类
class StackingRegressor(object):
    
    def __init__(self, fir_models, fir_model_names, sec_model, cv):
        # 第一层的基模型
        self.fir_models = fir_models
        self.fir_model_names = fir_model_names
        # 第二层用来预测结果的模型
        self.sec_model = sec_model
        # 交叉验证模式，必须为k_fold对象
        self.cv = cv
    
    def fit_predict(self, X, y, test):    # X,y,test必须为DataFrame
        # 创建空DataFrame
        stacked_train = pd.DataFrame()
        stacked_test = pd.DataFrame()
        # 初始化折数
        n_fold = 0

        # 遍历每个模型，做交叉验证
        for i, model in enumerate(self.fir_models):
            # 初始化stacked_train
            stacked_train[self.fir_model_names[i]] = np.zeros(shape=(X.shape[0], ))

            #遍历每一折交叉验证
            for train_index, valid_index in self.cv.split(X):
                # 初始化stacked_test
                n_fold += 1
                stacked_test[self.fir_model_names[i] + str(n_fold)] = np.zeros(shape=(test.shape[0], ))

                # 划分数据集
                X_train, y_train = X.iloc[train_index, :], y.iloc[train_index]
                X_valid, y_valid = X.iloc[valid_index, :], y.iloc[valid_index]

                # 训练模型并预测结果
                model.fit(X_train, y_train)
                stacked_train.loc[valid_index, self.fir_model_names[i]] = model.predict(X_valid)
                stacked_test.loc[:, self.fir_model_names[i] + str(n_fold)] = model.predict(test)
            print('{} is done.'.format(self.fir_model_names[i]))

        # stacked_train加上真实值标签
        y.reset_index(drop=True, inplace=True)
        stacked_train['y_true'] = y

        # 计算stacked_test中每个模型预测结果的平均值
        for i, model_name in enumerate(self.fir_model_names):
            stacked_test[model_name] = stacked_test.iloc[:, :10].mean(axis=1)
            stacked_test.drop(stacked_test.iloc[:, :10], axis=1, inplace=True)
        
        # 打印stacked_train和stacked_test
        print('----stacked_train----\n', stacked_train)
        print('----stacked_test----\n', stacked_test)
        
        # 用sec_model预测结果
        self.sec_model.fit(stacked_train.drop(columns='y_true'), stacked_train['y_true'])
        y_pred = self.sec_model.predict(stacked_test)
        return y_pred
    
sr = StackingRegressor(models, model_names, xgb, kf)
stacking_pred = sr.fit_predict(Xtrain, ytrain, Xtest)

def rmse(y, y_pred):
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return rmse

stacking_score = rmse(ytest, stacking_pred)
print(stacking_score)

#%%模型均值融合
def blending(X, y, test):
    lgb.fit(X, y)
    lgb_pred = lgb.predict(test)

    xgb.fit(X, y)
    xgb_pred = xgb.predict(test)
    
    ridge.fit(X, y)
    ridge_pred = ridge.predict(test)
    
    svr.fit(X, y)
    svr_pred = svr.predict(test)
    
    gbr.fit(X, y)
    gbr_pred = gbr.predict(test)
    
    rf.fit(X, y)
    rf_pred = rf.predict(test)
    
    sr = StackingRegressor(models, model_names, xgb, kf)
    sr_pred = sr.fit_predict(X, y, test)
    
    # 加权求和
    blended_pred = (0.05 * lgb_pred +
                    0.1 * xgb_pred +
                    0.2 * ridge_pred +
                    0.25 * svr_pred +
                    0.15 * gbr_pred +
                    0.05 * rf_pred +
                    0.2 * sr_pred)
    return blended_pred

blended_pred = blending(Xtrain, ytrain, Xtest)
blending_score = rmse(ytest, blended_pred)
print(blending_score)
#%%导出结果
y_pred = np.exp(blending(X, y, clean_test)) - 1
#%%
sample = pd.read_csv('sample_submission.csv')

sample['SalePrice'] = y_pred
sample.to_csv('result.csv', index=False)

