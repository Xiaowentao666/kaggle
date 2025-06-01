#%%基本库
import pandas as pd
import numpy as np

#读取文件
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_Id = train['PassengerId']
test_Id = test['PassengerId']
train.drop(columns='PassengerId', inplace=True)
test.drop(columns='PassengerId', inplace=True)

print('删除ID列后的训练集大小：', train.shape)
print('删除ID列后的测试集大小：', test.shape)
# 保存标签列，并删除，方便数据清洗
Survived = train['Survived']
train = train.drop(columns='Survived')
# 合并训练集和测试集
all_data = pd.concat([train, test], axis=0, ignore_index=True)
print(max(all_data['Fare']))
all_data.info()

#%%# 创建excel表格，用来描述特征
# cols = train.columns
# col_type = train.dtypes
# excel = pd.DataFrame({'特征':cols, '数据类型':col_type})
# excel.to_excel('特征描述.xlsx', index=False)

#%%数据清洗
#查看缺失值
print(all_data.isnull().sum())
#%%处理缺失值
def missing_process(df):
    #删除缺少量大的数据
    df.drop(columns=['Cabin'],inplace=True)
    #Age按sibsp分组的众数填充
    def fill_with_mode(x):
        mode_values = x.mode()
        if not mode_values.empty:
            return x.fillna(mode_values[0])
        else:
            # 如果众数为空，使用全局的中位数填充
            return x.fillna(df['Age'].median())
    df['Age'] = df.groupby('SibSp')['Age'].transform(fill_with_mode)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    #用众数填充
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df
all_data = missing_process(all_data)
print(all_data.isnull().sum())



# %%特征提取
#性别
all_data['Sex'] = all_data['Sex'].map({'male':1,'female':0})
#%%姓名
"""
将Name中的称呼归为以下几类Title:
Rare 政府官员/王室
Mr 已婚男士
Mrs 已婚女士
Miss 未婚女士
Master 有技能的人/教师
"""

# 从name列中提取出title
# 提取Title并删除Name列
def get_title(name):
    return name.split(',')[1].split('.')[0].strip()

title_map = {
    "Capt": "Rare", "Col": "Rare", "Major": "Rare",
    "Jonkheer": "Rare", "Don": "Rare", "Sir": "Rare",
    "Dr": "Rare", "Rev": "Rare", "the Countess": "Rare",
    "Mme": "Mrs", "Mlle": "Miss", "Ms": "Miss", "Mr": "Mr",
    "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Lady": "Rare"
}
all_data['Title'] = all_data['Name'].apply(get_title).map(title_map)
element_counts = all_data['Title'].value_counts()

print("各元素的数量统计：")
print(element_counts)
#%%添加家庭规模特征
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
#是否单独出行
all_data['IsAlone'] = (all_data['FamilySize']==1).astype(int)

#%%票价年龄分箱
# 使用训练集数据定义分箱边界

fare_bins = [-1,10,20,30,50,75,100,150,200,250,300,np.inf]
age_bins = [0, 12, 18, 30, 50, 100]

# 应用分箱到全体数据
all_data['FareBin'] = pd.cut(all_data['Fare'], bins=fare_bins, labels=False)
all_data['AgeGroup'] = pd.cut(all_data['Age'], bins=age_bins, labels=False)
print("分箱后的缺失值统计：")
print(all_data[['FareBin', 'AgeGroup']].isnull().sum())
# %%处理数值类型
#把Pclass变为字符类型
all_data['Pclass'] = all_data['Pclass'].astype(str)

#%%进行独热编码
# all_data.drop(columns=['Name', 'SibSp', 'Parch', 'Ticket'], inplace=True)
all_data = pd.get_dummies(all_data)
# 检查缺失值
print("处理后的缺失值统计：")
print(all_data.isnull().sum())

#%%构建模型
#分离数据
clean_train = all_data.loc[:len(train)-1]
clean_test = all_data.loc[len(train):]


#%%库
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#%%
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(clean_train, Survived, test_size=0.2, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression(max_iter=1000)

# 训练模型
model.fit(X_train, y_train)

# 在验证集上进行预测
y_pred_val = model.predict(X_val)

# 计算验证集上的准确率
accuracy = accuracy_score(y_val, y_pred_val)
print(f"验证集准确率: {accuracy:.2f}")

# 在测试集上进行预测
y_pred = model.predict(clean_test)


#%%保存结果
result = pd.read_csv('gender_submission.csv')
result = pd.DataFrame({'PassengerId': test_Id, 'Survived': y_pred})
result.to_csv('titanic_submission.csv', index=False)
print('结果已保存')
