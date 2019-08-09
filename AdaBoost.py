from  sklearn.ensemble import AdaBoostRegressor
from  sklearn.model_selection import  train_test_split
from  sklearn.metrics import  mean_squared_error
from  sklearn.datasets import load_boston
import  numpy as np
import  pandas as pd
#加载数据
data=load_boston()
X_full = data.data
Y = data.target
boston = pd.DataFrame(X_full)
boston.columns = data.feature_names
boston['PRICE'] = Y
print(boston.head())
#分隔数据
train_x,test_x,train_y,test_y=train_test_split(data.data,data.target,test_size=0.25,random_state=33)
#使用AdaBoost回归模型
regressor=AdaBoostRegressor()
regressor.fit(train_x,train_y)
pred_y=regressor.predict(test_x)
mse=mean_squared_error(test_y,pred_y)
print('房价预测结果',pred_y)
print('均方误差=',round(mse,2))

#adaboost回归预测
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import  zero_one_loss
from  sklearn.tree import  DecisionTreeClassifier
from  sklearn.ensemble import  AdaBoostClassifier
#设置ADABOOST迭代次数
n_estimatord=200
#使用
X,y=datasets.make_hastie_10_2(n_samples=12000,random_state=1)
#从12000个数据中取前2000行作为测试集，其余作为训练集
train_x,train_y=X[2000:],y[2000:]
test_x,test_y=X[:2000],y[:2000]
#弱分类器
dt_stump=DecisionTreeClassifier(max_depth=1,min_samples_leaf=1)
dt_stump.fit(train_x,train_y)
dt_stump_err=1.0-dt_stump.score(test_x,test_y)
#决策树分类器
dt=DecisionTreeClassifier()
dt.fit(train_x,train_y)
dt_err=1.0-dt.score(test_x,test_y)
#AdaBoost分类器
ada=AdaBoostClassifier(base_estimator=dt_stump,n_estimators=n_estimatord)
ada.fit(train_x,train_y)
#仨个分类器的错误率可视化
fig=plt.figure()
#设置普拉提正确显示中文
plt.rcParams['font.sans-serif']=['SimHei']
ax=fig.add_subplot(111)
ax.plot([1,n_estimatord],[dt_stump_err]*2,'k-',label=u'决策树分类器 错误率')
ax.plot([1,n_estimatord],[dt_err]*2,'k--',label=u'决策树模型 错误率')
ada_err=np.zeros((n_estimatord,))
#遍历每次迭代的结果i为迭代次数，pred—y为结果
for i,pred_y in enumerate(ada.staged_predict(test_x)):
    #统计错误率
    ada_err[i]=zero_one_loss(pred_y,test_y)
#绘制每次迭代的Adaboost错误率
ax.plot(np.arange(n_estimatord)+1,ada_err,label='AdaBoost  Test错误率',color='orange')
ax.set_xlabel('迭代次数')
ax.set_ylabel('错误率')
leg=ax.legend(loc='upper right',fancybox=True)
plt.show()



from  sklearn.tree import  DecisionTreeRegressor
# 使用决策树回归模型
dec_regressor=DecisionTreeRegressor()
dec_regressor.fit(train_x,train_y)
pred_y = dec_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print(" 决策树均方误差 = ",round(mse,2))
from  sklearn.neighbors import  KNeighborsRegressor
# 使用 KNN 回归模型
knn_regressor=KNeighborsRegressor()
knn_regressor.fit(train_x,train_y)
pred_y = knn_regressor.predict(test_x)
mse = mean_squared_error(test_y, pred_y)
print("KNN 均方误差 = ",round(mse,2))
