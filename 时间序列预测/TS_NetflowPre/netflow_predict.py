# -*- coding: utf-8 -*-
'''
Created on 2017年2月19日
@author: Lu.yipiao
'''

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from get_accuracy import get_average_relative_error, get_average_absolute_error, cos_similarity

#——————————————————导入数据——————————————————————
#f='./dataset/dataset_1.csv'
#df=pd.read_csv(f)     #读入数据
#data=np.array(df['value'])   #获取序列
#data=data[::-1]      #反转，使数据按照日期先后顺序排列

data = [17820652, 18050364, 17911606, 16831316, 8759626, 8546053, 17573559, 17957414, 8579513, 8947416, 8896293, 8513787, 9689336, 7644881, 8187614, 8157440, 14690066, 13122113, 8189606, 8128143, 9928045, 12186196, 11087630, 8487980, 11550187, 15780109, 14416526, 12554730, 9677747, 11517624, 18042087, 17725251, 16918745, 13981266, 9067449, 7621155, 8233936, 6407346, 5129027, 5072525, 5379080, 2749548, 3386364, 2606616, 5753418, 11276798, 6900994, 3827738, 6735140, 10346079, 10970506, 9039637, 6130244, 7863363, 17851643, 16621718, 15299360, 6380432, 5777692, 5496734, 5051249, 4673845, 3902700, 2850843, 5616231, 2150059, 1836288, 2328483, 4835081, 7806310, 6582895, 3822465, 7020531, 9781196, 10954961, 5855414, 6374146, 7007517, 17624561, 11252861, 15389236, 7100923, 6288682, 5557393, 5507696, 4839543, 3215167, 2107288, 5284018, 1677593, 1975131, 2127995, 4371311, 7579893, 6320845, 3379414, 6971598, 10663938, 11656811, 9904060, 6615442, 7785164, 17951288, 18241499, 17124873, 7860011, 5068894, 6072006, 7082042, 5178642, 3504864, 3124104, 5363476, 2672646, 3056433, 2628075, 5603036, 8108877, 7231927, 4471953, 7084985, 12227811, 13953487, 9176698, 7023547, 13697545, 18330191, 18341313, 18270395, 9819666, 6596462, 7404952, 7209089, 4442841, 2527758, 3934994, 7647949, 3578724, 2257577, 3076758, 5145234, 8768478, 7397283, 4147334, 5192010, 5990813, 6407392, 5855151, 6151765, 5421948, 17275831, 13503030, 9697451, 7223771, 6569367, 4713950, 6340608, 3813842, 3705733, 3428458, 6596959, 2956116, 1902686, 2577152, 5127546, 7551840, 7490062, 4156895]

#data = np.log(data)
#data = np.tan(data)

#以折线图展示data
plt.figure()
plt.plot(data-500000)
plt.show()

'''
normalize_data=(data-np.mean(data))/np.std(data)  #标准化
normalize_data = normalize_data[0:144]
normalize_data=normalize_data[:,np.newaxis]       #增加维度

#生成训练集
#设置常量
time_step=24      #时间步
rnn_unit=10       #hidden layer units
batch_size=20     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0006         #学习率
train_x,train_y=[],[]   #训练集
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist()) 



#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #每批次tensor对应的标签
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }


#——————————————————定义神经网络变量——————————————————
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


#——————————————————训练模型——————————————————
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #重复训练10000次
        for i in range(1000):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #É每10步保存一次参数
                if step%10==0:
                    print(i,step,loss_)
                    print("保存模型：",saver.save(sess,'./model/stock_model.ckpt'))
                step+=1


#————————————————预测模型————————————————————
def prediction():
    pred,_=lstm(1)      #预测时只输入[1,time_step,input_size]的测试数据
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
#        module_file = tf.train.latest_checkpoint(base_path+'module/')
        module_file='./model/stock_model.ckpt'
        saver.restore(sess, module_file) 

        #取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq=train_x[-1]
        predict=[]
        #得到之后100个预测结果
        for i in range(24):
            next_seq=sess.run(pred,feed_dict={X:[prev_seq]})
            predict.append(next_seq[-1])
            #每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
            
        
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.show()
        
        #评估模型                  
        predict = [x[0]*np.std(data)+np.mean(data) for x in predict]
        print('Predict Results:',predict)
        test_data = data[144:168]
        print('average_relative_error: ', get_average_relative_error(test_data, predict))
        print('average_absolute_error: ', get_average_absolute_error(test_data, predict))
        print('cos_similarity: ', cos_similarity(test_data, predict))
        
#————————————————Main————————————————————    
with tf.variable_scope('train'):  
    train_lstm()
print('————————————————train end!————————————————————')
    
with tf.variable_scope('train',reuse=True):  
    prediction()  
print('————————————————prediction end!———————————————')

'''