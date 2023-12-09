import os
import sys
import time
import glob
import pdb
sys.path.append(r"/home/tarunpreet/NSM_PGI_IIT/")
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from time import process_time
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout,Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import cross_val_score
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
import joblib

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

if __name__=='__main__':
	

	t1_start=time.time()
	
	path="data0"
	all_files=glob.glob(path+"/*.csv") 
	all_files=glob.glob(os.path.join(path,"*.csv"))
	
	XList=pd.DataFrame()
	YList=pd.DataFrame()
	SD=np.array([])

	for filename in all_files:
		data=pd.read_csv(filename)
		
	
		# Define X dataset
		X_data=data.drop(labels=['Date','Time','IBPS','IBPD','ICU_Stay','Hospital_Stay','IBPM','Outcome'],axis=1)
		X_data=X_data.replace('---',np.nan)
		X_data=X_data.replace('N/R',np.nan)
		X_data1=X_data.dropna(axis=0)
		for column in X_data1.columns:
			
			D=X_data1[column]	
			DL=D[D>0]
			DL=DL.to_numpy().reshape(-1,1)
			scaler=MinMaxScaler()
			scaler.fit(DL)
			ScaledData= scaler.transform(DL)
			joblib.dump(scaler, 'scalerTS.pkl')
			SD = ScaledData.flatten()	
		s=len(SD)-25
		for x in range (0,s,5):
			X=SD[x:20+x]
			Y=SD[x+20:x+25]
			
			XL=pd.DataFrame(X).transpose()
			XList=pd.concat([XList,XL],axis=0)
			YL=pd.DataFrame(Y).transpose()
			YList=pd.concat([YList,YL],axis=0)
	
	X_train,X_test,y_train,y_test=train_test_split(XList.to_numpy(), YList.to_numpy(),test_size=0.20,random_state=42)
	
	
	#ANN
	ann = Sequential()
	ann.add(Dense(20, activation = 'relu', kernel_initializer='normal',input_shape=(20,)))
	ann.add(Dense(20, activation = 'relu'))
	ann.add(Dense(10, activation = 'relu'))
	ann.add(Dense(5))
	
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')

	ann.compile(loss="mean_squared_error",optimizer='SGD',metrics=['mse'])
	history_ann=ann.fit(X_train, y_train, epochs = 50, verbose=1,validation_split=0.3,batch_size=100,callbacks=[early_stopping])
		
	#ann.save("annHR.h5")	
	ytrain_pred=ann.predict(X_train)
	ytest_pred = ann.predict(X_test)
	trainPredict = scaler.inverse_transform(ytrain_pred)
	y_train = scaler.inverse_transform(y_train)
	testPredict = scaler.inverse_transform(ytest_pred)
	y_test = scaler.inverse_transform(y_test)
	train_score = math.sqrt(mean_squared_error(y_train, trainPredict))
	print('Train Score',train_score)
	test_score = math.sqrt(mean_squared_error(y_test, testPredict))
	print('Test Score', test_score)
	mse = mean_squared_error(y_test, testPredict)
	mae = mean_absolute_error(y_test, testPredict)
	mape = mean_absolute_percentage_error(y_test, testPredict)
	rms_error = rmse(y_test, testPredict)
	print('mean square error',mse)
	print('mean absolute error',mae)
	print('mean abs_percentage_error',mape)
	print('root_mean square error',rms_error)
	
	plt.plot(history_ann.history['loss'])
	plt.plot(history_ann.history['val_loss'])
	plt.title('Model Loss Progression During Training/Validation')
	plt.ylabel('Training and Validation Losses')
	plt.xlabel('Epoch Number')
	plt.legend(['Training Loss', 'Validation Loss'])
	plt.show()
	

	#RNN
	print('model used is RNN')
	X_train,X_test,y_train,y_test=train_test_split(XList.to_numpy(), YList.to_numpy(),test_size=0.20,random_state=42)
	print(len(X_train))
	print(len(X_test))
	trainX = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
	testX = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
	rnn = Sequential()
	rnn.add(SimpleRNN(20, input_shape = (20,1 ),activation='relu'))
	rnn.add(Dense(20, activation='relu'))
	rnn.add(Dense(10, activation='relu'))
	rnn.add(Dense(5))
	rnn.summary()
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
	rnn.compile(loss = 'mean_squared_error', optimizer = 'SGD',metrics=['mse'])
	history_rnn = rnn.fit(trainX, y_train, epochs = 50, verbose = 1,validation_split=0.3,batch_size=100,callbacks=[early_stopping])
	
	#rnn.save("rnnHR.h5")
	print("Saved model to disk")

	ytrain_pred=rnn.predict(trainX)
	ytest_pred = rnn.predict(testX)
	trainPredict = scaler.inverse_transform(ytrain_pred)
	y_train = scaler.inverse_transform(y_train)
	testPredict = scaler.inverse_transform(ytest_pred)
	y_test = scaler.inverse_transform(y_test)
	train_score = math.sqrt(mean_squared_error(y_train, trainPredict))
	print('Train Score',train_score)
	test_score = math.sqrt(mean_squared_error(y_test, testPredict))
	print('Test Score', test_score)
	mse = mean_squared_error(y_test, testPredict)
	mae = mean_absolute_error(y_test, testPredict)
	mape = mean_absolute_percentage_error(y_test, testPredict)
	rms_error = rmse(y_test, testPredict)
	print('mean square error',mse)
	print('mean absolute error',mae)
	print('mean abs_percentage_error',mape)
	print('root_mean square error',rms_error)
	plt.plot(history_rnn.history['loss'])
	plt.plot(history_rnn.history['val_loss'])
	plt.title('Model Loss Progression During Training/Validation')
	plt.ylabel('Training and Validation Losses')
	plt.xlabel('Epoch Number')
	plt.legend(['Training Loss', 'Validation Loss'])
	plt.show()
	
	
	
	#LSTM
	X_train,X_test,y_train,y_test=train_test_split(XList.to_numpy(), YList.to_numpy(),test_size=0.20,random_state=42)
	trainX = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
	testX = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
	print('model used in LSTM')
	lstm = Sequential()
	lstm.add(LSTM(input_shape = (20,1),units= 20,  activation='relu',return_sequences = True))
	
	lstm.add(LSTM(20))
	lstm.add(Dense(20, activation='relu'))
	lstm.add(Dense(10, activation='relu'))
	lstm.add(Dense(5))
	lstm.summary()
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
	lstm.compile(loss='mse', optimizer='SGD',metrics=['mse'])
	history_lstm = lstm.fit(trainX, y_train, epochs = 50, verbose=1,validation_split=0.3,batch_size=100, callbacks=[early_stopping])
	
	lstm.save("lstmHR.h5")
	print("Saved model to disk")
	
	ytrain_pred=lstm.predict(trainX)
	ytest_pred = lstm.predict(testX)
	trainPredict = scaler.inverse_transform(ytrain_pred)
	y_train = scaler.inverse_transform(y_train)
	testPredict = scaler.inverse_transform(ytest_pred)
	y_test = scaler.inverse_transform(y_test)
	train_score = math.sqrt(mean_squared_error(y_train, trainPredict))
	print('Train Score',train_score)
	test_score = math.sqrt(mean_squared_error(y_test, testPredict))
	print('Test Score', test_score)
	mse = mean_squared_error(y_test, testPredict)
	mae = mean_absolute_error(y_test, testPredict)
	mape = mean_absolute_percentage_error(y_test, testPredict)
	rms_error = rmse(y_test, testPredict)
	print('mean square error',mse)
	print('mean absolute error',mae)
	print('mean abs_percentage_error',mape)
	print('root_mean square error',rms_error)
	plt.plot(history_lstm.history['loss'])
	plt.plot(history_lstm.history['val_loss'])
	plt.title('Model Loss Progression During Training/Validation')
	plt.ylabel('Training and Validation Losses')
	plt.xlabel('Epoch Number')
	plt.legend(['Training Loss', 'Validation Loss'])
	plt.show()
	
	#Bidirectional lstm
	X_train,X_test,y_train,y_test=train_test_split(XList.to_numpy(), YList.to_numpy(),test_size=0.20,random_state=42)
	trainX = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
	testX = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
	print('model used in BiLSTM')
	bilstm = Sequential()
	bilstm.add(Bidirectional(LSTM(units = 20,return_sequences=True),input_shape=(20,1)))
	
	bilstm.add(Bidirectional(LSTM(units = 20)))
	bilstm.add(Dense(20, activation='relu'))
	bilstm.add(Dense(10, activation='relu'))
	bilstm.add(Dense(5))
	bilstm.summary()
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
	bilstm.compile(loss='mse', optimizer='SGD',metrics=['mse'])
	history_bilstm = bilstm.fit(trainX, y_train, epochs = 50, verbose=1,validation_split=0.3,batch_size=100,callbacks=[early_stopping])
	bilstm.save("bilstmHR.h5")
	print("Saved model to disk")
	
	ytrain_pred=bilstm.predict(trainX)
	ytest_pred = bilstm.predict(testX)
	trainPredict = scaler.inverse_transform(ytrain_pred)
	y_train = scaler.inverse_transform(y_train)
	testPredict = scaler.inverse_transform(ytest_pred)
	y_test = scaler.inverse_transform(y_test)
	train_score = math.sqrt(mean_squared_error(y_train, trainPredict))
	print('Train Score',train_score)
	test_score = math.sqrt(mean_squared_error(y_test, testPredict))
	print('Test Score', test_score)
	mse = mean_squared_error(y_test, testPredict)
	mae = mean_absolute_error(y_test, testPredict)
	mape = mean_absolute_percentage_error(y_test, testPredict)
	rms_error = rmse(y_test, testPredict)
	print('mean square error',mse)
	print('mean absolute error',mae)
	print('mean abs_percentage_error',mape)
	print('root_mean square error',rms_error)
	plt.plot(history_bilstm.history['loss'])
	plt.plot(history_bilstm.history['val_loss'])
	plt.title('Model Loss Progression During Training/Validation')
	plt.ylabel('Training and Validation Losses')
	plt.xlabel('Epoch Number')
	plt.legend(['Training Loss', 'Validation Loss'])
	plt.show()

	t1_stop=time.time()
	print("Elapsed time during the whole program in seconds:",
                                         t1_stop-t1_start) 
	
	
	
	
	
