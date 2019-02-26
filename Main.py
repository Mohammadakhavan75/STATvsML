from sklearn.linear_model import LogisticRegression
from numpy import genfromtxt
from sklearn import svm
from keras import Sequential
from keras.layers import Dense
import time

x = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(1, 2, 3, 4))
y = genfromtxt("E:\\Work\\GANN\\Data\\mydata.csv", delimiter=',', skip_header=1, usecols=(0,))

x_train = x[:900000]
y_train = y[:900000]

y_test = y[900000:]
x_test = x[900000:]

s = time.time()
logreg = LogisticRegression()
fit = logreg.fit(x_train, y_train)
e = time.time()
print(e-s)

yhat = fit.predict(x_test)

err = 0
for i in range(len(x_test)):
    if y_test[i] != yhat[i]:
       err += 1
print(100-(err/100000))

s = time.time()
clf = svm.SVC(gamma='scale')
clf.fit(x_train, y_train)
e = time.time()
print(e-s)

yhat = clf.predict(x_test)

err = 0
for i in range(len(x_test)):
    if y_test[i] != yhat[i]:
       err += 1
print(100-(err/100000))

model = Sequential()
model.add(Dense(10, activation='relu', use_bias=False, input_shape=(4,)))
model.add(Dense(20, activation='relu', use_bias=False))
model.add(Dense(1, activation='sigmoid', use_bias=False))
model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])

s = time.time()
model.fit(x_train, y_train, epochs=5, validation_split=0.1)
e = time.time()
print(e-s)

acc = model.evaluate(x_test, y_test)[1]
print(acc)