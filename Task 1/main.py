import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce


data = pd.read_csv('penguins.csv')

#pre processing
encoder = LabelEncoder()
data['gender'] = data['gender'].replace(np.nan,data['gender'].mode()[0])
data['gender'] = encoder.fit_transform(data['gender'])

#feature Scaling
scaling = MinMaxScaler()
data['bill_length_mm']= scaling.fit_transform(data[['bill_length_mm']])
data['bill_depth_mm']= scaling.fit_transform(data[['bill_depth_mm']])
data['flipper_length_mm']= scaling.fit_transform(data[['flipper_length_mm']])
data['body_mass_g']= scaling.fit_transform(data[['body_mass_g']])





#
Y = data.iloc[:,0]
X = data.iloc[:,1:6]


class1Features= X.iloc[0:50, :]
class2Features= X.iloc[50:100, :]
class3Features= X.iloc[100:, :]



def Training(X1, X2, Y1, Y2, eta, m, b1):
        #choosing classes
        global c1
        global c2
        if (Y1 == 1 and Y2 == 2) or (Y1 == 2 and Y2 == 1):
            c1 = 'Adelie'
            c2 = 'Gentoo'
            features1 = class1Features
            features2 = class2Features

            classes1 = Y.iloc[0:50]
            classes2 = Y.iloc[50:100]

        elif (Y1 == 1 and Y2 == 3) or (Y1 == 3 and Y2 == 1):
            c1 = 'Adelie'
            c2 = 'Chinstrap'
            features1 = class1Features
            features2 = class3Features

            classes1 = Y.iloc[0:50]
            classes2 = Y.loc[100:]

        elif (Y1 == 2 and Y2 == 3) or (Y1 == 3 and Y2 == 2):
            c1 = 'Gentoo'
            c2 = 'Chinstrap'
            features1 = class2Features
            features2 = class3Features

            classes1 = Y.iloc[50:100]
            classes2 = Y.iloc[100:]

    #

        classes1 = classes1.replace(c1, 1)
        classes2 = classes2.replace(c2, -1)

        SelectedFeature1c1 = features1.iloc[:, X1-1]
        SelectedFeature2c1 = features1.iloc[:, X2-1]


        SelectedFeature1c2 = features2.iloc[:, X1 - 1]
        SelectedFeature2c2 = features2.iloc[:, X2 - 1]


        cfeaturesc1 = pd.concat([SelectedFeature1c1,SelectedFeature2c1],axis=1)
        cfeaturesc2 = pd.concat([SelectedFeature1c2, SelectedFeature2c2], axis=1)

    #

        X_train1, X_test1, y_train1, y_test1 = train_test_split(cfeaturesc1, classes1, test_size=0.40, shuffle=True)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(cfeaturesc2, classes2, test_size=0.40, shuffle=True)

        X_train = pd.concat([X_train1,X_train2], axis=0)
        X_test = pd.concat([X_test1, X_test2], axis=0)
        y_train = pd.concat([y_train1, y_train2], axis=0)
        y_test = pd.concat([y_test1, y_test2], axis=0)
    #

        X_train = X_train.values.tolist()
        X_test = X_test.values.tolist()
        y_train = y_train.values.tolist()
        y_test = y_test.values.tolist()

#
        c = list(zip(X_train, y_train))
        random.shuffle(c)
        X_train, y_train = zip(*c)

        c = list(zip(X_test, y_test))
        random.shuffle(c)
        X_test, y_test = zip(*c)


        global w1
        global w2
        global b
        w1 = random.random()
        w2 = random.random()
        b = b1


        for i in range(m):
            for j in range(len(X_train)):

              y_pred = (w1 * X_train[j][0]) + (w2 * X_train[j][1]) + b

              if y_pred < 0:
                 y = -1
              else:
                 y = 1


              loss = y_train[j] - y
              w1 = w1 + eta*X_train[j][0]*loss
              w2 = w2 + eta*X_train[j][1]*loss
              if(b):
                 b = b + eta*loss

        #Decision bounday visualisation
        x_train_plot = pd.DataFrame(X_train,columns=['feature1', 'feature2'])
        seaborn.scatterplot(x=x_train_plot.iloc[:,0], y=x_train_plot.iloc[:,1], hue=y_train)

        y1 = -(b+ w1 *min(min(X_train))) / w2
        y2 = -(b+ w1 *max(max(X_train))) / w2
        y = [y1, y2]
        x = [min(min(X_train)), max(max(X_train))]

        plt.plot(x, y)
        plt.title("decision boundary")
        plt.show()
        accuracy = Testing(X_test, y_test)
        return accuracy




def Testing(X_test, Y_test):
    tp = 0
    tn = 0
    fp=0
    fn=0
    for i in range(len(X_test)):
        y_pred = w1*X_test[i][0] + w2 * X_test[i][1]+b
        if y_pred < 0:
            y = -1
        else:
            y = 1
        loss = int(Y_test[i]) - y

        if y==1 and loss==0:
            tp=tp+1
        elif y==-1 and loss==0:
            tn=tn+1
        elif y==1 and loss!=0:
            fn=fn+1
        elif y == -1 and loss!=0:
            fp =fp+ 1

    print("Confusion Matrix")
    print("TP",' ',"FP")
    print(tp," ",fp)
    print("FN", ' ', "TN")
    print(fn," ",tn)
    accuracy = ((tp+tn)/40)*100
    return accuracy

def userTesting(x1,x2,b):
    y_pred = w1* x1 + w2*x2 + b
    if y_pred < 0:
        y = c2
    else:
        y = c1
    return y


