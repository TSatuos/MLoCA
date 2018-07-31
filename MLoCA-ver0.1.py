# -*- coding: utf-8 -*-
# python3.xで動かしてくださ
# .csvファイルはこのプログラム（sample.py）と同じフォルダに置いてください
import pandas as pd
import numpy as np
from sklearn import tree,metrics,ensemble,naive_bayes,svm
def switch(a):
        if a==1:
            print("決定木")
        elif a==2:
            print("ランダムフォーレスト")
        elif a==3:
            print("ナイーブベイズ")
        else:
            print("リニアSVM")

def get(char,pred, test_y):
    #４つの指標(TP, FP, FN, TN)
    truly_normal=test_y==1
    truly_attack=test_y==-1
    positive=pred==-1
    negative=pred==1
    tn=len(test_y[truly_normal&negative])
    fp=len(test_y[truly_normal&positive])
    tp=len(test_y[truly_attack&positive])
    fn=len(test_y[truly_attack&negative])
    if char=='w':
        return (fp)/(tn+fp) #誤検知率 
    if char=='a':
        return (tp+tn)/(tp+tn+fp+fn) #正解率 
    if char=='c':
        return (tp)/(tp+fn) #検知率 
    
def printANS(a,pred, test_y):
    switch(a)
    #４つの指標(TP, FP, FN, TN)
    truly_normal=test_y==1
    truly_attack=test_y==-1
    positive=pred==-1
    negative=pred==1
    tn=len(test_y[truly_normal&negative])
    fp=len(test_y[truly_normal&positive])
    tp=len(test_y[truly_attack&positive])
    fn=len(test_y[truly_attack&negative])
    print("検知率   ",(tp)/(tp+fn)*100,"%       誤検知率    ",(fp)/(tn+fp)*100,"%       正解率  ",(tp+tn)/(tp+tn+fp+fn)*100,"%      適合率(精度)    ",(tp)/(tp+fp)*100,"%")

def main():
    #学習させたいデータを読み込む    
    train = pd.read_csv('train.csv') #CSVファイルの取得
    train_x = train[['x1','x3','x4','x5']] #学習させる特徴量をtrain_xへ
    train_y = train['x18'] #学習させる目的関数（正解ラベル）をtrain_yへ
    train_y /= abs(train_y) #-2を-1に修正

    #判別させたいデータを読み込む
    test = pd.read_csv('test.csv') 
    test_x = test[['x1','x3','x4','x5']]
    test_y = test['x18']
    test_y /= abs(test_y) #-2を-1に修正
    
    #判別器の作成・学習::決定木１、ランダムフォーレスト２、ナイーブベイズ３、リニアSVM４
    clf1 = tree.DecisionTreeClassifier(random_state=0) #判別器（決定木）を作成
    clf1 = clf1.fit(train_x,train_y) #fit(,) 第一引数：学習させたい特徴量,第二引数：学習させたい目的関数
    clf2 = ensemble.RandomForestClassifier(random_state=0) #判別器（決定木）を作成
    clf2 = clf2.fit(train_x,train_y) #fit(,) 第一引数：学習させたい特徴量,第二引数：学習させたい目的関数
    clf3 = naive_bayes.MultinomialNB(fit_prior=False) #判別器（決定木）を作成
    clf3 = clf3.fit(train_x,train_y) #fit(,) 第一引数：学習させたい特徴量,第二引数：学習させたい目的関数
    clf4 = svm.LinearSVC(random_state=0) #判別器（決定木）を作成
    clf4 = clf4.fit(train_x,train_y) #fit(,) 第一引数：学習させたい特徴量,第二引数：学習させたい目的関数

    #実際に判別する
    pred1 = clf1.predict(test_x) #predict() 引数：判定させたい特徴量
    #printANS(1,pred1,test_y)
    pred2 = clf2.predict(test_x) #predict() 引数：判定させたい特徴量
    #printANS(2,pred2,test_y)
    pred3 = clf3.predict(test_x) #predict() 引数：判定させたい特徴量
    #printANS(3,pred3,test_y)
    pred4 = clf4.predict(test_x) #predict() 引数：判定させたい特徴量
    #printANS(4,pred4,test_y)

    #誤検知率
    w1=get('w',pred1,test_y)
    w2=get('w',pred2,test_y)
    w3=get('w',pred3,test_y)
    w4=get('w',pred4,test_y)
    Wrong=[w1,w2,w3,w4]
    #正解率
    a1=get('a',pred1,test_y)
    a2=get('a',pred2,test_y)
    a3=get('a',pred3,test_y)
    a4=get('a',pred4,test_y)

    c1=get('c',pred1,test_y)
    c2=get('c',pred2,test_y)
    c3=get('c',pred3,test_y)
    c4=get('c',pred4,test_y)
    
    #黄金比による重みの付加
    l=(1.618*1.618)/(1.618*1.618+1.618+1)#large
    m=1.618/(1.618*1.618+1.618+1)#medium
    s=1/(1.618*1.618+1.618+1)#small

    if max(Wrong)==w1:
        #誤検知率による学習1の排除
        Ac=[a2,a3,a4]
        #正解率最大法を重み大
        if max(Ac)==a2:
            #検知率大きい方を重み中
            if max([c3,c4])==c3:
                FinAns=l*pred2+m*pred3+s*pred4
            else :
                FinAns=l*pred2+m*pred4+s*pred3
        elif max(Ac)==a3:
            #検知率大きい方を重み中
            if max([c2,c4])==c2:
                FinAns=l*pred3+m*pred2+s*pred4
            else :
                FinAns=l*pred3+m*pred4+s*pred2
        else :# max(Ac)==a4
            #検知率大きい方を重み中
            if max([c3,c2])==c3:
                FinAns=l*pred4+m*pred3+s*pred2
            else :
                FinAns=l*pred4+m*pred2+s*pred3            
    elif max(Wrong)==w2:
        #誤検知率による学習の排除
        Ac=[a1,a3,a4]
        #正解率最大法を重み大
        if max(Ac)==a1:
            #検知率大きい方を重み中
            if max([c3,c4])==c3:
                FinAns=l*pred1+m*pred3+s*pred4
            else :
                FinAns=l*pred1+m*pred4+s*pred3
        elif max(Ac)==a3:
            #検知率大きい方を重み中
            if max([c1,c4])==a1:
                FinAns=l*pred3+m*pred1+s*pred4
            else :
                FinAns=l*pred3+m*pred4+s*pred1
        else :#max(Ac)==a4
            #検知率大きい方を重み中
            if max([c3,c1])==c3:
                FinAns=l*pred4+m*pred3+s*pred1
            else :
                FinAns=l*pred4+m*pred1+s*pred3
    elif max(Wrong)==w3:
        #誤検知率による学習の排除
        Ac=[a1,a2,a4]
        #正解率最大法を重み大
        if max(Ac)==a1:
            #検知率大きい方を重み中
            if max([c2,c4])==c2:
                FinAns=l*pred1+m*pred2+s*pred4
            else :
                FinAns=l*pred1+m*pred4+s*pred2
        elif max(Ac)==a2:
            #検知率大きい方を重み中
            if max([c1,c4])==c1:
                FinAns=l*pred2+m*pred1+s*pred4
            else :
                FinAns=l*pred2+m*pred4+s*pred1
        else :#max(Ac)==a4
            #検知率大きい方を重み中
            if max([c2,c1])==c2:
                FinAns=l*pred4+m*pred2+s*pred1
            else :
                FinAns=l*pred4+m*pred1+s*pred2
    else :#max(Wrong)==w4
        #誤検知率による学習の排除
        Ac=[a1,a2,a3]
        #正解率最大法を重み大
        if max(Ac)==a1:
            #検知率大きい方を重み中
            if max([c3,a2])==c3:
                FinAns=l*pred1+m*pred3+s*pred2
            else :
                FinAns=l*pred1+m*pred2+s*pred3
        elif max(Ac)==a3:
            #検知率大きい方を重み中
            if max([a1,a2])==a1:
                FinAns=l*pred3+m*pred1+s*pred2
            else :
                FinAns=l*pred3+m*pred2+s*pred1
        else :#max(Ac)==a2
            #検知率大きい方を重み中
            if max([c3,c1])==c3:
                FinAns=l*pred2+m*pred3+s*pred1
            else :
                FinAns=l*pred2+m*pred1+s*pred3
    
    FinAns=FinAns/abs(FinAns)
    #正解率の表示
    FinAns = sum(FinAns == test_y) / len(test_y)
    print(FinAns*100, "%")

if __name__ == "__main__":
    main()