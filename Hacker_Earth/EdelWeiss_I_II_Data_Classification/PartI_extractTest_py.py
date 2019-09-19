import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
workPath = "V:/Naveen/Learning/Edelweiss"
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
    

testCSVFileName = os.path.join(workPath, "test_foreclosure.csv");
trainCSVFileName = os.path.join(workPath, "train_foreclosure.csv");
lmsCSVFileName = os.path.join(workPath, "LMS_31JAN2019.csv");

train_DataCalculatedFile = os.path.join(workPath, "train_LMS_CalculatedDataFile.csv");

allTestLines = [];
allTrainLines = [];
allLMSLines = [];

cols = [];
for i in range(0, 22):
    cols.append(i);
avgCols = cols[1:6] + cols[7:10] + cols[11:12] + cols[13:14] +  cols[16:18] + cols[19:len(cols)+1]      

maxCols = cols[10:11] + cols[12:13] + cols[14:15]

minCols = cols[15:16]

strCols = cols[6:7] + cols[18:19]

#print(avgCols)
#print(maxCols)
#print(minCols)
#print(strCols)

allTrainCalculatedLines = [];

def ExtractCSV():    	
    if os.path.exists(testCSVFileName):
        with open(testCSVFileName) as csvFile:
            allTestLines = csvFile.readlines();
            print(len(allTestLines));
            
    if os.path.exists(trainCSVFileName):
        with open(trainCSVFileName) as csvFile:
            allTrainLines = csvFile.readlines();
            print(len(allTrainLines));
            
    testAgreementIDs = [];
    
    if len(allTestLines) > 0:
        testAgreementIDs = [item.split(",")[0] for item in allTestLines[1:]];
        
    trainAgreementIDs = [];
    
    if len(allTrainLines) > 0:
        trainAgreementIDs = [item.split(",")[0] for item in allTrainLines[1:]];
        
    print(len(testAgreementIDs));
    print(len(trainAgreementIDs));
    		
    if os.path.exists(lmsCSVFileName):
        with open(lmsCSVFileName) as csvFile:
            allLMSLines = csvFile.readlines();
            print(len(allLMSLines));
    
    testAgreeCustLines = [];
    testAgreeCustLines = [x for x in list(allLMSLines) if x.split(",")[0] in testAgreementIDs];
    
    trainAgreeCustLines = [];
    trainAgreeCustLines = [x for x in list(allLMSLines) if x.split(",")[0] in trainAgreementIDs];
    
    print(len(testAgreeCustLines));        
    print(len(trainAgreeCustLines));
    
    train_DataFile = os.path.join(workPath, "train_LMS_DataFile.csv");
    test_DataFile = os.path.join(workPath, "test_LMS_DataFile.csv");
    
    with open(train_DataFile, "w+") as csvFile:
        csvFile.write(''.join(trainAgreeCustLines));
    
    with open(test_DataFile, "w+") as csvFile:
        csvFile.write(''.join(testAgreeCustLines));
        
def Calculate(selectedLines):
    try:
        calculatedLine = [None] * 22;        
        
        calculatedLine[0] = selectedLines[0][0];
        
        # Calculate avg
        for i in list(avgCols):
            sum = 0.0;
            for j in range(0, len(selectedLines)):
                if selectedLines[j][i].strip() == "": continue
                try:
                    sum = sum + float(selectedLines[j][i]);
                except:
                    continue;
#                print(sum)
            avg = sum / len(selectedLines);
            
            calculatedLine[i] = round(avg, 2);        
        
        # Calculate max value
        for i in list(maxCols):
            vals = [];
            for j in range(0, len(selectedLines)):
                if selectedLines[j][i].strip() == "": continue
                try:
                    vals.append(float(selectedLines[j][i]));
                except:
                    continue;
                    
            if len(vals) > 0:
                maxVal = max(vals);
            
                calculatedLine[i] = round(maxVal);
            else:
                calculatedLine[i] = 0;
            
        # Calculate min value
        for i in list(minCols):
            vals = [];
            for j in range(0, len(selectedLines)):
                if selectedLines[j][i].strip() == "": continue;
                try:
                    vals.append(float(selectedLines[j][i]));
                except:
                    continue;
            if len(vals) > 0:
                minVal = min(vals);
            
                calculatedLine[i] = round(minVal);
            else:
                calculatedLine[i] = 0;
        
        # Select any one string
        for i in list(strCols):
            calculatedLine[i] = selectedLines[0][i];   
        
#        print(calculatedLine);
        return calculatedLine;
    except Exception as ex:
        print("Failed to Calculate. Exception: " + str(ex));
        
def CleanUpData(oldFile, newFile):
    try:
        trainEditedFile = os.path.join(workPath, oldFile);     
        
        if os.path.exists(trainEditedFile):
            with open(trainEditedFile) as csvFile:
                allTrainEditedLines = csvFile.readlines();
#                print(len(allTrainEditedLines));        
        
        allUniqueIDs = [];
        
        for i in range(2, len(allTrainEditedLines)):
            splitLine1 = allTrainEditedLines[i].split(",");
            if not splitLine1[0] in allUniqueIDs:
                allUniqueIDs.append(splitLine1[0]);
        
        allTrainCalculatedLines = [];
        cnt = 2
        for id in list(allUniqueIDs):
            tempLines = [];
            for i in range(cnt, len(allTrainEditedLines)):                            
                splitLine1 = allTrainEditedLines[i].split(",");
                if splitLine1[0] == "":
                    break;
                    
                if splitLine1[0] == id:
                    tempLines.append(splitLine1);
                else:
                    cnt = i;
                    break;
            
            if len(tempLines) > 0:
                allTrainCalculatedLines.append(Calculate(tempLines));
        
        train_DataCalculatedFile = os.path.join(workPath, newFile);
        
        with open(train_DataCalculatedFile, "w+") as csvFile:
            for calLine in allTrainCalculatedLines:
                line = ""
                for eachItem in calLine:
                    if line == "":
                        line = line + str(eachItem);
                    else:
                        line = line + "," + str(eachItem);
                csvFile.write(line + "\n");
    except Exception as ex:
        print("Failed to clean up data" + str(ex));   
        
def ReadCSV():        
    X = pd.read_csv("train_LMS_CalculatedDataFile.csv")
    y = pd.read_csv("train_foreclosure.csv")
    
    y = y['FORECLOSURE']    
    
    y.fillna(-1, inplace = True)
    
    X_RealTest = pd.read_csv("test_LMS_CalculatedDataFile.csv")    
    
def ExecuteClassifier():
    X.head(10)
    
    labelencoder = LabelEncoder()
#    X['AGREEMENTID'] = labelencoder.fit_transform(X['AGREEMENTID'])
    
    X['PRODUCT'] = labelencoder.fit_transform(X['PRODUCT'])
    
    
    X['CITY'].fillna('NOCITY', inplace = True)
    X['CITY'] = labelencoder.fit_transform(X['CITY'])
    
    # X Real Test    
#    X_RealTest['AGREEMENTID'] = labelencoder.fit_transform(X_RealTest['AGREEMENTID'])
#    TestAgreeIDS = labelencoder.inverse_transform(X_RealTest['AGREEMENTID'])
    
    X_RealTest['PRODUCT'] = labelencoder.fit_transform(X_RealTest['PRODUCT'])
    
    
    X_RealTest['CITY'].fillna('NOCITY', inplace = True)
    X_RealTest['CITY'] = labelencoder.fit_transform(X_RealTest['CITY'])
    
#    onehotencoder_X = OneHotEncoder(categorical_features=[0])
#    X = onehotencoder_X.fit_transform(X).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
    y_test = pd.DataFrame(y_test)
    
    y_test.to_csv('y_test.csv')
    
#    X_train = X_train.values
#    
#    y_train = y_train.values
#    
#    y_test = y_test.values
#    
#    X_test = X_test.values
    
def LogRegression():
#    LogReg_classifier = LogisticRegression(random_state = 0)
#    LogReg_classifier.fit(X_train, y_train)


    KNN_classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
    KNN_classifier.fit(X_train, y_train)
    
    yPred = KNN_classifier.predict(X_test);   
    
    yPredDf = pd.DataFrame(yPred, columns=["FORECLOSURE"])
    
    yPredDf.to_csv("yPred.csv")
    
    y_test = y_test.values.tolist()
    
    pos = 0;
    neg = 0;
    for i in range(0, len(y_test)):
        if yPred[i] == y_test[i]:
            pos +=1;
            
    accuracy = (pos/ len(y_test)) * 100
    print("Accuracy: $", accuracy)
    
    y_PredReal = KNN_classifier.predict(X_RealTest);
    
    y_PredRealDf = pd.DataFrame(y_PredReal, columns=["FORECLOSURE"])
    
    y_PredRealDf.to_csv("yPredReal.csv")
    
    list(TestAgreeIDS + y_PredReal)
    
    labelencoder.inverse_transform(X_RealTest['AGREEMENTID'])
    
    
    

def Main():
    start_time = time.time();
    
#    ExtractCSV();
    
#    CleanUpData("train_LMS_DataFile_Edited.csv", "train_LMS_CalculatedDataFile.csv");
#
#    CleanUpData("test_LMS_DataFile_Edited.csv", "test_LMS_CalculatedDataFile.csv");
    
    ReadCSV();
    
#    ExecuteClassifier();
    
    print('Time: ', time.time() - start_time);   
    
Main();