# In[116]:


# create three txt files: one with all the data, one with only the test and one with only the train data
import csv, math, random
import numpy as np
from anytree import Node, RenderTree
import copy

minLeafLen = 2

#this function returns false if what was sent wasnt a number else it returns the number as float
#for some reason it does not work for 0
def is_number(s):
    try:
        return float(s)
    except ValueError:
        return False

#if the decision column has 1 or 0 we change that to Yes/No   
with open('example.csv', 'r') as r, open('heartYESNO.csv', 'w') as w:
    data = r.readlines()
    w.write(data[0])
    check = data[1].rstrip('\n').split(',')[-1]
    if type(is_number(check)) == float:
#         print("mpika")
        for i in data[1:]:
            newData = i.split(',')
            if int(newData[-1]) == 1:
                target = 'Yes'
            elif int(newData[-1]) == 0:
                target = 'No'
            newData = newData[:-1]
            newData.append(target)
            w.write(','.join(newData))
            w.write('\n')
    else:
        for i in data[1:]:
            w.write(i)

with open('heartYESNO.csv', 'rt') as csvFile, open("heartTRAIN.txt", "w") as train, open("heartTEST.txt", "w") as test:
    totalLines = sum(1 for _ in csvFile)
    csvFile.seek(0)  # go back to beginning of file
    
    reader=csv.reader(csvFile,delimiter=",")
    totalColumns=len(next(reader))
    csvFile.seek(0) 
    
    trainLines = math.floor(totalLines*0.9) #keep the 70% of the data for the training part
    reader = csv.reader(csvFile)

    i=0 
    for row in reader:
        if i == 0:
            print(','.join(row), file=train)
            print(','.join(row), file=test)
        elif i <= trainLines:
            print(','.join(row), file=train)
        else:
            print(','.join(row), file=test)
        i+=1

data = [[] for f in range(totalColumns+1)]
f = open('heartTRAIN.txt', 'r')
line = f.readline()

line = line.rstrip('\n')  # remove end of line char
line2 = line.split(",")
for index,i in enumerate(line2):
    data[0].append(i)
line = f.readline()

while line:
    line = line.rstrip('\n')  # remove end of line char
    line2 = line.split(",")
    for index,i in enumerate(line2):
        if i == 0 or i == '0':
            data[index+1].append(float(i))
        elif is_number(i):
            data[index+1].append(float(i))
        else:
            data[index+1].append(str(i))
    line = f.readline()

dataTEST = []
f = open('heartTEST.txt', 'r')
line = f.readline()

line = line.rstrip('\n')  # remove end of line char
line2 = line.split(",")
dataTEST.append(line2)
line = f.readline()

while line:
    temp = []
    line = line.rstrip('\n')  # remove end of line char
    line2 = line.split(",")
    for i in line2:
        if i == 0 or i == '0':
            temp.append(float(i))
        elif is_number(i):
            temp.append(float(i))
        else:
            temp.append(str(i))
    dataTEST.append(temp)
    line = f.readline()
# print("test Data= ", dataTEST)
 


# In[117]:


#returns a column   
def get_a_column(num,data):
    dataGiven = copy.deepcopy(data)
    return data[num]

#returns the entropy of one column, works for both cont. and discr. data
def calc_entropy(column):
    totalLines = len(column)
    value,counts = np.unique(column, return_counts=True)
    probs = counts / totalLines
    n_classes = np.count_nonzero(probs)
    entropy = 0
    for i in probs:
        entropy -= i * math.log(i, 2)
    return entropy

#returns a column, specifically: the values of the targetCol that have the givenValue on the valuesCol
def get_rows(givenValue, targetCol, valuesCol):
    output = []
    for index,i in enumerate(valuesCol):
        if i == givenValue:
            output.append(targetCol[index])
    return output 

#calculates the gain for both cont and discrete values
def calc_gain_of_column(givenData, breakValue=None):
    totalLines = len(givenData[1]) #how many lines of data we have
    entropy = calc_entropy(givenData[0])
    
    if givenData[1][0] == 0 or givenData[1][0] == "0": #we had a problem with 0 
        breakValue = 0
    elif is_number(givenData[1][0]):
        # then we believe it is a continues field
        breakValue = givenData[1][0] # give initial brekValue
    
    if breakValue != None: #for continious values
        value,counts = np.unique(givenData[1], return_counts=True) 
        gainRatioCon = []
        gainTable = []
        for index,breakValue in enumerate(value):
            entropyLow = []
            entropyHigh = []
            for index,i in enumerate(givenData[1]): #take the values of the values column
                if i <= breakValue:
                    entropyLow.append(givenData[0][index])
                else:
                    entropyHigh.append(givenData[0][index])
            entropies = np.array([calc_entropy(entropyLow), calc_entropy(entropyHigh)])  
            counts = np.array([len(entropyLow), len(entropyHigh)])
            probs = counts/totalLines
            
            x = 0
            for i in range(len(probs)):
                x -= probs[i]*entropies[i]
                
            # splitInfoCon only need the numbers of each leaf and 
            splitInfoCon = get_split_info_con(counts[0], counts[1], totalLines)  
            gainRatioCon.append([get_gain_ratio(x+entropy,splitInfoCon), breakValue])
            
            ## edo koitame gia an to kathe paidi pou tha prokipsei exei eparki fula
            ## midenizoume to gain tou analogou stoixeiou kai opote to max den protimaei
            ## auti tin diaspasi
            if counts[0] < minLeafLen or counts[1] < minLeafLen:
                gainTable.append(0)
            else:
                gainTable.append(x+entropy)
        tempV = max(gainRatioCon)
        
        # me vasei to megalitero gain perno to analogo gain ratio index
        return gainRatioCon[gainTable.index(max(gainTable))][0], gainRatioCon[gainTable.index(max(gainTable))][1]
    
    else:   #for discrete values
        #values: the discrete values, counts: how many times each value is seen
        value,counts = np.unique(givenData[1], return_counts=True) 
        x = 0
        for index,i in enumerate(value):
            colElem = get_rows(i, givenData[0], givenData[1])
            sizeAll = sum(counts)
            probs = counts[index] / sizeAll
            x -= calc_entropy(colElem)*probs
        splitInfo = get_split_info(givenData[1], None)
        tempGain = entropy + x 
        return get_gain_ratio(tempGain, splitInfo), breakValue
    
#returns an array that has only the data with the given value with all its columns
def get_smaller_array(nodeIndex, value, data):
    dataGiven = copy.deepcopy(data) #create a copy so that python does not compromise the original data
    output = [[] for f in range(len(dataGiven))]
    output[0]=dataGiven[0] #the names of the collumns stay the same
    for index,i in enumerate(dataGiven[nodeIndex]):
        if i == value:
            for j in range(1,len(dataGiven)):
                output[j].append(dataGiven[j][index])
    output.pop(nodeIndex) # delete the column that was used as node of the tree
    output[0].pop(nodeIndex-1) # and delete its name
    return output   


#returns an array that has only the data with the given breakValue or higher if high=1 all its columns
#else returns an array that has only the data with lower than the breakValue if high=0 all its columns
def get_smaller_array_con(nodeIndex, breakValue, data, high):
    dataGiven = copy.deepcopy(data) #create a copy so that python does not compromise the original data
    output = [[] for f in range(len(dataGiven))]
    output[0] = dataGiven[0] #the names of the collumns stay the same
    
    if high == 1:
        for index, i in enumerate(dataGiven[nodeIndex]):
            if float(i) > float(breakValue):
                for j in range(1, len(dataGiven)):
                    output[j].append(dataGiven[j][index])
    else:
        for index,i in enumerate(dataGiven[nodeIndex]):
            if i <= breakValue:
                for j in range(1,len(dataGiven)):
                    output[j].append(dataGiven[j][index])

    output.pop(nodeIndex) # delete the column that was used as node of the tree
    output[0].pop(nodeIndex-1) # and delete its name
    return output 

# sorts the contDataCol and adjusts the desCol accordinately
def sortData(desCol, contDataCol):
    output= [desCol,contDataCol]
    n = len(contDataCol)
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            if contDataCol[j] > contDataCol[j+1] :
                contDataCol[j], contDataCol[j+1] = contDataCol[j+1], contDataCol[j]
                desCol[j], desCol[j+1] = desCol[j+1], desCol[j]       
    output = [desCol,contDataCol]
    return output

#works for discrete and cont values, finds splitInfo 
def get_split_info(valuesCol, breakValue):
    totalLines = len(valuesCol)
    value,counts = np.unique(valuesCol, return_counts=True)
    if breakValue != None: #for cont data we need to create a new count array
        index = value.index(breakValue)
        newCount = 0
        for i in range(index+1):
            newCount += counts[i]
        counts = np.array([newCount, totalLines-newCount]) 
    probs = counts / totalLines
    x = 0
    for i in probs:
        x -= i*math.log(i, 2)
    return x

#works for cont values, finds splitInfo we know we have only 2 fields 
def get_split_info_con(low, high, total):
    low = low/total
    high = high/total

    if low == 0:    
        x = -low*math.log(low,2)
    elif high == 0:    
        x = -low*math.log(low,2)
    else:
        x = -low*math.log(low,2) - high*math.log(high,2)
    return x
    
#returns the gain ratio
def get_gain_ratio(gain, split):
    if split == 0:
        return 0
    return gain/split

#finds the max gainRatio on a cont. column
def find_max_gain_ratio(desCol, contDataCol):
    sortedData = sortData(desCol, contDataCol)
    value = np.unique(sortedData[1])
    gains = []
    for i in value:
        gains.append([getGainContData(sortedData, int(i)), i])
    maxGain = maxValue = 0
    for i in gains:
        if i[0] > maxGain:
            maxGain = i[0]
            maxValue = i[1]
    return maxGain, maxValue

# targetCol is an array with yes,no values
# counts the number of yes/no values and returns the greatest
# if the numbers are equal then play it safe and return no
def get_mo_of_des(targetCol):
    yes = no = 0
    for i in targetCol:
        if i == "Yes" or i == 1 or i == "yes" or i =="y" or i=="Y":
            yes += 1
        else:
            no += 1     
    if yes > no:
        return "Yes"
    else:
        return "No"

# implements the desicion tree andreturns it
def tree(data, node):
    dataGiven = copy.deepcopy(data) #create a copy so that python does not compromise the original data
    if dataGiven != None: 
        gain = []
        gainRatio = []
        targetCol= get_a_column(-1, dataGiven) #get the target column and check if it 
        targetValue = np.unique(targetCol)    #has more than one targets (eg. yes/no)
        
        if len(targetValue) <= 1:
            Node(targetValue[0], parent=node)
            
        #this conditions helps so that the tree haws enough examples to split
        #the minLeafLen*2 is used so that every child (in cont data) of the tree has enough examples
        elif minLeafLen*2 > len(targetCol):
#             nameDebug = ("kopikeeeeeeeee", get_mo_of_des(targetCol))
            name = get_mo_of_des(targetCol)
            Node(name, parent=node)
            get_mo_of_des(targetCol)
            
        elif len(targetValue) > 1:              #if it does then find the gain of every column
            for i in range(1, len(dataGiven)-1):
                column = get_a_column(i, dataGiven)
                calc_gainRatio, breakValue = calc_gain_of_column([targetCol, column], None)
                gain.append([calc_gainRatio, i-1, breakValue])

            nodeIndex = max(gain)[1]          #find which column has the biggest gain

            #if variable node is empty then we are at the root of the tree
            #else we connect the nodes together
            if node == None: 
                root = Node(dataGiven[0][nodeIndex]) 
            else: 
                root = Node(dataGiven[0][nodeIndex], parent=node) 

            #take each of the values (eg. hot,mild,cool) and call tree for each one
            values = np.unique(dataGiven[nodeIndex+1]) 

            # check if the chosen column has a breakValue so it is continues data column
            if gain[nodeIndex][2] == None:
                #take each of the values (eg. hot,mild,cool) and call tree for each one
                values = np.unique(dataGiven[nodeIndex+1]) 
                for i in values:
                    nodeChild = Node(i, parent=root) 
                    tree(get_smaller_array(nodeIndex+1, i, dataGiven), nodeChild)    
            else:
                #break continues data column with the breakValue
                nameMax = ">" + str(gain[nodeIndex][2])
                nodeChild = Node(nameMax, parent=root)
                tree(get_smaller_array_con(nodeIndex+1, gain[nodeIndex][2], dataGiven, 1), nodeChild)
                nameMin = "<=" + str(gain[nodeIndex][2]) 
                nodeChild = Node(nameMin, parent=root)
                tree(get_smaller_array_con(nodeIndex+1, gain[nodeIndex][2], dataGiven, 0), nodeChild)
            return root
    #if we only have one target then we have a leaf
        else:
            Node(targetValue[0], parent=node)  
            


# In[118]:


copy_data = copy.deepcopy(data) #create a copy so that python does not compromise the original data

#creates and prints the tree
root = tree(copy_data,None) 
for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))
               


# In[119]:


from anytree import Node, RenderTree, AsciiStyle, PostOrderIter
import re
treeArrayTemp = [node.name for node in PostOrderIter(root)] #writes the tree in post order 

# iterate testArray and stop when we find a value on the current index
# eg. if we have ['outlook', 'wind', 'strong', 'yes'] then we will stop at the strong 
# cause outlook and wind are the names of the fields while strong is a value
def iterate(i,thisValue, test, testArray):
    flag = 0
    while flag == 0:
        try:
            pointer = test[0].index(thisValue)
            thisValue = testArray[pointer]
            i += 1
        except:
            flag = 1
            i -= 1
    return i, thisValue, pointer

# Here we give test data and we write in a file our predictions according to the tree we made
# also return number of correct and false predictions
def predict(dataTEST):
    f = open('results.txt', 'w')
    test = copy.deepcopy(dataTEST) #copy the data
    itertest = iter(test)          #itertest cointains olny the data we want to 
    next(itertest)                 #predict without the names of the atributes
    treeArray = []
    for i in reversed(treeArrayTemp):
        treeArray.append(i)
#     print("treeArray = " , treeArray)

    rootEl = treeArray.pop(0) # pop the root element cause it is always the same
    correct = false = 0
    for testArray in itertest:

#         print("------------")
#         print("Was: ", testArray[-1])

        pointer = test[0].index(rootEl) #finds the name of the attribute of the root
        thisValue = testArray[pointer]  #finds the value testArray has in that attribute
        i = 0;
        while i in range(len(treeArray)):  

            #for continious values:
            if is_number(thisValue) or thisValue =='0' or thisValue == 0:
                if treeArray[i].find("<=") != -1: #check if the tree value has <= in its name
                    value = treeArray[i].split("<=")
                    if thisValue <= float(value[1]): #for values lower than treeNode value
                        thisValue = treeArray[i+1]
                        if thisValue == 'Yes' or thisValue == 'No':
                            f.write(thisValue+'\n')
#                             print("Found: ",thisValue)
                            if thisValue == testArray[-1]: #find if the predictions was correct
                                correct += 1
                            else:
                                false += 1
                            break
                        else:
                            i, thisValue, pointer = iterate(i, thisValue, test, testArray)
                elif treeArray[i].find(">") != -1: #check if the tree value has > in its name
                    value = treeArray[i].split(">")
                    if thisValue > float(value[1]): #for values greater than treeNode value
                        thisValue = treeArray[i+1]
                        if thisValue == 'Yes' or thisValue == 'No':
                            f.write(thisValue+'\n')
#                             print("Found: ",thisValue)
                            if thisValue == testArray[-1]: #find if the predictions was correct
                                correct += 1
                            else:
                                false += 1
                            break
                        else:
                            i, thisValue, pointer = iterate(i, thisValue, test, testArray)

            #for discrete values:
            else:
                if treeArray[i] == thisValue:
                    thisValue = treeArray[i+1]
                    i += 1
                    if thisValue == 'Yes' or thisValue == 'No': 
                        f.write(thisValue+'\n')
#                         print("Found: ",thisValue)
                        if thisValue == testArray[-1]: #find if the predictions was correct
                                correct += 1
                        else:
                            false += 1
                        break
                    else:
                        i, thisValue, pointer = iterate(i, thisValue, test, testArray)
            i+= 1        
    return correct,false


# In[120]:


#predict results for dataTEST and find statistics

correct,false=predict(dataTEST)
print("Correct predictions:", correct)
print("False predictions:", false)
