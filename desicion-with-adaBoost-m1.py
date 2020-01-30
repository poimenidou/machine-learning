#Code to accompany Machine Learning Recipes #8. 
#We'll write a Decision Tree Classifier, in pure Python.
#Below each of the methods, I've written a little demo to help explain what it does.

#basic tree format taken from:
#We found which question was the best for the tree split.Then we saved each question 
# in a tree where each node had the previously mentioned question and
# question ID(which column refered to which question).
# next stump shows what's the next stump in the forest.

# Basic decision tree taken from: 
# https://github.com/random-forests/tutorials/blob/master/decision_tree.py?fbclid=IwAR3es6hiOaN8_5jls_hCXFI48c3u9aUUX9o9JNS7t9qfjPLA5pL-Sj2zGBY


# In[26]:


import math, csv, os
import random
import copy
# For Python 2 / 3 compatability
from __future__ import print_function


# In[27]:


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))
    
    
# Since we saved the question in the tree(in order for the tree to be more 
# understendable to humans),we wanted to extract the data from the question.
# etc question= "is weight >= 180?",we kept 180
def extractDataFromQ(question):
    
    q = copy.deepcopy(question)
    q = str(q)
    q = q.split()
    q[-1] = q[-1].replace('?', '')
    
    return q[-1] #return q[-2]    


# we use it to save the columns each question is reffered to
def findQuestionIndex(theQ):
    
    q = copy.deepcopy(theQ)
    q = str(q)
    q = q.split()
    # print(q[1])
    try:
        r = header.index(q[1])
    except:
        print("question: ", q, " not in header list")
    return r
    
    
def partition(rows, question):
    """Partitions a dataset.

    For each row in the dataset, check if it matches the question. If
    so, add it to 'true rows', otherwise, add it to 'false rows'.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    
    # n_features = len(rows[0]) - 1  # number of columns
    n_features = len(rows[0]) - 2  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            question = Question(col, val)

            # try splitting the dataset
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


# we use it to find the error percentage and then the amount of say of each stump 
def findErrorPercent(T_rows, F_rows):
 
    errCount = 0
    totalError = 0
    for i in T_rows:
        #print(i)
        if not i[-1] == "yes":
            errCount +=1
         
    for i in F_rows:
        if i[-1] == "yes":
            errCount += 1
    totalError = errCount * initSampleweight
    return totalError


# we use it to find the amount of say of each stump
def getAmountOfSay(true_rows, false_rows):
    
    totalEr = findErrorPercent(true_rows, false_rows)
    if totalEr == 0:
        totalEr = 0.01
    elif totalEr == 1:
        totalEr = 0.99
    amountOfSay = 1/2 * math.log((1-totalEr) / totalEr, 10)     
    return amountOfSay


def myPred(rows):
    yes = 0
    no = 0
    for i in rows:
        if i[-1] == "yes":
            yes += 1
        else:
            no += 1
    
    if yes > no:
        return "yes"
    else:
        return "no"
    
    
# it changes the weight of the false and true predicts depending on their type 
def changeWeights(T_rows, F_rows, stamp_amount_of_say):
    
    coreectPredict = []
    falsePredict = []
    returned_Data = []
    for i in T_rows:
        # print(i[-2])
        if i[-1] == "yes":
            coreectPredict.append(i)
        else:
            falsePredict.append(i)
    for i in F_rows:
        if i[-1] == "no":
            coreectPredict.append(i)
        else:
            falsePredict.append(i)
    
    for i in falsePredict:
        
        i[-2] = round(i[-2]*math.exp(stamp_amount_of_say), 3)
        returned_Data.append(i)
    
    for i in coreectPredict:
        i[-2] =round(i[-2]*math.exp(stamp_amount_of_say*(-1)), 3)
        returned_Data.append(i)
    
    return returned_Data


# after we assign the new weights to the columns,we normalise them in order to 
# resample them
def normaliseWeights(Ndem_data):
    
    Norm_dem_data = copy.deepcopy(Ndem_data)
    summ = 0

    returned_data = copy.deepcopy([])
    for i in Norm_dem_data:
        summ += i[-2]

    for i in Norm_dem_data:

        row = copy.deepcopy(i)
        row[-2] = row[-2]/summ
        returned_data.append(row)

    sumDebug = 0
    for i in returned_data:
        sumDebug += i[-2]
    return returned_data


# We create a new dataset from the existing one,taking into account the weights
# We pick a random number from 0-1 and then subtract the weight of each column
# When the number is 0 or less we include this column to the new sample
# Repeat the process until the new data are the same size as the old ones
def resample(Rdem_Data):
    reborn_data = copy.deepcopy([])
    dem_Data = copy.deepcopy(Rdem_Data)
    for i in range(datasize):
        selection = round(random.randint(1, 98) * 0.01, 3) 
        index = 0
#         print()
#         print(selection)
        while selection > 0:
            selection = selection - dem_Data[index][-2]
#             print("in while:",selection)
            if selection <= 0:
                reborn_data.append(dem_Data[index])
            else:
                index += 1
    return  reborn_data  


# Get weights to the initial state(after data resample)
def weightResample(WDem_Data):
    Dem_Data = copy.deepcopy(WDem_Data)
    for i in Dem_Data:
        i[-2] = initSampleweight 
    return Dem_Data


# In[41]:


class Leaf2:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)
        
        
class Decision_Node2:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 questionIndex,
                 true_pred,
                 false_pred,
                 aos,
                 nextStump):
        self.question = question
        self.questionIndex = questionIndex
        self.true_pred = true_pred
        self.false_pred = false_pred
        self.aos = aos
        self.nextStump = nextStump

        
# build the tree
# it runs as many times as the times variable
def build_tree2(the_treeData, times):
    
    treeData = copy.deepcopy(the_treeData)

    if times <= 0:
        return None
    
    best_gain, question = find_best_split(treeData)
    
    if question == None:
        print("all the data are the same")
        return None
    
    questionIndex =  findQuestionIndex(question)
    true_rows, false_rows = partition(treeData, question)
    
    true_pred = myPred(true_rows)
    false_pred = myPred(false_rows)       
    aos = getAmountOfSay(true_rows, false_rows)

    changeWeightData = changeWeights(true_rows, false_rows, aos2)
    normalizedWeightsData = normaliseWeights(changeWeightData)
    
    new_Data = resample(normalizedWeightsData)
    freshData = weightResample(new_Data)
    
    times = times-1
    nextStump = build_tree2(freshData, times)

    return Decision_Node2(question,questionIndex, true_pred, false_pred, aos, nextStump)



def print_tree2(node, spacing=""):
    """World's most elegant tree printing function."""

#     # Base case: we've reached a leaf
#     if isinstance(node, Leaf):
#         print (spacing + "Predict", node.predictions)
#         return
    
    
    # Print the question at this node
    print(spacing + str(node.question))

    if isinstance(node, Decision_Node2):
        
        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        print(spacing + " |       " + "Predict: ", node.true_pred)
        #print_tree2(node.true_pred, spacing + " |       ")
    
        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        print(spacing + " |       " + "Predict: ", node.false_pred)
        #print_tree2(node.false_pred, spacing + " |      ")
        
        print (spacing + '--> Amount Of Say :')
        print(spacing + " |       " + "aos =  ", node.aos)
        
        print (spacing + '--> Q index :')
        print(spacing + " |       " + "qIndex =  ", node.questionIndex)
        
        if node.nextStump == None:
            return 
        else:
            print_tree2(node.nextStump, "|           ")

            
            
def prediction(forestNode, pred_data, finalAnsYes, finalAnsNo ):
    forest = copy.deepcopy(forestNode)
    # finalAnsYes = 0
    # finalAnsNo = 0
    # print("...")
    if isinstance(forestNode, Decision_Node2):
        if is_numeric(pred_data[forestNode.questionIndex]):
            if float(extractDataFromQ(forestNode.question)) <= float(pred_data[forestNode.questionIndex]):       
                finalAnsYes += forestNode.aos
            else:
                finalAnsNo += forestNode.aos
        else:
            if str(extractDataFromQ(forestNode.question)) == str(pred_data[forestNode.questionIndex]):
                finalAnsYes += forestNode.aos
            else:
                finalAnsNo += forestNode.aos
            
            
    if forestNode.nextStump == None:
        if finalAnsYes > finalAnsNo:
#             print("yes = ", finalAnsYes)
#             print("no = ", finalAnsNo)
#             print("yes")
            return "yes"
        else:
#             print("yes = ", finalAnsYes)
#             print("no = ", finalAnsNo)
#             print("no")
            return "no"
    else:  
        # print("yes = ", finalAnsYes)
        # print("no = ", finalAnsNo)
        return prediction(forest.nextStump, pred_data, finalAnsYes, finalAnsNo)


# In[42]:


# Toy dataset.
# Format: each row is an example.

# Column labels.
# These are used only to print the tree.
######## 

# header1 = ["ProductId", "color", "diameter", "sampleWeight", "label"]

# header3 = ["chestPain", "blocked_Arteris", "patient_Weight", "heartDisease"]

# header = header3

# training_data1 = [
#     ["1", 'Green', 3, '1'],
#     ["1", 'Yellow', 3, '1'],
#     ["2", 'Red', 1, '1'],
#     ["2", 'Red', 1, '0'],
#     ["1", 'Yellow', 3, '0'],
#     ["3", 'Yellow', 3, '0']
# ]


# training_data2 = [
#     ["1", 'Green', 3, 'Apple'],
#     ["1", 'Yellow', 3, 'Apple'],
#     ["1", 'Red', 1, 'Grape'],
#     ["2", 'Red', 1, 'Grape'],
#     ["3", 'Yellow', 3, 'Lemon'],
#     ["3", 'Yellow', 3, 'Grape']
# ]

# training_data3 = [
#     ['yes', 'yes', 205, 'yes'],
#     ['no', 'yes', 180, 'yes'],
#     ['yes', 'no', 210, 'yes'],
#     ['yes', 'yes', 167, 'yes'],
#     ['no', 'yes', 156, 'no'],
#     ['no', 'yes', 125, 'no'],
#     ['yes', 'no', 168, 'no'],
#     ['yes', 'yes', 172, 'no']
# ]


# training_data4 = [['yes', 'no', 210, 0.125, 'yes'],
#      ['no', 'yes', 125, 0.125, 'no'], 
#      ['yes', 'no', 168, 0.125, 'no'], 
#      ['no', 'yes', 156, 0.125, 'no'], 
#      ['no', 'yes', 125, 0.125, 'no'], 
#      ['no', 'yes', 156, 0.125, 'no'], 
#      ['no', 'yes', 156, 0.125, 'no'], 
#      ['no', 'yes', 180, 0.125, 'yes']]

# training_data5 = [
#     ['yes', 'yes', 205, 'yes'],
#     ['no', 'yes', 180, 'yes'],
#     ['yes', 'no', 210, 'yes'],
#     ['yes', 'yes', 167, 'yes'],
#     ['no', 'yes', 156, 'no'],
#     ['no', 'no', 125, 'no'],
#     ['yes', 'no', 168, 'no'],
#     ['yes', 'yes', 172, 'no']
# ]

# training_data = training_data5


#this function returns false if what was sent wasnt a number else it returns the number as float
#for some reason it does not work for 0
def is_number(s):
    try:
        return float(s)
    except ValueError:
        return False

#if the decision column has 1 or 0 we change that to Yes/No   
with open('heart.csv', 'r') as r, open('heartYESNO.csv', 'w') as w:
    data = r.readlines()
    w.write(data[0])
    check = data[1].rstrip('\n').split(',')[-1]
    if type(is_number(check)) == float:
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

header = []
for index,i in enumerate(line2):
    header.append(i)
# print(header)


training_data = []
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
    training_data.append(temp)
    line = f.readline()
# print("train Data= ", training_data)


dataTEST = []
f = open('heartTEST.txt', 'r')
line = f.readline()
line = line.rstrip('\n')  # remove end of line char
line2 = line.split(",")
# dataTEST.append(line2)
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



os.remove("heartTRAIN.txt")
os.remove("heartTEST.txt")
os.remove("heartYESNO.csv")


# In[43]:


# We use global init sample weight because after each resample the weights are 
# always the same
datasize = len(training_data)
initSampleweight = 1/len(training_data)
initSampleweight = round(initSampleweight, 3)

# We import the weights into our data in initial state, in position -2
def initializeTheWeights(data):
    for i in training_data:
        i.insert(-1, initSampleweight)
#         print(i)
        
initializeTheWeights(training_data)

best_gain, question = find_best_split(training_data)
true_rows, false_rows = partition(training_data, question)

#error =  findErrorPercent(true_rows, false_rows)
aos2 = getAmountOfSay(true_rows, false_rows)
# print("aos2 = ", aos2)

# print("old data = ", training_data)
# print()
new_Data = changeWeights(true_rows, false_rows, aos2)
# print("new data = ", new_Data)
new_Data = normaliseWeights(new_Data)
data = copy.deepcopy(new_Data)
resample(data)


# In[58]:


# input: csv file, and max N you want to create 
# output: AdaBoost.M1.
my_tree2 = build_tree2(training_data, 15)
print_tree2(my_tree2)


# In[61]:


# input: test case
# output:  file with predicts and stats

f = open('results.txt', 'w')
correct = false = 0
for i in dataTEST:
    result = prediction(my_tree2, i, 0, 0)
    f.write(result+'\n')
    if i[-1].lower() == result.lower():
        correct +=1
    else: false +=1

print("Correct predictions:", correct)
print("False predictions:", false)
f.close()
