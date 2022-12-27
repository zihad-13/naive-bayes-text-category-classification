import numpy as np
from string import digits
from collections import Counter
import time
start_time = time.time()
#import pandas as pd
def exclude():
    
    f=open('Stopwords.txt','r')
    d=[]
    dd=[]
    for i in f:
        d.append(i)
    
    for i in range(len(d)):
        dd.append(d[i].replace('\n',''))
    return dd
# XML porar jonno
path ='F:\\Python\\Training\\Training'
def dir_file_name(path):
    import os
    dirs = list(os.listdir( path ))
    dirs.remove('stop_read.py')
    dirs.remove('Stopwords.txt')
    dirs.remove('ass_3.py')
    dirs.remove('ass_3_knn_eucld.py')
    dirs.remove('ass3_test.py')
    dirs.remove('Naive_bayes.py')
    return dirs

filename_train= dir_file_name(path)
dic = {'<p>':'','</p>':'','<a>':'','</a>':'','href':'','="https://':'','"http://':'','<em>':'','</em>':'','<strong>':'','</strong>':'','<a':'','<h1>':'','</h1>':'','=':'','?':'',',':'','(':'',')':'','"':'',';':'','/':' ','>':' '}
 
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
def NO_digit(s):
    
    #s = 'abc123def456ghi789zero0'
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    return res

def preprocessing1(n,list_of_filename):
    final_dictionary=[]
    for k in range(len(list_of_filename)):
            
        import xml.etree.ElementTree as ET
        tree = ET.parse(list_of_filename[k])  
        root = tree.getroot()
        D=[]
        for i in range(n):
            
            dict=root[i].attrib
            D.append(dict['Body'])
        
        
        
        new_D=[]
        for i in range(len(D)):
            new_D.append(replace_all(D[i], dic))
        
        final_dictionary.append(new_D)
     
    
    flat_list = [item for sublist in final_dictionary for item in sublist]
    return flat_list
def filter_list(main,exclude):
    return [x for x in main if x not in exclude]
no_doc_each_xml=150
D=preprocessing1(no_doc_each_xml,filename_train)



#zero_ind=[]
#D=['I play cricket. I play football.','Play this music','I like singing','Cricket is very small insect']
def Preprocessing2(D):
    
    doc=[]
    c=[]
    
    
    for i in range(len(D)):
        D[i]=D[i].replace('.','')
        D[i]=D[i].replace('-',' ')
        D[i]=D[i].replace('\n','')
        
        doc.append((NO_digit(D[i]).lower()).split(' '))
    doc=np.asarray(doc)
    
    exclud=exclude()
    DD=[]
    for k in range(len(doc)):
        DD.append(filter_list(doc[k],exclud))
        
    for j in range(len(DD)):
        c.append(Counter(DD[j]))
    
    #c=np.asarray(c)
#    zero_ind=[]    
#    for p in range(len(c)):
#        if len(c[p])==0:
#            zero_ind.append(p)
#    #        
#    for y in range(len(zero_ind)):
#        #np.delete(c,np.asarray(zero_ind))
#        c.pop(zero_ind[y]-y)
#    
#    return c,zero_ind
    return c

c=Preprocessing2(D)
def topic_word_count(c):
    
    data=[]
    for i in range(len(c)):
        data.append(list(c[i].keys()))
    flat_list = [item for sublist in data for item in sublist]
    fil_count=Counter(flat_list)
    dictionary=list(fil_count.keys())
    wrd_freq=list(fil_count.values())
    dict_value=list(zip(dictionary, wrd_freq))
    return dict_value

def dic_count(c):
    data=[]
    for i in range(len(c)):
        data.append(list(c[i].keys()))
    flat_list = [item for sublist in data for item in sublist]
    
    
    #exclude=['i','is','very','this','that','are']
#    exclude=exclude()
#    filtered=filter_list(flat_list,exclude)
    fil_count=Counter(flat_list)
    dictionary=list(fil_count.keys())
    return dictionary,len(dictionary)

#C=topic_word_count(c[0:300])
n_class=12
test_no=25
path_test='F:\\Python\\Test\\Test'
test_list=dir_file_name(path_test)
T=preprocessing1(test_no,test_list)
c_test=Preprocessing2(T)
#alpha=1
alphas=np.linspace(0.1,30,50)
a,diclen=dic_count(c)

    
def Prob_test_doc(c_test,C,alpha):
    P=[]
    #N=[]
    
    for i in range(int(len(c_test))):
        N=[]
        for j in range(len(c_test[i])):
            for k in range(len(C)):
                
                if list(c_test[i].keys())[j] == C[k][0]:
                    N.append(C[k][1])
    #                Pw=(((N)+alpha)/((np.sum(np.asarray(list(dict(C).values()))))+alpha*len(C)))
                    break
                elif k==len(C)-1:
                    N.append(0)
    #                Pw=(((N)+alpha)/((np.sum(np.asarray(list(dict(C).values()))))+alpha*len(C)))
        
        Pw=((np.asarray(N)+alpha)/((np.sum(np.asarray(list(dict(C).values()))))+alpha*diclen))
        P.append(np.prod(Pw)*(1/n_class))

    return P

def accuracy(x,y):
    count=0
    for i, j in zip(x, y):
        if i==j:
            count=count+1
    acc=(count/len(x))*100
    return acc

            

acc=[]
for a in alphas:
    alpha=a
    all_prob=[]            
    for n in range(n_class):
        all_prob.append(Prob_test_doc(c_test,topic_word_count(c[no_doc_each_xml*n : no_doc_each_xml*(n+1)]),alpha))
            
    
    pred=[]
    for i in range(len(all_prob[0])):
        val=[]
        for j in range(len(all_prob)):
            val.append(all_prob[j][i])
        pred.append(np.argsort(val)[::-1][:1])
    
    out = np.concatenate(pred).ravel() 
    actual=[]
    for i in range(len(test_list)):
        actual.append([i]*test_no)
    act_label = [item for sublist in actual for item in sublist]
    acc.append(accuracy(act_label, out))
    print("accuracy is : %s for the alpha=%s \n " % (acc, alpha))

#acc=accuracy(act_label, out)
print("--- %s seconds ---" % (time.time() - start_time))     
        
