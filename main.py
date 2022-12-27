## STop_word porar jonno
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
no_doc_each_xml=70
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
    zero_ind=[]    
    for p in range(len(c)):
        if len(c[p])==0:
            zero_ind.append(p)
    #        
    for y in range(len(zero_ind)):
        #np.delete(c,np.asarray(zero_ind))
        c.pop(zero_ind[y]-y)
    
    return c,zero_ind

c=Preprocessing2(D)[0]
zero_index=  Preprocessing2(D)[1]  
##########FOR eUCLIDEAN DISTANCE####################
all_eu_dist=[]

def find_euclidean(v1_inst,v2_test):
#    v1_inst={'play':2,'cricket':1,'football':1}
#    v2_test={'want':1,'play':1,'music':1}
    
    v3=list(list(v1_inst.keys())+list(v2_test.keys()))
    v4=list(Counter(v3).keys())
    a=np.zeros(len(v4))
    b=np.zeros(len(v4))
    for i in range(len(v2_test)):
        for j in range(len(v4)):
            if (v4[j]==list(v2_test.keys())[i]):
                a[j]=list(v2_test.values())[i]
                
    
    for i in range(len(v1_inst)):
        for j in range(len(v4)):
            if (v4[j]==list(v1_inst.keys())[i]):
                b[j]=list(v1_inst.values())[i]
                
    eu_dis=np.linalg.norm(a-b)
    return eu_dis
#
#
#
#for y in range(len(c)):
#    if y==5:
#        continue
#    all_eu_dist.append(find_euclidean(dict(c[y]),dict(c[5])))
#    
#m=np.argmin(all_eu_dist)    
#printer=[i for i in range(0,200)]
#Anime=[i for i in range(200,400)]
#Arduino=[i for i in range(400,600)]
#documents_name=['printer','Anime','Arduino']
#
##
#if printer.count(m)==1:
#    print('Test document belongs to 3d_printer')
#elif Anime.count(m)==1:
#    print('Test document belongs to Anime')
#
#elif Arduino.count(m)==1:
#    print('Test document belongs to Arduino')
    
        
def doc_counts(c,keys):
    count=0
    for i in range(len(c)):
        if(c[i][keys]>0):
            count=count+1
    return count
#v1={'play':2,'cricket':1,'football':1}
#v2={'want':1,'play':1,'music':1}
#v=[v1 , v2]
def TF_IDF(c):
    t=[]
    for r in range(len(c)):
        
        
        #c1=Counter({'play':2,'cricket':1,'football':1})
        
        #c2=Counter({'want':1,'play':1,'music':1})
        tf_idf=[]
        #cc=[c1,c2]
        for i in range(len(c[r])):
            tf_idf.append((list(c[r].values())[i]/np.sum(np.asarray(list(c[r].values()))))*np.log10(len(c)/(1+doc_counts(c,list(c[r].keys())[i]))))
            
        t.append(Counter(dict(zip(list(c[r].keys()),tf_idf))))
    return t        

def find_cos_similarity(v1_inst,v2_test):
    v3=list(list(v1_inst.keys())+list(v2_test.keys()))
    v4=list(Counter(v3).keys())
    a=np.zeros(len(v4))
    b=np.zeros(len(v4))
    for i in range(len(v2_test)):
        for j in range(len(v4)):
            if (v4[j]==list(v2_test.keys())[i]):
                a[j]=list(v2_test.values())[i]
                
    
    for i in range(len(v1_inst)):
        for j in range(len(v4)):
            if (v4[j]==list(v1_inst.keys())[i]):
                b[j]=list(v1_inst.values())[i]
                
    Cos=(np.dot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b))
    return Cos
test_no=15
n_cls=6
#t=TF_IDF(c)
path_test='F:\\Python\\Test\\Test'
test_list=dir_file_name(path_test)
T=preprocessing1(test_no,test_list)
c_test=Preprocessing2(T)[0]
t_test=TF_IDF(c_test)
#t_test=c_test ###for euclidean
def Testing(t,test):
    
    all_cos=[]
    for y in range(len(t)):
#        if y==560:
#            continue
        all_cos.append(find_cos_similarity(dict(t[y]),dict(test)))
    return all_cos
K=7
def result(all_cos ):
    
    match=np.argsort(all_cos)[::-1][:K]  ##### TFIDF
    #match=np.argsort(all_cos)[:K]   #euclidean
    vect=[[i for i in zero_index if i<no_doc_each_xml],[i for i in zero_index if (i>=no_doc_each_xml and i<no_doc_each_xml*2)],
          [i for i in zero_index if (i>=no_doc_each_xml*2 and i<no_doc_each_xml*3)],[i for i in zero_index if (i>=no_doc_each_xml*3 and i<no_doc_each_xml*4)],
          [i for i in zero_index if (i>=no_doc_each_xml*4 and i<no_doc_each_xml*5)],[i for i in zero_index if (i>=no_doc_each_xml*5 and i<no_doc_each_xml*6)],
          [i for i in zero_index if (i>=no_doc_each_xml*6 and i<no_doc_each_xml*7)],[i for i in zero_index if (i>=no_doc_each_xml*7 and i<no_doc_each_xml*8)],
          [i for i in zero_index if (i>=no_doc_each_xml*8 and i<no_doc_each_xml*9)],[i for i in zero_index if (i>=no_doc_each_xml*9 and i<no_doc_each_xml*10)],
          [i for i in zero_index if (i>=no_doc_each_xml*10 and i<no_doc_each_xml*11)],[i for i in zero_index if i>=no_doc_each_xml*11]]
    
    att=[]
    for i in range(len(vect)):
        att.append([filename_train[i]]*(no_doc_each_xml-len(vect[i])))
        
    xml = [item for sublist in att for item in sublist]
    vote=[]
    for i in range(len(match)):
        vote.append(xml[match[i]])
    
    aa=[k for k,v in Counter(vote).items() if v>1]
    if len(aa)==0:
        predict=vote[0]
    else:
        
        predict=aa[0]
    #predict=((max(k for k,v in Counter(vote).items() if v>1)))
    return predict

predicted=[]
for x in range(len(t_test)):
    
    all_cos=Testing(c,t_test[x])
    predicted.append(result(all_cos))
#######prediction
z=Preprocessing2(T)[1]
#vect=[[i for i in z if i<test_no],[i for i in z if (i>=test_no and i<test_no*2)],[i for i in z if i>=test_no*2]]
vect=[[i for i in z if i<test_no],[i for i in z if (i>=test_no and i<test_no*2)],
      [i for i in z if (i>=test_no*2 and i<test_no*3)],[i for i in z if (i>=test_no*3 and i<test_no*4)],
      [i for i in z if (i>=test_no*4 and i<test_no*5)],[i for i in z if (i>=test_no*5 and i<test_no*6)],
      [i for i in z if (i>=test_no*6 and i<test_no*7)],[i for i in z if (i>=test_no*7 and i<test_no*8)],
      [i for i in z if (i>=test_no*8 and i<test_no*9)],[i for i in z if (i>=test_no*9 and i<test_no*10)],
      [i for i in z if (i>=test_no*10 and i<test_no*11)],[i for i in z if i>=test_no*11]]
att=[]
for i in range(len(vect)):
    att.append([test_list[i]]*(test_no-len(vect[i])))

act = [item for sublist in att for item in sublist]

def accuracy(x,y):
    count=0
    for i, j in zip(x, y):
        if i==j:
            count=count+1
    acc=(count/len(x))*100
    return acc

acc=accuracy(act, predicted)

print("accuracy is : %s for the K=%s " % (acc, K))
print("--- %s seconds ---" % (time.time() - start_time))
#pred=result(all_cos)
    