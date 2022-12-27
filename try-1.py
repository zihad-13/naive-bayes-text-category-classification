## STop_word porar jonno
import numpy as np
from string import digits
import pandas as pd
from collections import Counter
import time
start_time = time.time()
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
dic_rep = {'<p>':'','</p>':'','<a>':'','</a>':'','href':'','="https://':'','"http://':'','<em>':'','</em>':'','<strong>':'','</strong>':'','<a':'','<h1>':'','</h1>':'','=':'','?':'',',':'','(':'',')':'','"':'',';':'','/':' ','>':' '}
 
def replace_all(text, dic_rep):
    for i, j in dic_rep.items():
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
            new_D.append(replace_all(D[i], dic_rep))
        
        final_dictionary.append(new_D)
     
    
    flat_list = [item for sublist in final_dictionary for item in sublist]
    return flat_list
def filter_list(main,exclude):
    return [x for x in main if x not in exclude]
no_doc_each_xml=200
D=preprocessing1(no_doc_each_xml,filename_train)





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
    
#    
#
#
data=[]
for i in range(len(c)):
    data.append(list(c[i].keys()))
flat_list = [item for sublist in data for item in sublist]
fil_count=Counter(flat_list)
dictionary=list(fil_count.keys())
#wrd_freq=list(fil_count.values())
#dict_value=list(zip(dictionary, wrd_freq))








######hamming
#dic=['play','cricket','football','music','like','singing','very','small','insect','want']
#d1=['like','singing']
dic=dictionary
def dataframe(c,dic):
    
    df=pd.DataFrame()
    for x in range(len(c)):
        d1=list(c[x].keys())
        a=np.zeros(len(dic))
        
        for i in range(len(dic)):
            for j in range(len(d1)):
                if d1[j]==dic[i]:
                    a[i] = 1
        
        df[x] = a
    return df        
            
#col=['3D_printer']*200 + ['Anime']*200 + ['Arduino']*200
#df.columns=col
df_tr=dataframe(c,dic)
df_train=df_tr.as_matrix()
test_no=80
path_test='F:\\Python\\Test\\Test'
test_list=dir_file_name(path_test)
T=preprocessing1(test_no,test_list)
c_test=Preprocessing2(T)[0]

df_tst=dataframe(c_test,dic)
df_test=df_tst.as_matrix()
#g=[]

def Testing(df_train,df_test):
    g=[]
    for y in range(df_train.shape[1]):
    #    if y==408:
    #        continue
        h=np.logical_xor(np.array(df_train[:, y]),np.array(df_test))
        h=h*1
        g.append(np.sum(h))
    return g
K=5    
def result(all_cos):
    
    #match=np.argsort(all_cos)[::-1][:5]  ##### TFIDF
    match=np.argsort(all_cos)[:K]   #euclidean
    #vect=[[i for i in zero_index if i<no_doc_each_xml],[i for i in zero_index if (i>=no_doc_each_xml and i<no_doc_each_xml*2)],[i for i in zero_index if i>=no_doc_each_xml*2]]
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
    #predict=((max(k for k,v in Counter(vote).items() if v>1)))
    aa=[k for k,v in Counter(vote).items() if v>1]
    if len(aa)==0:
        predict=vote[1]
    else:
        
        predict=aa[0]
    return predict
#g=np.asarray(g)
#index=np.argsort(g)[:3]
predicted=[]
for x in range(df_test.shape[1]):
    
    all_cos=Testing(df_train,df_test[:, x])
    predicted.append(result(all_cos))

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
    
#g=np.logical_xor(ndf[:, 1],ndf[:, 17])            
############Eucledian er vectore bananor jonno
#import numpy as np
#from collections import Counter
#v1={'play':2,'cricket':1,'football':1}
#v2={'want':1,'play':1,'music':1}
#
#v3=list(list(v1.keys())+list(v2.keys()))
#v4=list(Counter(v3).keys())
#a=np.zeros(len(v4))
#for i in range(len(v2)):
#    for j in range(len(v4)):
#        if v4[j]==list(v2.keys())[i]:
#            a[j]=list(v2.values())[i]


#for p in range(len(c)):
#    if len(c[p])==0:
#        print(p)
        