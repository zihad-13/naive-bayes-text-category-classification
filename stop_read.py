## STop_word porar jonno
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

list_of_filename= dir_file_name(path)
dic = {'<p>':'','</p>':'','<a>':'','</a>':'','href':'','="https://':'','"http://':'','<em>':'','</em>':'','<strong>':'','</strong>':'','<a':'','<h1>':'','</h1>':'','=':'','?':'',',':''}
 
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def preprocessing1():
    final_dictionary=[]
    for k in range(len(list_of_filename)):
            
        import xml.etree.ElementTree as ET
        tree = ET.parse(list_of_filename[k])  
        root = tree.getroot()
        D=[]
        for i in range(200):
            
            dict=root[i].attrib
            D.append(dict['Body'])
        
        
        
        new_D=[]
        for i in range(len(D)):
            new_D.append(replace_all(D[i], dic))
        
        final_dictionary.append(new_D)
     
    
    flat_list = [item for sublist in final_dictionary for item in sublist]
    return flat_list

D=preprocessing1()


from collections import Counter
#D=['I play cricket. I play football.','Play this music','I like singing','Cricket is very small insect']
doc=[]
c=[]
data=[]
DD=[]
for i in range(len(D)):
    D[i]=D[i].replace('.','')
    D[i]=D[i].replace('\n','')
    DD.append(D[i].lower())
    doc.append((D[i].lower()).split(' '))
    
for j in range(len(doc)):
    c.append(Counter(doc[j]))
for i in range(len(c)):
    data.append(list(c[i].keys()))
flat_list = [item for sublist in data for item in sublist]

def filter_list(main,exclude):
    return [x for x in main if x not in exclude]
#exclude=['i','is','very','this','that','are']
exclude=exclude()
filtered=filter_list(flat_list,exclude)
fil_count=Counter(filtered)
dictionary=list(fil_count.keys())
wrd_freq=list(fil_count.values())
dict_value=list(zip(dictionary, wrd_freq))
