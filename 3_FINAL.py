import nltk.translate.ibm1 as i1
import nltk.translate.ibm2 as i2
import nltk.translate.phrase_based as pb
import pprint
import json
import time
from nltk.translate import AlignedSent
starttime= time.time()
A_text = []
##load the json file
with open('own.json') as f:
	loaded_data = json.load(f)

english=[]
German=[]
for i in range(len(loaded_data)):
	laoded_dictionary_i={}
	laoded_dictionary_i = loaded_data[i]
	l_1 = laoded_dictionary_i['en']  ## all english
	l_2 = laoded_dictionary_i['gn']  ## all german
	english.append(l_1)
	German.append(l_2)
	l_3=[]
	l_3 = l_1.split()
	l_4=[]
	l_4=l_2.split()	
	A_text.append(AlignedSent(l_4,l_3))
ib1 = i1.IBMModel1(A_text,5)

l_align=[]
for i in A_text:
	temp1= i.alignment
	temp2=[]
	for x in range(len(temp1)):
		temp2.append(temp1[x][0])
	l_align.append(temp2)
	
x=0
laoded_dictionary_it={}
myDict={}
for i in range(len(english)):
	phrases = pb.phrase_extraction(english[i],German[i],l_align[i])
	for j in sorted(phrases):
		k = (j[2],j[3],len(j[2].split()))
		l= j[3]
		if k in laoded_dictionary_it.keys():
			laoded_dictionary_it[k]=laoded_dictionary_it[k]+1
		else:
			laoded_dictionary_it[k]=1
			x=x+1
		if l in myDict.keys():
			myDict[l] = myDict[l]+1
		else:
			myDict[l]=1

for i in laoded_dictionary_it.keys():
	laoded_dictionary_it[i]/=myDict[i[1]]

l_translate=[]
l_length=[]
l_probability=[]
for i in laoded_dictionary_it.keys():
	l_translate.append((i[0],i[1]))
	l_length.append(i[2])
	l_probability.append(laoded_dictionary_it[i])


for i in range(len(l_translate)):
	for j in range(1,len(l_translate)):
		if l_probability[j-1] < l_probability[j]:
			temp = l_probability[j-1]
			l_probability[j-1] = l_probability[j]
			l_probability[j] = temp
			temp = l_length[j-1]
			l_length[j-1] = l_length[j]
			l_length[j] = temp
			temp = l_translate[j-1]
			l_translate[j-1] = l_translate[j]
			l_translate[j] = temp
		elif l_probability[j-1] == l_probability[j]  :
			if l_length[j-1] < l_length[j]:
				temp = l_length[j-1]
				l_length[j-1] = l_length[j]
				l_length[j] = temp
				temp = l_translate[j-1]
				l_translate[j-1] = l_translate[j]
				l_translate[j] = temp
		else:
			pass

for i in range(len(l_translate)):
	if l_length[i]<=4:
		print(l_translate[i],l_length[i],l_probability[i])

endtime=time.time()
