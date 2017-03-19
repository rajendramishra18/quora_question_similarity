

list_pred = []
for i in range(24):
	path = "test_X_"+str(i)
	test_X = cp.load(open(path,'r'))
	pred = clf.predict_proba(test_X)
	path_id = "id"+str(i)
	idq = cp.load(open(path_id,'r'))
	if i==0:
		j=1
	else:
		j=0
	while j<len(test_X):
		list_pred.append([idq[j],pred[j][1]])
		j+=1
 
with open("prediction_2.csv",'w') as predfile:
	predwriter = csv.writer(predfile,delimiter=',')
	predwriter.writerow(['test_id','is_duplicate'])
	for j in range(0,len(list_pred)):
		predwriter.writerow(list_pred[j])
	print(len(list_pred))
 
