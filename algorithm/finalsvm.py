from numpy import *
import random
import csv


######Preprocessing##########################################################################################################################################
def PCA(dataMat,n):                          #use PCA to reduce columns
    newData = dataMat
    covMat = cov(newData,rowvar=0)
    eigenVals,eigenVects=linalg.eig(mat(covMat))
    print("eig finished")
    eigenValIndex = argsort(eigenVals)

    n_eigenValIndex = eigenValIndex[-n:]
    n_eigenVect=eigenVects[:,n_eigenValIndex]

    lowDataMat = newData*n_eigenVect
    lowDataMatreal=lowDataMat.real
    print(lowDataMatreal)
    print("PCA finished")
    
    return lowDataMatreal


def std_rowdata(dataMat,appnameMat,n):       #select the samples with top high standard deviation
    column_std_var = std(dataMat,axis=1)
    index_sort = argsort(column_std_var.T)

    index_selected = index_sort[0,-n:]
    newData = dataMat[index_selected,:][0,:]
    new_appnameMat=appnameMat[index_selected,:][0,:]
    print("std_rowdata finished")
    return newData,new_appnameMat


######################################################################################################################################################

####LoadData##########################################################################################################################################

def translabel(appnamemat,lablefilename,labelname):  #transform labels into either 1 or -1

    m,n=shape(appnamemat)
    label=[]
    #print(appnamemat)
    for i in range(m):
        appname=appnamemat[i,0]
        #print(appname)
        labelFile = open(lablefilename,'r')
        labelInfo = csv.reader(labelFile)
        for linfo in labelInfo:
            #linelabel=[]
            if linfo[0] == appname:
                #print(appname)
                if linfo[1] == labelname:
                    label.append([1])
                else:
                    label.append([-1])
                break
        #label.append(linelabel)


    finallabelMat=mat(label)
    return finallabelMat




def loadtestdata(testfilename):                     #load dataset
    testFile = open(testfilename,'r')
    testInfo = csv.reader(testFile)
    testdata=[]
    testappname=[]
    for tinfo in testInfo:
        linetestdata=[]
        linetestappname=[]        
        n=0
        for i in tinfo:
            if n ==0:
                appname=tinfo[n]
                linetestappname.append(appname)
            else:
                linetestdata.append(float(tinfo[n]))
            n += 1
        testdata.append(linetestdata)
        testappname.append(linetestappname)
    return mat(testdata),mat(testappname)


def final(trainingfilename,testfilename,sample_num,column_num):                    #finally get the data matrix we need
    traingdataMat_all,appnameMat_all=loadtestdata(trainingfilename)
    new_trainingdataMat,new_appnameMat=std_rowdata(traingdataMat_all,appnameMat_all,sample_num)
    traingdataMat_all=mat([])
    #appnameMat_all=mat([])
    testdataMat_all,testappname_all=loadtestdata(testfilename)
    m1,n1=shape(testdataMat_all)
    dataMat_all=concatenate((new_trainingdataMat,testdataMat_all))
    new_trainingdataMat=mat([])
    testdataMat_all=mat([])

    lowdataMat=PCA(dataMat_all,column_num)
    dataMat=mat([])
    
    finaltrainingdataMat=lowdataMat[:-m1,:]
    finaltestdataMat=lowdataMat[-m1:,:]
    print("finalPCA finished")
    return finaltrainingdataMat,new_appnameMat,finaltestdataMat,testappname_all



#########################################################################################################################################################

#####SMO######################################################################################################################################

def kernelTrans(X,A,kTup):               #define Gaussian kernels
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'rbf':
        for j in range(m):
            deltaRow=X[j,:] -A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))

    return K


class optstruct:                         # define class
    def __init__(self,dataMat,labelMat,C,toler,kTup):
        self.dataMat =dataMat
        self.labelMat = labelMat
        self.C = C
        self.toler =toler
        self.m = shape(dataMat)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b =0
        self.eCache = mat(zeros((self.m,2)))
        self.K=mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.dataMat,self.dataMat[i,:],kTup)




def calcEk(os,k):                       #calculate the error value of point k
	fXk = float(multiply(os.alphas,os.labelMat).T*os.K[:,k]+os.b)
	Ek = fXk - float(os.labelMat[k])
	return Ek

def selectJ(i,os,Ei):                  #after choosing alpha i, choosing alpha j and make sure there's a maximum step size
	maxK = -1
	maxDeltaE = 0
	Ej = 0
	os.eCache[i] = [1,Ei]
	validEcacheList = nonzero(os.eCache[:,0].A)[0]
	if (len(validEcacheList)>1):
		for k in validEcacheList:
			if k == i:
				continue
			Ek = calcEk(os,k)
			deltaE = abs(Ei-Ek)
			if (deltaE > maxDeltaE):
				maxK =k
				maxDeltaE = deltaE
				Ej =Ek
		return maxK,Ej
	else:
		j= selectJrand(i,os.m)
		Ej = calcEk(os,j)
	return j,Ej
    

def selectJrand(i,a):               #randomly select alpha j
    j=i
    while (j==i):
        j = int(random.uniform(0,a))
    return j

def updateEk(os,k):                #update the error value
	Ek = calcEk(os,k)
	os.eCache[k] = [1,Ek]

def clipAlpha(aj,Highbound,Lowbound):             #choose the maximum one among aj, Highbound and Lowbound
    if aj>Highbound:
        aj=Highbound
    if Lowbound>aj:
        aj=Lowbound
    return aj


def innerL(i,os):                                 #after alpha i is chosen, choose alpha j to get the maximum step size
	Ei = calcEk(os,i)
	if ((os.labelMat[i]*Ei < -os.toler) and (os.alphas[i] < os.C)) or ((os.labelMat[i]*Ei > os.toler) and (os.alphas[i]>0)):
		j,Ej = selectJ(i,os,Ei)
		alphaIold=os.alphas[i].copy()
		alphaJold=os.alphas[j].copy()
		if (os.labelMat[i] != os.labelMat[j]):
			L=max(0,os.alphas[j]-os.alphas[i])
			H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
		else:
			L=max(0,os.alphas[j]+os.alphas[i]-os.C)
			H=min(os.C,os.alphas[j]+os.alphas[i])
		if L==H:

		    return 0
		eta = -1*(2*os.K[i,j]-os.K[i,i]-os.K[j,j])
		if eta <= 0:

		    return 0
		os.alphas[j] += os.labelMat[j]*(Ei-Ej)/eta
		os.alphas[j] = clipAlpha(os.alphas[j],H,L)
		updateEk(os,j)
		if (abs(os.alphas[j]-alphaJold)<0.00001):

		    return 0
		os.alphas[i] += os.labelMat[j]*os.labelMat[i]*(alphaJold-os.alphas[j])
		updateEk(os,i)
		b1=os.b-Ei-os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,i]-os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[i,j]
		b2=os.b-Ej-os.labelMat[i]*(os.alphas[i]-alphaIold)*os.K[i,j]-os.labelMat[j]*(os.alphas[j]-alphaJold)*os.K[j,j]
		if (0<os.alphas[i]) and (os.C>os.alphas[i]):
			os.b=b1
		elif (0<os.alphas[j]) and (os.C>os.alphas[j]):
			os.b=b2
		else:
			os.b=(b1+b2)/2
		return 1
	else:
		return 0



def smoP(dataMat,labelMat,C,toler,maxIter,kTup):                       #define an outer loop to change the alpha i and to output constant b and array alphas
    os =optstruct(dataMat,labelMat,C,toler,kTup)
    iter=0
    entireSet = True
    PairsChanged = 0
    while ((iter < maxIter) and (PairsChanged>0) or (entireSet)):
    	PairsChanged = 0
    	if entireSet:
    		for i in range(os.m):
    			PairsChanged += innerL(i,os)

    		iter += 1
    	else:
    		nonBoundIs = nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]
    		for i in nonBoundIs:
    			PairsChanged += innerL(i,os)

    		iter += 1
    	if entireSet:
    		entireSet = False
    	elif (PairsChanged == 0):
    		entireSet = True

    return os.b,os.alphas
############################################################################################################################################

#####Classify###############################################################################################################################


def getlabels(lablefilename):                              #get the list of labels in training dataset and the length of it
    labelslist=[]
    labelFile = open(lablefilename,'r')
    labelInfo = csv.reader(labelFile)
    for linfo in labelInfo:
        if linfo[1] not in labelslist:
            labelslist.append(linfo[1])
    n=len(labelslist)
    return labelslist,n



def givelabel(w,b,x):                                #Judge whether this point is in the class label i
    y=x*w+b
    if y>0:
        return 1
    else:
        return -1


def classify(datafilename,lablefilename,testfilename,C,toler,maxIter,rows,columns,kTup):      #classify the testing dataset and output the result
    labelslist,n=getlabels(lablefilename)
    trainingdataMat,trainingappnameMat,testdataMat,testappnameMat=final(datafilename,testfilename,rows,columns)
    
    trainm,trainn= shape(trainingdataMat)
    trappm,trappn= shape(trainingappnameMat)

    with open('trainingdataMat.csv', 'w',newline='') as File:
        abcsv = csv.writer(File, dialect='excel')
        
        for i in range(trainm):
            writein=[]
            for j in range(trainn):
            	writein.append(trainingdataMat[i,j])
            abcsv.writerow(writein)

    with open('trainingappnameMat.csv', 'w',newline='') as File2:
        abcsv2 = csv.writer(File2, dialect='excel')
        
        for i in range(trappm):
            writein2=[]
            for j in range(trappn):
            	writein2.append(trainingappnameMat[i,j])
            abcsv2.writerow(writein2)


    m2,n2 = shape(testdataMat)
    classifiedlabel=mat(zeros((m2,2)))
    classifiedlabel=classifiedlabel.astype(str)
    for k in range(n):

        labelname=labelslist[k]

        labelMat=translabel(trainingappnameMat,lablefilename,labelname)

        b,alphas = smoP(trainingdataMat,labelMat,C,toler,maxIter,kTup)

        m,n = shape(trainingdataMat)
        w = mat(zeros((n,1)))
        for i in range(m):
            w += multiply(alphas[i]*labelMat[i],trainingdataMat[i,:].T)
        alreadyselec=[]



        for j in range(m2):

            if j not in alreadyselec:
                if givelabel(w,b,testdataMat[j,:])==1:
                    alreadyselec.append(j)
                    appname=testappnameMat[j,0]
                    classifiedlabel[j,0]=appname
                    classifiedlabel[j,1]=labelname

        print("label "+str(k)+" finished")
    print(classifiedlabel)
    with open('predicted_labels.csv', 'w',newline='') as combineFile:
        abcsv = csv.writer(combineFile, dialect='excel')
        for i in range(m2):
            writein=[]
            writein.append(classifiedlabel[i,0])
            writein.append(classifiedlabel[i,1])
            abcsv.writerow(writein) 
    return classifiedlabel

#########################################################################################################################################################

#######Envaluation########################################################################################################################################

def calcaccu(lablefilename,classifiedlabel):         #used to envalution the result of classificaition
    labelFile = open(lablefilename,'r')
    labelInfo = csv.reader(labelFile)
    m,n = shape(classifiedlabel)
    Tp=0
    Fp=0
    for linfo in labelInfo:
        appname=linfo[0]
        labelname=linfo[1]
        for i in range(m):
            if classifiedlabel[i,0] in appname:
                #print(classifiedlabel[i,0])
                if classifiedlabel[i,1]==labelname:
                    Tp += 1
                else:
                    Fp += 1

    accu=Tp/(Tp+Fp)
    print("accuracy is "+str(accu*100)+'%')



######################################################################################################################################################

#####Main#########################################################################################################################################
classify('training_data.csv','training_labels.csv','test_data.csv',0.6,0.001,50,3000,3000,kTup=('rbf',5))

#classifiedlabel=classify('training_data.csv','training_labels.csv','test_data.csv',0.6,0.001,50,3000,3000,kTup=('rbf',5))














