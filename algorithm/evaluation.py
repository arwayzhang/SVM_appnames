from numpy import *
import random
import csv

###SMO#########################################################################################################################################
def kernelTrans(X,A,kTup):
    m,n = shape(X)
    K = mat(zeros((m,1)))

    if kTup[0] == 'rbf':
        for j in range(m):
            deltaRow=X[j,:] -A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:
        print("error!")

    return K


class optstruct:
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




def calcEk(os,k):
	fXk = float(multiply(os.alphas,os.labelMat).T*os.K[:,k]+os.b)
	Ek = fXk - float(os.labelMat[k])
	return Ek

def selectJ(i,os,Ei):
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
    

def selectJrand(i,a):
    j=i
    while (j==i):
        j = int(random.uniform(0,a))
    return j

def updateEk(os,k):
	Ek = calcEk(os,k)
	os.eCache[k] = [1,Ek]

def clipAlpha(aj,Highbound,Lowbound):
    if aj>Highbound:
        aj=Highbound
    if Lowbound>aj:
        aj=Lowbound
    return aj


def innerL(i,os):
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



def smoP(dataMat,labelMat,C,toler,maxIter,kTup):
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
    	print("iteration number: %d" %iter)
    return os.b,os.alphas

##################################################################################################################################################

###Load Dataset########################################################################################################################################


def loadDataSet(datafilename,lablefilename,labelname):                  #！！！difference from "finalsvm.py", "evaluation.py" can load the cross-evaluation dataset directly
    dataFile = open(datafilename,'r')
    dataInfo = csv.reader(dataFile)

    data=[]
    label=[]
    for dinfo in dataInfo:
        linedata=[]
        n=0
        for i in dinfo:
            if n ==0:
                appname=dinfo[n]
            else:
                linedata.append(float(dinfo[n]))
            n += 1    #????????????????????
        labelFile = open(lablefilename,'r')
        labelInfo = csv.reader(labelFile)
        for linfo in labelInfo:
            linelabel=[]
            if linfo[0] == appname:
                if linfo[1] == labelname:
                    linelabel.append(1)

                else:
                    linelabel.append(-1)

                data.append(linedata)
                label.append(linelabel)
                break
    #print(label)
    dataMat=mat(data)
    labelMat=mat(label)
    return dataMat,labelMat




def loadtestdata(testfilename):
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


###################################################################################################################################333

#####Classification##############################################################################################################################3


def getlabels(lablefilename):
    labelslist=[]
    labelFile = open(lablefilename,'r')
    labelInfo = csv.reader(labelFile)
    for linfo in labelInfo:
        if linfo[1] not in labelslist:
            labelslist.append(linfo[1])
    n=len(labelslist)
    return labelslist,n



def givelabel(w,b,x):
    y=x*w+b
    if y>0:
        return 1
    else:
        return -1


def classify(datafilename,lablefilename,testfilename,C,toler,maxIter,kTup):
    labelslist,n=getlabels(lablefilename)
    testdataMat,testappnameMat=loadtestdata(testfilename)

    m2,n2 = shape(testdataMat)
    classifiedlabel=mat(zeros((m2,2)))
    classifiedlabel=classifiedlabel.astype(str)
    for k in range(n):

        labelname=labelslist[k]
        dataMat,labelMat = loadDataSet(datafilename,lablefilename,labelname)

        b,alphas = smoP(dataMat,labelMat,C,toler,maxIter,kTup)

        m,n = shape(dataMat)
        w = mat(zeros((n,1)))
        for i in range(m):
            w += multiply(alphas[i]*labelMat[i],dataMat[i,:].T)
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
    with open('finaldata&label.csv', 'w',newline='') as combineFile:
        abcsv = csv.writer(combineFile, dialect='excel')
        for i in range(m2):
            writein=[]
            writein.append(classifiedlabel[i,0])
            writein.append(classifiedlabel[i,1])
            abcsv.writerow(writein) 
    return classifiedlabel



##########################################################################################################333

#####Evaluation################################################################################################333

def calcaccu(lablefilename,classifiedlabel):
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

                if classifiedlabel[i,1]==labelname:
                    Tp += 1
                else:
                    Fp += 1

    accu=Tp/(Tp+Fp)
    print("accuracy is "+str(accu*100)+'%')




################################################################################################################

######Main######################################################################################################
classifiedlabel=classify('smalldata.csv','training_labels.csv','smalltestdata.csv',0.6,0.001,50,kTup=('rbf',5))
calcaccu('training_labels.csv',classifiedlabel)













