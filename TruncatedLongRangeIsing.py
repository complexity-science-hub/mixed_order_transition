### Python code for the Monte Carlo simulations for the Truncated Long-range Ising model ###
# List of functions:
# -------------------------------
# initSpin(n,pSpins=[0.5,0.5])
# initCold(n)
# initHot(n)
# spinFlip(spin,ind=None)
# dist(x,y,C=1,alpha=2)
# maxd(spins,i)
# ndom(spins)
# lendom(spins)
# hdom(l,alpha=2)
# endom(listd,alpha=2)
# mag(spin)
# H(spins,C=1,alpha=2,J=1,h=0)
# MCcond(oldspin,newspin,T,C=1,alpha=2,J=1,h=0)
# MCinit(npart,init)
# makeMove(spin,T,C=1,alpha=2,J=1,h=0,quiet=True)
# MCsimulation(T,npart,nSteps,init,C=1,alpha=2,J=1,h=0,quiet=True)
# background(f)
# MCrun(T,npart,nSteps,init,reps,C=1,alpha=2,J=1,h=0,quiet=True)
# MC(Ti,Tf,Tstep,reps,npart,nSteps,init,C=1,alpha=2,J=1,h=0,quiet=True)
# -------------------------------
# For the simulation, use the function MC, which generates the runs over a temperature range (see documentation below)


# loading packages
# -------------------------------
import numpy as np
import random
import asyncio
import os
import datetime
# -------------------------------


#Randomly initialize the spins of the particles to [-1,1] with given probability
def initSpin(n,pSpins=[0.5,0.5]):
    if sum(pSpins) != 1:
        print('error')
        return 'error'
    rnd = np.random.random(n)
    Spins = np.where(rnd < pSpins[1], 1, -1)
    return Spins

#Randomly initialize the spins of the particles all to +1
def initCold(n):
    return initSpin(n, pSpins=[0,1])

#Randomly initialize the spins of the particles to [-1,1] with equal probability
def initHot(n):
    return initSpin(n, pSpins=[0.5, 0.5])

#Flip a spin. If ind is given, it flips the ind component of the spin, otherwise it flips a random spin
def spinFlip(spin,ind=None):    
    spinNew=np.copy(spin)
    if ind is None:
        ind=np.random.randint(len(spinNew))
    spinNew[ind]=-spinNew[ind]
    return(spinNew)

#Definion of the distance function, i.e., the long-range coupling, C is the constant and alpha is the exponent of the distance 
def dist(x,y,C=1,alpha=2):
    return(C/(np.abs(x-y)**alpha))

#maximal distance of the spin domain (i.e., the part with constant spin)
def maxd(spins,i):
    n=len(spins)
    ind=1
    si=spins[i]
    for j in range(1,n):
        if spins[(i+j)%n]!=si:
            return(j-1)
    return(n-1)

#number of domains (parts of the chain with constant spin)
def ndom(spins):
    nd=1
    for j in range(len(spins)-1):
        if spins[j]!=spins[j+1]:
            nd=nd+1
    #check if the last spin has the same value as the first spin, then they belong to the same domain
    if spins[len(spins)-1]==spins[0] and nd>1:
        nd=nd-1
    return(nd)

#length of domains (parts of the chain with constant spin)
def lendom(spins):
    n=len(spins)
    listd=()
    ld=1
    for j in range(len(spins)-1):
        if spins[j]!=spins[j+1]:
            listd=np.append(listd,ld)
            ld=1
        else:
            ld=ld+1
    if spins[0]==spins[n-1] and len(listd)>0:
        listd=np.append(listd,ld+listd[0])
        listd[0]=0
    else:
        listd=np.append(listd,int(ld))
    return(listd)

#energy contribution of the domain
def hdom(l,alpha=2):
    sum = 0
    for k in range(1, int(l) + 1):
        sum += (l - k) / k**alpha
    return sum

#total energy contribution from all domains
def endom(listd,alpha=2):
    sum=0
    for k in range(len(listd)):
        sum +=hdom(listd[k],alpha)
    return(sum)


#total spin magnetization
def mag(spin):
    return(spin.sum())


#The total Hamiltonian of the system, it combines both nearest-neighbor interaction with coupling J and the truncated long-range interaction with coupling C and exponent alpha
def H(spins,C=1,alpha=2,J=1,h=0):
    n=len(spins)
    enshort=0
    for i in range(len(spins)):  
        enshort+= - J*spins[i]*spins[(i+1)%n]
    energy = enshort-C*endom(lendom(spins),alpha)
    return(energy)

#Tests whether to accept a new state. If the Hamiltonian has decreased, always accept, otherwise use the MC formula
def MCcond(oldspin,newspin,T,C=1,alpha=2,J=1,h=0):
    HamOld=H(oldspin,C,alpha,J,h)
    HamNew=H(newspin,C,alpha,J,h)
    if HamNew <= HamOld:
        return(True)
    else:
        r=np.exp(-1/T*(HamNew-HamOld))
        u=np.random.random()
        if u<=r:
            return(True)
        else:
            return(False)


#initialize the system
def MCinit(npart,init):
    #initialization of spins
    if init=="hot":
        spinN=initHot(npart)
    elif init=="cold":
        spinN=initCold(npart)
    else:
        pup=np.random.random()
        spinN=initSpin(npart,pSpins=[pup,1-pup])
    return(spinN)


#Make a MC step. Try to flip a spin and accept it with the MH probability. spin and adj are spin vector and adjacency matrix. 
#nhid is the number of hidden states, q does not have a purpose now, T is temperature, alpha is the exponent in the distance, J is the nearest-neighbor coupling, C is long-range coupling, h is external field, quiet is for checking purposes (if false it prints whether the state is accepted or rejected) 
def makeMove(spin,T,C=1,alpha=2,J=1,h=0,quiet=True):
    spinOld=np.copy(spin)
    spinNew=spinFlip(spinOld)
    if MCcond(spinOld,spinNew,T,C,alpha,J,h):
        if not quiet:
            print("accepted")
            #print(H(spinNew,C,alpha,J,h))
        return(spinNew) 
    else:
        if not quiet:
            print("rejected")
            #print(np.exp(-1/T*((H(spinNew,C,alpha,J,h)-H(spinOld,C,alpha,J,h)))))
        return(spinOld)


#MC simulation of the whole trajectory. Initialize the system and then run the simulation for nSteps steps. 
def MCsimulation(T,npart,nSteps,init,C=1,alpha=2,J=1,h=0,quiet=True):
   
    #initialization
    spinN = MCinit(npart,init)

    #initial state
    statetraj=([mag(spinN),ndom(spinN)])

    #MC step
    t=0
    while t<nSteps:
        spinN = makeMove(spinN,T,C,alpha,J,h,quiet)
        magdomh=([mag(spinN),ndom(spinN)])
        statetraj=np.vstack([statetraj,magdomh])
        t=t+1
    return(statetraj)






    
#wrapper function for asyncio paralellization
def background(f):
    def wrapped(*args,**kwargs):
        return asyncio.get_event_loop().run_in_executor(None,f,*args,**kwargs)
    
    return wrapped

#the whole run is then saved to a .csv file
@background
def MCrun(T,npart,nSteps,init,reps,C=1,alpha=2,J=1,h=0,quiet=True):
    traj=MCsimulation(T,npart,nSteps,init,C=1,alpha=2,J=1,h=0,quiet=quiet) 
    #print(len(su),len(sd),len(sn))
    np.savetxt('run_{}_{}_{}_{}_{}.csv'.format(T,npart,nSteps,init,reps),traj,delimiter=',')
    print(T,npart,init,reps)
    return


#Monte Carlo simulation for a temperature range. It saves its configuration to an ini.tex file
def MC(Ti,Tf,Tstep,reps,npart,nSteps,init,C=1,alpha=2,J=1,h=0,quiet=True):
    oldf=os.getcwd()
    mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    os.chdir(mydir)
    ininames = ("Ti","Tf","Tstep","reps","npart","nSteps","init","C","alpha","J","h")
    inivals = (Ti,Tf,Tstep,reps,npart,nSteps,init,C,alpha,J,h)
    np.savetxt('ini.txt',list(zip(*(ininames,inivals))),delimiter=" = ",fmt="%s")
    Temps=np.linspace(Ti,Tf,Tstep)
    for iT in range(len(Temps)):
        for iR in range(reps):
            MCrun(Temps[iT],npart,nSteps,init,iR,C,alpha,J,h,quiet)
    os.chdir(oldf)


