### Python code for the Monte Carlo simulations for the Potts model with hidden states ###
# List of functions:
# -------------------------------
# initSpin(n,nhid=1,pSpins=None)
# initCold(n,nhid=1)
# initHot(n,nhid=1)
# spinFlip(spin,nhid=1,ind=None)
# initNetw(siz,probl)
# sym(adj)
# eff(spin)
# mag(spin)
# MCcond(oldspin,newspin,adj,T,J=1,h=0,n=None)
# MCinit(npart,init,nhid=1,avdeg=None)
# makeMove(spin,adj,T,nhid=1,q=1,J=1,h=0,n=None,quiet=True)
# MCsimulation(T,npart,nSteps,init,nhid=1,avdeg=None,q=1,J=1,h=0,n=None,quiet=True)
# background(f)
# MCrun(T,npart,nSteps,init,reps,output_dir,nhid=1,avdeg=None,q=1,J=1,h=0,n=None,quiet=True)
# MC(Ti,Tf,Tstep,reps,npart,nSteps,init,nhid=1,avdeg=None,q=1,J=1,h=0,n=None,quiet=True)
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


### Functions: ###

#Initialize spins, nhid is the number of hidden spins. pSpins is probability vector. It is possible to define all probabilities, only the probabilities of visible spins +-1 (the rest is then added as zero) or if this is missing, the probability distribution is defined as uniform over all states. The hidden states with zero spin are denoted as 0,2,4,... so that the effective spin can be obtained by spin%2
def initSpin(n,nhid=1,pSpins=None):
    if pSpins==None:
        pSpins=[1/(nhid+2)]*(nhid+2)
    if len(pSpins) == 2:
        pSpins = pSpins+ [0]*nhid
    if len(pSpins) != nhid+2:
        nhid=len(pSpins)-2
    hidstates= [i for i in range(0,2*(nhid),2)]
    states=[1,-1]+hidstates
    rnd = np.random.random(n)
    Spins=np.random.choice(states, size=n, p=pSpins)
    return Spins

#Randomly initialize the spins of the particles all to +1
def initCold(n,nhid=1):
    return initSpin(n,nhid,pSpins=[1,0])

#Randomly initialize the spins of the particles to [-1,1] with equal probability (i.e., no hidden states)
def initHot(n,nhid=1):
    return initSpin(n,nhid,pSpins=[0.5, 0.5])

#Flips one spin. If not specified, it flips random spin.
def spinFlip(spin,nhid=1,ind=None):
    hidstates=[i for i in range(0,2*(nhid),2)]
    states=[1,-1]+hidstates
    spinNew=np.copy(spin)
    if ind is None:
        ind=np.random.randint(len(spin))
    curspin=spin[ind]
    spinNew[ind]=np.random.choice([x for x in states if x!=curspin])
    return(spinNew)

#count the number of spins corresponding to each state
def countSpins(spin,nhid):
    hidstates=[i for i in range(0,2*(nhid),2)]
    states=[1,-1]+hidstates
    return([list(spin).count(x) for x in states])


#Randomly initialize the underlying network. The function gives the (lower-triangle) adjacency matrix. Alternatively, it is possible to use any (lower-triangle) adjacency matrix
def initNetw(siz,probl):
    U = np.random.choice([0, 1], size=(siz,siz), p=[1-probl, probl])
    S = np.tril(U)
    S[np.diag_indices_from(S)]=0
    return S

#Calculates the matrix for matrix multiplication. It makes the lower-triangular matrix symmetrical 
#and makes the links between and to molecules equal to zero, so that they do not contribute to the Hamiltonian 
def sym(adj):
    return((adj+adj.T)%2)

#Returns the effective spin vector. Since the hidden states are represented as 0,2,4,etc., it assigns the effective spin to all hidden states to zero.
def eff(spin):
    spinE=np.copy(spin)
    spinE[spinE % 2 == 0] = 0
    return(spinE)

#Calculates the Hamiltonian of the system. Spins of particles in molecule state do not contribute.
#J is the coupling constant, h is the external field, 
#and n is another couling constant that is meant to compensate for different sizes
def H(spin,adj,J=1,h=0,n=None):
    if n is None:
        n=2*len(np.where(adj>0)[0])/len(spin)
    spinE=eff(spin)
    Ham = -J/2*spinE.dot(sym(adj).dot(spinE))/(n)-h*spinE.sum()
    return(Ham)


#Returns total magnetization (with making hidden spins equal to zero
def mag(spin):
    return(eff(spin).sum())


#Tests whether to accept a new state. If the Hamiltonian has decreased, always accept, otherwise use the MC formula
def MCcond(oldspin,newspin,adj,T,J=1,h=0,n=None):
    HamOld=H(oldspin,adj,J,h,n)
    HamNew=H(newspin,adj,J,h,n)
    if HamNew <= HamOld:
        return(True)
    else:
        r=np.exp(-1/T*(HamNew-HamOld))
        u=np.random.random()
        #print("probability: {}".format(r))
        if u<=r:
            return(True)
        else:
            return(False)

#initialize the system
def MCinit(npart,init,nhid=1,avdeg=None):
    #initialization of spins
    if init=="hot":
        spinN=initHot(npart,nhid)
    elif init=="cold":
        spinN=initCold(npart,nhid)
    else:
        pup=np.random.random()
        spinN=initSpin(npart,nhid,pSpins=[pup,1-pup])
        
#initialization of interaction matrix
    if avdeg is None:
        avdeg=npart-1
    adjN=initNetw(npart,avdeg/(npart-1))
    return(spinN,adjN)



#Make a MC step. Try to flip a spin and accept it with the MH probability. spin and adj are spin vector and adjacency matrix. 
#nhid is the number of hidden states, q does not have a purpose now, T is temperature, h is external field, J and n are coupling constants 
#quiet is for checking purposes (if false it prints whether t) 
def makeMove(spin,adj,T,nhid=1,q=1,J=1,h=0,n=None,quiet=True):
    spinOld=np.copy(spin)
    spinNew=spinFlip(spinOld,nhid)
    if MCcond(spinOld,spinNew,adj,T,J,h=0,n=None):
        if not quiet:
            print("accepted")
            print(H(spin,adj))
        return(spinNew) 
    else:
        if not quiet:
            print("rejected")
        return(spinOld)

#MC simulation of the whole trajectory. Initialize the system and then run the simulation for nSteps steps. 
def MCsimulation(T,npart,nSteps,init,nhid=1,avdeg=None,q=1,J=1,h=0,n=None,quiet=True):
   
    #initialization
    (spinN,adjN) = MCinit(npart,init,nhid,avdeg)

    #initial state
    statetraj=countSpins(spinN,nhid)

    #MC step
    t=0
    while t<nSteps:
        spinN = makeMove(spinN,adjN,T,nhid,q,J,h,n,quiet)
        statetraj=np.vstack([statetraj,countSpins(spinN,nhid)])
        t=t+1
    return(statetraj)

#wrapper function for asyncio paralellization
def background(f):
    def wrapped(*args,**kwargs):
        return asyncio.get_event_loop().run_in_executor(None,f,*args,**kwargs)
    
    return wrapped

#the whole run is then saved to a .csv file
@background
def MCrun(T,npart,nSteps,init,reps,output_dir,nhid=1,avdeg=None,q=1,J=1,h=0,n=None,quiet=True):
    traj=MCsimulation(T,npart,nSteps,init,nhid,avdeg,q,J,h,n,quiet) 
    #print(len(su),len(sd),len(sn))
    output_path = os.path.join(output_dir, 'run_{}_{}_{}_{}_{}_{}.csv'.format(T, npart, nSteps, init, q, reps))
    np.savetxt(output_path,traj,delimiter=',')
    print(T,npart,init,reps)
    return


#Monte Carlo simulation for a temperature range. It saves its configuration to an ini.tex file
def MC(Ti,Tf,Tstep,reps,npart,nSteps,init,nhid=1,avdeg=None,q=1,J=1,h=0,n=None,quiet=True):
    
    oldf=os.getcwd()
    mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    
    ininames = ("Ti","Tf","Tstep","reps","npart","nSteps","init","nhid","avdeg","q","J","h")
    inivals = (Ti,Tf,Tstep,reps,npart,nSteps,init,nhid,avdeg,q,J,h)
    np.savetxt(os.path.join(mydir, 'ini.txt'), list(zip(*(ininames, inivals))), delimiter=" = ", fmt="%s")
    
    Temps=np.linspace(Ti,Tf,Tstep)
    for iT in range(len(Temps)):
        for iR in range(reps):
            MCrun(Temps[iT],npart,nSteps,init,iR,mydir,nhid,avdeg,q,J,h,n,quiet)
    os.chdir(oldf)







