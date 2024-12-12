### Python code for the Monte Carlo simulations for the Ising model with molecule formation ###
# List of functions:
# -------------------------------
# initSpin(n,pSpins=[0.5,0.5])
# initCold(n)
# initHot(n)
# initNetw(siz,probl)
# listMolPart(adj)
# listFree(adj)
# spinFree(spin,adj)
# addMol(adj,mol=None)
# remMol(adj,mol=None)
# sym(adj)
# H(spin,adj,J=1,h=0,n=None)
# mag(spin,adj)
# MCcond(oldspin,oldadj,newspin,newadj,T,J=1,h=0,n=None)
# MCinit(npart,init,avdeg=None,pmol=0,pup=None)
# sysState(spinN,adjN)
# makeMove(spin,adj,T,q,J=1,h=0,n=None,quiet=True,makeflip=True,makemol=True,randmolspin=True)
# MCsimulation(T,npart,nSteps,init,avdeg=None,q=1,J=1,h=0,n=None,pmol=0,pup=None,makeflip=True,makemol=True,randmolspin=True,quiet=True):
# background(f)
# MCrun(T, npart, nSteps, init, reps, output_dir, avdeg=None, q=1, J=1, h=0, n=None, pmol=0, pup=None, makeflip=True, makemol=True, randmolspin=True, quiet=True)
# MC(Ti, Tf, Tstep, reps, npart, nSteps, init, avdeg=None, q=1, J=1, h=0, n=None, pmol=0, pup=None, makeflip=True, makemol=True, randmolspin=True, quiet=True)
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

#Randomly initialize the spins of the particles to [-1,1] with given probability
def initSpin(n,pSpins=[0.5,0.5]):
    if sum(pSpins) != 1:
        print('error')
        return 'error'
    rnd = np.random.random(n)
    Spins = np.where(rnd < pSpins[1], 1, -1)
    return Spin

#Randomly initialize the spins of the particles all to +1
def initCold(n):
    return initSpin(n, pSpins=[0,1])

#Randomly initialize the spins of the particles to [-1,1] with equal probability
def initHot(n):
    return initSpin(n, pSpins=[0.5, 0.5])

#Randomly initialize the underlying network. The function gives the (lower-triangle) adjacency matrix. Alternatively, it is possible to use any (lower-triangle) adjacency matrix
def initNetw(siz,probl):
    U = np.random.choice([0, 1], size=(siz,siz), p=[1-probl, probl])
    S = np.tril(U)
    S[np.diag_indices_from(S)]=0
    return S



#Returns the list of particles in the molecule state from the adjacency matrix
def listMolPart(adj):
    return([*np.where(adj==2)[0],*np.where(adj==2)[1]])

#Returns the list of molecules adjacency matrix
def listMol(adj):
    return(np.where(adj==2))

#Returns the list of particles in the molecule state from the adjacency matrix
def listFree(adj):
    return(np.setdiff1d(range(len(adj)),[*np.where(adj==2)[0],*np.where(adj==2)[1]]))

#Returns the spin vector of the free particles
def spinFree(spin,adj):
    return(spin[listFree(adj)])

#Returns the list of possible candidates for new molecules
def candMol(adj):
    return(np.where(adj==1))

#Flips one spin. If not specified, it flips random spin. If free is true, it flips only spin of a free particle, else it formally flips the spin of the molecule
def spinFlip(spin,adj,ind=None,free=True):
    
    spinNew=np.copy(spin)
    if free:
        lst=listFree(adj)
    else:
        lst=np.setdiff1d(range(len(spin)),listFree(adj))
    if len(lst)>0:
        if ind is None:
            ind=np.random.randint(len(lst))
        spinNew[lst[ind]]=-spinNew[lst[ind]]
    return(spinNew)
    
    
#Try adding a molecule to a system. The molecule is represented in the adjacency matrix by weight 2 
#in the element corresponding to the link between the elements.
#Additionally, all links to the particles that are part of the molecules are represented by weight 4. 
#(2 and 4 are chosen so that they are 0 (mod 2))
#The function either tries to add nth candidate molecule, or if not specified
#The function chooses randomly two particles that are not part of the molecule and tries to join them. 
#If there is no molecule to be added, it returns the original adjacency matrix
def addMol(adj,mol=None):
    adjNew=np.copy(adj)
    cands = candMol(adjNew)
    l=len(cands[0])
    if l>0:
        if mol is None:
            mol=np.random.randint(l)
        if mol<= l:
            ind1=max(cands[0][mol],cands[1][mol])
            ind2=min(cands[0][mol],cands[1][mol])
            adjNew[ind1,ind2]=2 #denote that the nodes of the links are in the same molecule
            for i in range(len(adjNew)): 
                if adjNew[ind1,i]==1:
                    adjNew[ind1,i]=4
                if adjNew[i,ind1]==1:
                    adjNew[i,ind1]=4
                if adjNew[i,ind2]==1:
                    adjNew[i,ind2]=4
                if adjNew[ind2,i]==1:
                    adjNew[ind2,i]=4
    return(adjNew)
        


    
#Tries to remove one molecule. The function either tries to remove nth molecule, 
#if not specified, it chooses randomly one molecule and removes it. 
#If there are no molecules, it returns the same adjacency matrix.
def remMol(adj,mol=None):
    adjNew=np.copy(adj)
    lst=listMol(adjNew) #get list of molecules
    lst2=[*lst[0],*lst[1]] #get list of particles in the molecules
    if len(lst[0])==0:
        return(adjNew)
    if len(lst[0])==1:
        ind1=max(lst[0],lst[1])
        ind2=min(lst[0],lst[1])
        adjNew[ind1,ind2]=1
        for i in range(len(adjNew)):
            if adjNew[ind1,i]==4:
                adjNew[ind1,i]=1
            if adjNew[i,ind1]==4:
                adjNew[i,ind1]=1
            if adjNew[i,ind2]==4:
                adjNew[i,ind2]=1
            if adjNew[ind2,i]==4:
                adjNew[ind2,i]=1
        return(adjNew)
    if len(lst[0])>1:
        l=len(lst[0])
        if mol is None:
            mol=np.random.randint(l)
        if mol <= l:
            ind1=max(lst[0][mol],lst[1][mol])
            ind2=min(lst[0][mol],lst[1][mol])
            adjNew[ind1,ind2]=1
            for i in range(len(adjNew)):
                if adjNew[ind1,i]==4 and i not in lst2: 
                    adjNew[ind1,i]=1
                if adjNew[i,ind1]==4 and i not in lst2:
                    adjNew[i,ind1]=1
                if adjNew[i,ind2]==4 and i not in lst2:
                    adjNew[i,ind2]=1
                if adjNew[ind2,i]==4 and i not in lst2:
                    adjNew[ind2,i]=1

        return(adjNew)
    

    
#Calculates the matrix for matrix multiplication. It makes the lower-triangular matrix symmetrical 
#and makes the links between and to molecules equal to zero, so that they do not contribute to the Hamiltonian 
def sym(adj):
    return((adj+adj.T)%2)

#Calculates the Hamiltonian of the system. Spins of particles in molecule state do not contribute.
#J is the coupling constant, h is the external field, 
#and n is another couling constant that is meant to compensate for different sizes
def H(spin,adj,J=1,h=0,n=None):
    if n is None:
        n=2*len(np.where(adj>0)[0])/len(spin)
    Ham = -J/2*spin.dot(sym(adj).dot(spin))/(n)-h*spinFree(spin,adj).sum()
    return(Ham)

#Returns total magnetization of free particles
def mag(spin,adj):
    return(spin[listFree(adj)].sum())


#Tests whether to accept a new state. If the Hamiltonian has decreased, always accept, otherwise use the MC formula
def MCcond(oldspin,oldadj,newspin,newadj,T,J=1,h=0,n=None):
    HamOld=H(oldspin,oldadj,J,h,n)
    HamNew=H(newspin,newadj,J,h,n)
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

#initialize the system, n part is the number of molecules, init the way to initialize the system - "hot" means uniform spin distribution "cold" means all spins in +1, everything else takes the probability pup (or drawn randomly if not speficied). avdeg is the average degree (default is fully connected) pmol is the initial probability of observing molecules 
def MCinit(npart,init,avdeg=None,pmol=0,pup=None):
    #initialization of spins
    if init=="hot":
        spinN=initHot(npart)
    elif init=="cold":
        spinN=initCold(npart)
    else:
        if pup is None:
            pup=np.random.random()
        spinN=initSpin(n,pSpins=[pup,1-pup])
#initialization of interaction matrix
    if avdeg is None:
        avdeg=npart-1
    adjN=initNetw(npart,avdeg/(npart-1))
#initialization of molecules
    if pmol>0:
        nmol=math.floor((npart*pmol)/2)
        for i in range(nmol):
            adjN=addMol(adjN)
    return(spinN,adjN)


def sysState(spinN,adjN):
    if (np.unique(spinFree(spinN,adjN))==[-1,1]).all():
        (spindown,spinup)=np.unique(spinFree(spinN,adjN),return_counts = True)[1]
    elif (np.unique(spinFree(spinN,adjN))==[-1]).all():
        (spindown,spinup)=(np.unique(spinFree(spinN,adjN),return_counts = True)[1],0)
    elif (np.unique(spinFree(spinN,adjN))==[1]).all():
        (spindown,spinup)=(0,np.unique(spinFree(spinN,adjN),return_counts = True)[1])
    nmol= len(listMol(adjN)[0])
    return(spindown,spinup,nmol)
    
        
#Make a MC step, including spin flip, add or remove molecule. spin and adj are spin vector and effective adjacency matrix. 
#T is temperature, h is external field, J and n are coupling constants, maxStep is the maximum number of rejected trials, 
#quiet is for checking purposes (if false it writes all attempts) 
#makeflip is whether make spin flips is allowed, makemol is whether making molecule moves is allowed
#randmolspin makes random spin moves in the molecules to get random spin of the molecules once the molecule is destroyed
#here we take the step even if it is rejected

def makeMove(spin,adj,T,q,J=1,h=0,n=None,quiet=True,makeflip=True,makemol=True,randmolspin=True):
    spinOld=np.copy(spin)
    adjOld=np.copy(adj)
    if makemol:
        cm=candMol(adjOld)
        lm=listMol(adjOld)
    else:
        cm=[[],[]]
        lm=[[],[]]
    if makeflip:
        lf=listFree(adjOld)
    else:
        lf=[]
    tot=len(cm[0])+len(lm[0])+len(lf)
    npart=len(spinOld)
    if len(lm[0])+1>(q*npart)/2:
        ind=np.random.randint(len(cm[0]),tot-1)
    else:
        ind=np.random.randint(0,tot-1)
    if not quiet:
        print("index {}".format(ind))
    if ind < len(cm[0]):
        adjNew=addMol(adjOld,ind)
        spinNew=np.copy(spinOld)
        if not quiet:
            print("adding molecule {}-{}".format(cm[0][ind],cm[1][ind]))
    elif ind - len(cm[0]) < len(lm[0]):
        adjNew=remMol(adjOld,ind-len(cm[0]))
        spinNew=np.copy(spinOld)
        if not quiet:
            print("removing molecule {}-{}".format(lm[0][ind-len(cm[0])],lm[1][ind-len(cm[0])]))
    elif ind - len(cm[0]) - len(lm[0]) <len(lf):
        spinNew=spinFlip(spinOld,adjOld,ind - len(cm[0]) - len(lm[0]))
        adjNew=np.copy(adjOld)
        if not quiet:
            print("flipping spin {}".format(lf[ind - len(cm[0]) - len(lm[0])]))
    if MCcond(spinOld,adjOld,spinNew,adjNew,T,J,h=0,n=None):
        return(spinNew,adjNew) 
    else:
        if randmolspin:
            for i in range(len(lm[0])):
                spinOld=spinFlip(spinOld,adjOld,free=False)
        if not quiet:
            print("rejected")
        return(spinOld,adjOld)

   
#MC simulation of the Ising system with molecules. if new is True, we count both rejected and accepted steps. If new is false we try to get an accepted transition until a maximum number of trials is exceeded. 

def MCsimulation(T,npart,nSteps,init,avdeg=None,q=1,J=1,h=0,n=None,pmol=0,pup=None,makeflip=True,makemol=True,randmolspin=True,quiet=True):
    #initialization
    (spinN,adjN) = MCinit(npart,init,avdeg,pmol,pup)

    #initial state
    if (np.unique(spinFree(spinN,adjN))==[-1,1]).all():
        (spindown,spinup)=np.unique(spinFree(spinN,adjN),return_counts = True)[1]
    elif (np.unique(spinFree(spinN,adjN))==[-1]).all():
        spindown=np.unique(spinFree(spinN,adjN),return_counts = True)[1]
        spinup=0
    elif (np.unique(spinFree(spinN,adjN))==[1]).all():
        spinup=np.unique(spinFree(spinN,adjN),return_counts = True)[1]
        spindown=0
    
    nmol= len(listMol(adjN)[0])

    t=0
    while t<nSteps:
            (spinN,adjN)=makeMove(spinN,adjN,T,q,J,h,n=None,quiet=quiet,makeflip=True,makemol=True,randmolspin=True)
        if len(spinFree(spinN,adjN))==0:
            spinup = np.append(spinup,0)
            spindown = np.append(spindown,0)
        elif (np.unique(spinFree(spinN,adjN))==-1).all():
            spindown = np.append(spindown,np.unique(spinFree(spinN,adjN),return_counts = True)[1])
            spinup = np.append(spinup,0)
        elif (np.unique(spinFree(spinN,adjN))==1).all():
            spinup = np.append(spinup,np.unique(spinFree(spinN,adjN),return_counts = True)[1])
            spindown = np.append(spindown,0)
        elif (np.unique(spinFree(spinN,adjN))==[-1,1]).all():
            spindown = np.append(spindown,np.unique(spinFree(spinN,adjN),return_counts = True)[1][0])
            spinup = np.append(spinup,np.unique(spinFree(spinN,adjN),return_counts = True)[1][1])
       
        nmol = np.append(nmol,len(listMol(adjN)[0]))
        t+=1
        #print(np.unique(spinFree(spinN,adjN),return_counts = True))
        #print(len(spinup),len(spindown),len(nmol))
    return(spinup,spindown,nmol)


#wrapper function for asyncio paralellization
def background(f):
    def wrapped(*args,**kwargs):
        return asyncio.get_event_loop().run_in_executor(None,f,*args,**kwargs)
    
    return wrapped

#parallelized run. The whole run is then saved to a .csv file
@background
def MCrun(T, npart, nSteps, init, reps, output_dir, avdeg=None, q=1, J=1, h=0, n=None, pmol=0, pup=None, makeflip=True, makemol=True, randmolspin=True, quiet=True):
    (su, sd, sn) = MCsimulation(T, npart, nSteps, init, avdeg, q, J, h, n, pmol, pup, makeflip, makemol, randmolspin, quiet)
    output_path = os.path.join(output_dir, 'run_{}_{}_{}_{}_{}_{}.csv'.format(T, npart, nSteps, init, q, reps))
    np.savetxt(output_path, (su, sd, sn), delimiter=',')
    print(T,npart,init,q,reps)
    return



#Monte Carlo simulation over the given temperature range (Ti is the initial temperature, Tf is the final temperature and Tstep is the number of steps).
def MC(Ti, Tf, Tstep, reps, npart, nSteps, init, avdeg=None, q=1, J=1, h=0, n=None, pmol=0, pup=None, makeflip=True, makemol=True, randmolspin=True, quiet=True):
    oldf = os.getcwd()  # Save current directory
    mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))  # Create a new directory for this run
    os.makedirs(mydir)
    
    # Save simulation parameters to an ini file
    ininames = ("Ti", "Tf", "Tstep", "reps", "npart", "nSteps", "init", "avdeg", "q", "J", "h", "n", "pmol", "pup", "makeflip", "makemol", "randmolspin")
    inivals = (Ti, Tf, Tstep, reps, npart, nSteps, init, avdeg, q, J, h, n, pmol, pup, makeflip, makemol, randmolspin)
    np.savetxt(os.path.join(mydir, 'ini.txt'), list(zip(*(ininames, inivals))), delimiter=" = ", fmt="%s")
    
    # Generate temperature range
    Temps = np.linspace(Ti, Tf, Tstep)
    
    # Run simulations in parallel for each temperature and repetition
    for iT in range(len(Temps)):
        for iR in range(reps):
            MCrun(Temps[iT], npart, nSteps, init, iR, mydir, avdeg, q, J, h, n, pmol, pup, makeflip, makemol, randmolspin, quiet)
    
    os.chdir(oldf)  # Change back to the original directory
    return

