import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools as IT

exp = np.exp; norm = np.linalg.norm; Arr=np.array; md = np.linalg.multi_dot
def CZ(s): return np.zeros(s,dtype=np.complex128)
def RZ(s): return np.zeros(s,dtype=np.float64)
TB_Directory='/home/aleks/Desktop/QE_calc2/Gen/TB_data_me/'
Tr    = np.trace; 
Inv   = np.linalg.inv; 
Solve = np.linalg.solve
MM=np.matmul
        
from scipy.constants import hbar, eV, m_e
from scipy import integrate
from scipy.special import jv
Thz=hbar/eV * 2*np.pi*10**12

def x_min_x_max(mu1,mu2,hw,kT=0.025,p=5):
    b=(mu1-mu2+hw)/kT
    eb=np.exp(b)
    pot=10**p
    
    sol_0  = np.log(0.5*np.exp(-b) * (  -( (-eb*pot+eb+pot+1)**2 -4*eb    )**0.5  +  eb*pot - eb - pot-1))
    sol_1  = np.log(0.5*np.exp(-b) * (  +( (-eb*pot+eb+pot+1)**2 -4*eb    )**0.5  +  eb*pot - eb - pot-1))
    
    return sol_0*kT+mu1, sol_1*kT+mu1

def integrate_f_x(f,x):
    check_n=len(x)-1
    if (check_n & (check_n-1) == 0) and check_n != 0 and np.isclose(x,np.linspace(x.min(),x.max(),check_n+1)).all():
        dx=x[1]-x[0]
        #Romberg integration converges a bit fast
        #return integrate.romb(f,dx=dx)
    else:
        return integrate.simps(f,x=x,even='last')

def five_point_stencil(four_vals,h):
    # https://en.wikipedia.org/wiki/Numerical_differentiation#cite_note-9
    val2p,val1p,val1m,val2m  = four_vals[0],four_vals[1],four_vals[2],four_vals[3]
    return (-val2p+8*val1p-8*val1m + val2m)/(12*h)

def W(x,x1,x2,c,Type = '??'):
    dx =x2-x1
    y = c*(x-x1)/dx
    if Type == '??': 
        #Wp=-(hbar/eV)**2/(2*m_e)*(2*np.pi/dx)**2*(4/(c-y)**2 + 4/(c+y)**2 - 8/c**2)
        Wp= (hbar/eV)**2/(2*m_e)*(2*np.pi/dx)**2* 4 / c**2 * (dx**2/(x2 - 2 * x1 +x)**2 + dx**2/(x2-x)**2 - 2)
        
    if Type == 'Potens':
        return (c+1)*np.abs(x-x1)**c
    return Wp

def SC(p):
    plt.scatter(p[:,0],p[:,1])

def Tr_Prod(a,b):
    if len(a.shape)==len(b.shape)==2:
        return (a*(b.T)).sum()
    if len(a.shape)==len(b.shape)==3:
        return (a*(b.transpose(0,2,1))).sum(axis=(1,2))
    if len(a.shape)==len(b.shape)==4:
        return (a*(b.transpose(0,1,3,2))).sum(axis=(2,3))
    if len(a.shape)==2 and len(b.shape)==3:
        return (a*(b.transpose(0,2,1))).sum(axis=(1,2))
    if len(a.shape)==3 and len(b.shape)==2:
        return (a*(b.T)).sum(axis=(1,2))
    if len(a.shape)==2 and len(b.shape)==4:
        return (a*(b.transpose(0,1,3,2))).sum(axis=(2,3))
    if len(a.shape)==4 and len(b.shape)==2:
        return (a*(b.T)).sum(axis=(2,3))


def Gmmlssr(SE_array,E_arr,mu = 0,kT = 0.025):
    #SE_array i format(N_e,n_orb,n_orb)
    #E_arr i format (N_e)
    dist = f(E_arr,mu = mu,kT = kT)
    dist = dist[:,np.newaxis,np.newaxis]
    return 1j * Gamma(SE_array) * dist

def UAU_D(U,A):
    if len(A.shape)==len(U.shape)==2:
        return md([U,A,Dag(U)])
    if len(A.shape)==3 and len(U.shape)==2:
        return MM(MM(U,A),Dag(U))
    if len(A.shape)==4 and len(U.shape)==2:
        return MM(MM(U,A),Dag(U))


def SE(E_in,H,V,eta=1e-3,eps=1e-15):
    n=len(H)
    alpha = V.copy()
    beta =  V.T.conj().copy()
    I=np.diag(np.ones(n))
    igb=I*(E_in+1j*eta) - H
    sse=np.zeros((n,n),np.complex128)
    it_count=0
    while True:
        gb=np.linalg.inv(igb)
        gb_beta=np.dot(gb,beta)
        gb_alpha=np.dot(gb,alpha)
        sse += alpha.dot(gb_beta)
        igb-=alpha.dot(gb_beta) + beta.dot(gb_alpha)
        alpha=alpha.dot(gb_alpha)
        beta=beta.dot(gb_beta)
        it_count+=1
        if np.abs(alpha).sum()+np.abs(beta).sum()<eps:
            return sse

def SE_vectorised(E_in,H,V,S00=None,S01=None,eta=1e-3,eps=1e-15,DT=np.complex128):
    # numpy broadcaster til de sidste to indexer i arrayet, dvs så længe vi holder de egentlige
    # to indekser vi vi gerne vil invertere, matrix-multiplicere osv i de sidste to kan man vectorisere
    # indexet over energi. H og V afhænger af k-indekset, ud den går ikke her
    n_e = len(E_in)
    n=len(H)
    alpha = CZ((n_e,n,n))
    #alpha[:,:,:] += V
    beta = CZ((n_e,n,n))
    #beta [:,:,:] +=  V.T.conj()
    if S00 is None and S01 is None:
        I=np.diag(np.ones(n))
        igb = CZ((n_e,n,n))
        for i in range(n_e):
            igb[i,:,:] =I*(E_in[i]+1j*eta) - H
            alpha[i,:,:] = V
            beta [i,:,:] = V.conj().T
    else:
        igb = CZ((n_e,n,n))
        for i in range(n_e):
            z = (E_in[i]+1j*eta)
            igb[i,:,:] =S00*z - H
            alpha[i,:,:] = V - S01 * z
            beta [i,:,:] = V.conj().T - S01.conj().T * z
    sse=np.zeros((n_e,n,n),dtype=DT)
    while True:
        gb       = Inv(igb)
        gb_beta  = MM(gb,beta)
        gb_alpha = MM(gb,alpha)
        sse     += MM(alpha,gb_beta)
        igb     -= MM(alpha,gb_beta) + MM(beta,gb_alpha)
        alpha    = MM(alpha,gb_alpha)
        beta     = MM(beta,gb_beta)
        if ((np.sum(np.abs(alpha),axis=(1,2))+np.sum(np.abs(beta),axis=(1,2)))<eps).all():
            return sse

def MMM(x,y,z):
    return MM(MM(x,y),z)

def f(e,mu = 0,kT = 0.025): 
    x=(e-mu)/kT
    return 1/(1+np.exp(x))

def Gamma(A):
    if len(A.shape)==2:
        return 1j*(A-A.conj().T)
    if len(A.shape)==3:
        return 1j*(A-A.conj().transpose(0,2,1))
    if len(A.shape)==4:
        return 1j*(A-A.conj().transpose(0,1,3,2))

def Dag(A):
    if len(A.shape) == 2:
        return A.conj().T
    if len(A.shape) == 3:
        return A.conj().transpose(0,2,1)
    if len(A.shape) == 4:
        return A.conj().transpose(0,1,3,2)

def batch_k_points(k,n_batches):
    n=len(k[:,0])
    L = []
    for i in range(n_batches):
        L+=[[]]
    
    for i in range(n):
        L[np.mod(i,n_batches)]+=[k[i,:]]
    
    for i in range(len(L)):
        L[i] = np.array(L[i])
    return L

class TB_Model:
    def __init__(self,
                 tb_model,
                 Thz_max=0,
                 k_uv=np.array([0,1,0]),
                 nsc=(3,3,1),
                 orbi_types=None,
                 Emin = -2.5,
                 Emax = 2.5,
                 ne = 100,
                 eta = 1e-3,
                 manual_ef = None,
                 multiple_absorption=0):
        
        tb_model.set_nsc(nsc)
        self.TB = tb_model
        self.k_uv=k_uv
        self.Reciprocal_lattice()
        self.Get_H_R()
        self.orbi_types=orbi_types
        self.E =     np.linspace(Emin,Emax, ne)
        self.dE = self.E[1]-self.E[0]
        
        below = np.arange(Emin-self.dE,Emin-multiple_absorption*Thz_max-2*self.dE,-self.dE)
        above = np.arange(Emax+self.dE,Emax+multiple_absorption*Thz_max+2*self.dE,+self.dE)
        
        if multiple_absorption>0:
            self.E = np.hstack([below,
                                self.E,
                                above])
            self.inds_cent = np.arange(len(below),len(below)+ne)
        
        self.max_shift = len(above)
        self._above = above
        self._below = below
        self.eta=eta
        self.Thz_max = Thz_max
        self.fot_E_for_copy=np.arange(Emax+self.dE,Emax+Thz_max, self.dE)-Emax
        self.CAP_ADDED=False
    
    def shift_ind(self,n,tol = 1e-5):
        # giver indeksmængden som passer med arrayet bliver shiftet med skridt i 
        # E-indexet når n er et heltal
        # de to indeksmænder til lineær interpolering i E-indexet hvis n er  n float
        f = n-int(n)
        if int(np.ceil(abs(n)))>self.max_shift:
            print('shifted index out of range of the ones calculated.....')
            assert 1==0
        if np.abs(f)<tol:
            return [self.inds_cent+int(n)-1, self.inds_cent+int(n)    ], [0,1]
        elif 1>np.abs(f)>1-tol:
            return [self.inds_cent+int(n)  , self.inds_cent+int(n)+1  ], [0,1]
        else:
            if f >  0:
                return [self.inds_cent+int(n),self.inds_cent+int(n)+1],[1-f,f]
            elif f <  0:
                return [self.inds_cent+int(n)-1,self.inds_cent+int(n)],[abs(f),1-abs(f)]
    
    def Set_kp(self,k_in):
        self.k_avg = k_in
        self.nk    = len(self.k_avg)
        if self.nk==1 and isinstance(k_in,list) and k_in==[None]:
            print('\n No k-points!\n')
            self.phases = [np.complex128(0+0j)]
        else:
            self.phases = [np.exp(1j*self.k_avg[i,:].dot(self.k_uv)) for i in range(self.nk)]
    
    def Reciprocal_lattice(self):
        a1=self.TB.cell[0,:]
        a2=self.TB.cell[1,:]
        a3=self.TB.cell[2,:]
        V=np.cross(a1,a2).dot(a3)
        b1 = 2*np.pi*np.cross(a2,a3)/V
        b2 = 2*np.pi*np.cross(a3,a1)/V
        b3 = 2*np.pi*np.cross(a1,a2)/V
        self.a1=a1
        self.a2=a2
        self.a3=a3
        self.b1=b1
        self.b2=b2
        self.b3=b3
    
    def Get_H_R(self):
        n_orb = self.TB.shape[0]
        n_uc = self.TB.shape[1]//n_orb
        RR = self.TB.Rij()
        Rij = np.zeros((n_orb,n_orb,n_uc,3))
        Hij = CZ((n_orb,n_orb,n_uc))
        R_bloch = RZ(Rij.shape)
        l1=self.TB.nsc[0]//2
        l2=self.TB.nsc[1]//2
        l3=self.TB.nsc[2]//2
        
        for i in range(n_orb):
            for j in range(n_orb):
                it=0
                for I in range(-l1,l1+1):
                    for J in range(-l2,l2+1):
                        for K in range(-l3,l3+1):
                            hij=self.TB[i,j,(I,J,K)]
                            rij =             I*self.a1 + J*self.a2 + K*self.a3 - (self.TB.xyz[i,:]-self.TB.xyz[j,:])
                            R_bloch[i,j,it,:]=I*self.a1 + J*self.a2 + K*self.a3
                            Hij[i,j,it]  = hij
                            if hij!=0:
                                Rij[i,j,it,:]=RR[i,j,(I,J,K)]
                            else:
                                Rij[i,j,it,:]=rij
                        it+=1
        self.Rarr = Rij
        self.Harr = Hij
        self.R_bloch=R_bloch
        self.n_orb = n_orb
        self.n_uc  = n_uc
    
    
    def Get_hoppings_s1mple(self,r1,r2,t1,t2,TreD=False):
        R=r2-r1
        hij = 0+0j
        ####
        T_3d = (self.Rarr[t1,t2,:,2]-R[2])**2
        ####
        N=np.sqrt((self.Rarr[t1,t2,:,0]-R[0])**2+(self.Rarr[t1,t2,:,1]-R[1])**2+T_3d)
        inds=np.where(N<1e-10)[0]
        if len(inds)>0:
            hij = self.Harr[t1,t2,inds[0]]
            if len(inds)>1:
                print('???')
        return hij
    
    def Get_hoppings_distance_dependent(self,r1,r2,d0=1.46,r_cut=1.5,force_const = False,hop = -2.7):
        dij=np.linalg.norm(r1-r2)
        if 0.1<dij<r_cut:
            if force_const == True:
                return hop*1
            else:
                return 1*hop*(d0/dij)**2
        else:
            return 0
    
    def Make_new_basis(self,S,B):
        Lat = self.TB.cell.T.dot(S)
        Basis=[]
        orbitals = []
        for i in range(len(B)):
            for j in range(len(self.TB.xyz[:,0])):
                Basis+=[self.TB.xyz[j,:]+self.a1*B[i][0]+self.a2*B[i][1]]
                orbitals+=[j]
        self.new_basis_orbitals  = orbitals
        self.A1 = Lat[:,0]
        self.A2 = Lat[:,1]
        self.new_basis=Basis
        A3=np.array([0,0,1])
        Vol=np.cross(self.A1,self.A2).dot(A3)
        B1=2*np.pi*np.cross(self.A2,A3)/Vol
        B2=2*np.pi*np.cross(A3,self.A1)/Vol
        self._B1=B1
        self._B2=B2
    
    def Manual_input(self,pos_d,pos_em,pos_ep,cell_d,cell_l,cell_r,orb_d,orb_l,orb_r):
        self.pos_d=pos_d
        self.pos_l=pos_em
        self.pos_r=pos_ep
        self._cell_d=cell_d
        self._cell_r=cell_r
        self._cell_l=cell_l
        self.orbitals_d=orb_d
        self.orbitals_r=orb_r
        self.orbitals_l=orb_l
        
        T_d_up=cell_d[1,:]
        T_r_up=cell_r[1,:]
        T_l_up=cell_l[1,:]
        T_r_right=cell_r[0,:]
        T_l_left=-cell_l[0,:]
        
        self.t_pos_d_up    = np.array([p+T_d_up    for p in self.pos_d])
        self.t_pos_r_up    = np.array([p+T_r_up    for p in self.pos_r])
        self.t_pos_l_up    = np.array([p+T_l_up    for p in self.pos_l])
        
        self.t_pos_d_down  = np.array([p-T_d_up    for p in self.pos_d])
        self.t_pos_r_down  = np.array([p-T_r_up    for p in self.pos_r])
        self.t_pos_l_down  = np.array([p-T_l_up    for p in self.pos_l])
        
        self.t_pos_r_right = np.array([p+T_r_right for p in self.pos_r])
        self.t_pos_l_left  = np.array([p+T_l_left  for p in self.pos_l])
        
        self.t_pos_r_right_up =   np.array([p+T_r_right + T_r_up for p in self.pos_r])
        self.t_pos_l_left_up  =   np.array([p+T_l_left  + T_l_up for p in self.pos_l])
        
        self.t_pos_r_right_down = np.array([p+T_r_right - T_r_up for p in self.pos_r])
        self.t_pos_l_left_down  = np.array([p+T_l_left  - T_l_up  for p in self.pos_l])
    
    def Make_device_lead_pos(self,
                             Rep1_d,Rep2_d,
                             Rep1_ll,Rep2_ll,
                             Rep1_lr,Rep2_lr,
                             move_d=np.zeros(3),move_ll=np.zeros(3),move_lr=np.zeros(3),
                             CHole=[np.zeros(3)],
                             RHole=[-0.5],
                             Ribbon = False):
        pos_d=[]
        orbitals_d = []
        for i in range(Rep1_d):
            for j in range(Rep2_d):
                T=i*self.A1 + j*self.A2
                i_orb = 0
                for p in self.new_basis:
                    r=p+T+move_d
                    hits = 0
                    for it,h in enumerate(CHole):
                        if np.linalg.norm(r-h)<=RHole[it]:
                            hits+=1
                    if hits==0:
                        pos_d+=[p+T+move_d]
                        orbitals_d+=[self.new_basis_orbitals[i_orb]]
                    i_orb+=1
                if j==Rep2_d-1 and Ribbon:
                    i_orb=2
                    for p in self.new_basis[2:]:
                        pos_d+=[p+T+self.A2+move_d]
                        orbitals_d+=[self.new_basis_orbitals[i_orb]]
                        i_orb+=1
        
        pos_l=[]
        orbitals_l=[]
        for i in range(-Rep1_ll,0):
            for j in range(Rep2_ll):
                T=i*self.A1 + j*self.A2
                i_orb = 0
                for p in self.new_basis:
                    pos_l+=[p+T+move_ll]
                    orbitals_l+=[self.new_basis_orbitals[i_orb]]
                    i_orb+=1
                if j==Rep2_ll-1 and Ribbon:
                    i_orb=2
                    for p in self.new_basis[2:]:
                        pos_l+=[p+T+self.A2+move_ll]
                        orbitals_l+=[self.new_basis_orbitals[i_orb]]
                        i_orb+=1
        
        pos_r=[]
        orbitals_r=[]
        for i in range(Rep1_d,Rep1_d+Rep1_ll):
            for j in range(Rep2_lr):
                T=i*self.A1 + j*self.A2
                i_orb=0
                for p in self.new_basis:
                    pos_r+=[p+T+move_lr]
                    orbitals_r+=[self.new_basis_orbitals[i_orb]]
                    i_orb+=1
                if j==Rep2_lr-1 and Ribbon:
                    i_orb=2
                    for p in self.new_basis[2:]:
                        pos_r+=[p+T+self.A2+move_lr]
                        orbitals_r+=[self.new_basis_orbitals[i_orb]]
                        i_orb+=1
        
        self.pos_d=np.array(pos_d)
        self.pos_r=np.array(pos_r)
        self.pos_l=np.array(pos_l)
        self.orbitals_d=orbitals_d
        self.orbitals_r=orbitals_r
        self.orbitals_l=orbitals_l
        self.Rep1_d=Rep1_d
        self.Rep2_d=Rep2_d
        self.Rep1_ll=Rep1_ll
        self.Rep2_ll=Rep2_ll
        self.Rep1_lr=Rep1_lr
        self.Rep2_lr=Rep2_lr
    
    def Make_translations(self):
        T_d_up     =  self.A2*self.Rep2_d
        T_r_up     =  self.A2*self.Rep2_lr
        T_l_up     =  self.A2*self.Rep2_ll
        T_r_right  =  self.A1*self.Rep1_lr
        T_l_left   = -self.A1*self.Rep1_ll
        
        self.t_pos_d_up    = np.array([p+T_d_up    for p in self.pos_d])
        self.t_pos_r_up    = np.array([p+T_r_up    for p in self.pos_r])
        self.t_pos_l_up    = np.array([p+T_l_up    for p in self.pos_l])
        
        self.t_pos_d_down  = np.array([p-T_d_up    for p in self.pos_d])
        self.t_pos_r_down  = np.array([p-T_r_up    for p in self.pos_r])
        self.t_pos_l_down  = np.array([p-T_l_up    for p in self.pos_l])
        
        self.t_pos_r_right = np.array([p+T_r_right for p in self.pos_r])
        self.t_pos_l_left  = np.array([p+T_l_left  for p in self.pos_l])
        
        self.t_pos_r_right_up =   np.array([p+T_r_right + T_r_up for p in self.pos_r])
        self.t_pos_l_left_up  =   np.array([p+T_l_left  + T_l_up for p in self.pos_l])
        
        self.t_pos_r_right_down = np.array([p+T_r_right - T_r_up for p in self.pos_r])
        self.t_pos_l_left_down  = np.array([p+T_l_left  - T_l_up  for p in self.pos_l])
    
    def Build_H(self,P1,O1,P2,O2,give_pos=False,distance_dependent=False, force_const = False,hop=-2.7):
        H=CZ((len(P1),len(P2)))
        for i in range(len(P1)):
            pi=P1[i,:]; oi=O1[i]
            for j in range(len(P2)):
                pj=P2[j,:]; oj=O2[j]
                if distance_dependent==False:
                    H[i,j] = self.Get_hoppings_s1mple(pi,pj,oi,oj)
                elif distance_dependent==True:
                    H[i,j] = self.Get_hoppings_distance_dependent(pi,pj,force_const=force_const,hop=hop)
        return H
    
    def RAM(self):
        return self.nk*len(self.E)*self.H_dd.shape[0]*self.H_dd.shape[1]*3*16*10**-9
    
    def Build_all(self,distance_dependent=False,force_const = False, hop = -2.7):
        OD = self.orbitals_d
        OR = self.orbitals_r
        OL = self.orbitals_l
        print('-----Building TB Hamiltonians-----\n')
        
        self.H_dd    = self.Build_H(self.pos_d,OD,self.pos_d,OD,     distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_01_dd = self.Build_H(self.pos_d,OD,self.t_pos_d_up,OD,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.V_00_dl = self.Build_H(self.pos_d,OD,self.pos_l,OL,     distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_00_dr = self.Build_H(self.pos_d,OD,self.pos_r,OR,     distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.V_01_dl = self.Build_H(self.pos_d,OD,self.t_pos_l_up,OL,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_01_ld = self.Build_H(self.pos_l,OL,self.t_pos_d_up,OD,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.V_01_dr = self.Build_H(self.pos_d,OD,self.t_pos_r_up,OR,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_01_rd = self.Build_H(self.pos_r,OR,self.t_pos_d_up,OD,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.H_ll    = self.Build_H(self.pos_l,OL,self.pos_l,            OL,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_lL    = self.Build_H(self.pos_l,OL,self.t_pos_l_left,     OL,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_lL_01 = self.Build_H(self.pos_l,OL,self.t_pos_l_left_up,  OL,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_lL_0m1= self.Build_H(self.pos_l,OL,self.t_pos_l_left_down,OL,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.V_01_ll = self.Build_H(self.pos_l,OL,self.t_pos_l_up,   OL,     distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.H_rr    = self.Build_H(self.pos_r,OR,self.pos_r,        OR,     distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_rR    = self.Build_H(self.pos_r,OR,self.t_pos_r_right,OR,     distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.V_rR_01 = self.Build_H(self.pos_r,OR,self.t_pos_r_right_up,  OR,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        self.V_rR_0m1= self.Build_H(self.pos_r,OR,self.t_pos_r_right_down,OR,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
        
        self.V_01_rr = self.Build_H(self.pos_r,OR,self.t_pos_r_up,        OR,distance_dependent=distance_dependent,force_const=force_const,hop=hop)
    
    def Make_band_diagram_for_basis(self,nkp,EtD = True):
        assert self.Rep1_d==self.Rep1_ll
        assert self.Rep2_d==self.Rep2_ll
        assert self.Rep1_d==self.Rep1_lr
        assert self.Rep2_d==self.Rep2_lr
        if EtD == True:
            path = np.zeros((nkp,3))
            it=0
            for i in IT.product(range(nkp),repeat=1):
                path[it,:]=np.array([i[0]/nkp,0,0])
                it+=1
            eigs = np.zeros((nkp,self.H_dd.shape[0]))
        else:   
            path = np.zeros((nkp*nkp,3))
            it=0
            for i, j in IT.product(range(nkp),repeat=2):
                path[it,:]=np.array([i/nkp,j/nkp,0])
                it+=1
                
            eigs = np.zeros((nkp*nkp,self.H_dd.shape[0]))
        it=0
        for k in path:
            K=k[0]*self._B1 + k[1]*self._B2
            H_k =self.H_dd+(     self.V_01_dd*         np.exp(1j*K.dot( self.A2))+
                                 self.V_01_dd.T.conj()*np.exp(1j*K.dot(-self.A2))+
                                 self.V_00_dl*     np.exp(1j*K.dot(-self.A1))+
                                 self.V_00_dr*     np.exp(1j*K.dot(+self.A1))+
                                 self.V_01_dl*     np.exp(1j*K.dot(-self.A1+self.A2))+
                                 self.V_01_dr*  np.exp(1j*K.dot(+self.A1+self.A2))+
                                 self.V_01_ld.conj().T*  np.exp(1j*K.dot(-self.A1-self.A2))+
                                 self.V_01_rd.conj().T*  np.exp(1j*K.dot(+self.A1-self.A2)))
            e,v=np.linalg.eigh(H_k)
            eigs[it,:] = e
            it+=1
        return path,eigs
    
    def Add_CAP(self,scale=1,c=1,remove=False,offset=1,Type = '??',Manual_L = '', Manual_R = ''):
        xl1 = self.pos_l[:,0].min() - offset
        xl0 = (self.pos_d[:,0].min()+self.pos_l[:,0].max())/2
        xr0 = (self.pos_d[:,0].max()+self.pos_r[:,0].min())/2
        xr1 =  self.pos_r[:,0].max() + offset
        if isinstance(Manual_L,str):
            Wl = np.diag(1j*W(self.pos_l[:,0],xl0,xl1,c,Type = Type))*scale
        else:
            Wl = Manual_L*scale
        if isinstance(Manual_R,str):
            Wr = np.diag(1j*W(self.pos_r[:,0],xr0,xr1,c,Type = Type))*scale
        else:
            Wr = Manual_R*scale
        
        p=1
        if remove==True: p=-1
        self.Wl = Wl
        self.Wr = Wr
        self.H_ll += p*Wl
        self.H_rr += p*Wr
    
    def Add_Ramp(self,Vl,Vr,remove=False):
        x_min=self.pos_d[:,0].min()
        x_max=self.pos_d[:,0].max()
        V=Vl+0j+(Vr-Vl)*(self.pos_d[:,0]-x_min)/(x_max-x_min)
        Ramp=np.diag(V)
        Potl=np.diag(Vl*np.ones(len(self.pos_l)))
        Potr=np.diag(Vr*np.ones(len(self.pos_r)))
        
        if remove==False:
            p=1
        else:
            p=-1
        
        self.H_dd+=p*Ramp
        self.H_ll += p*Potl
        self.H_rr += p*Potr
        self.CAP_ADDED=True
    
    def Gen_SE(self, method='decimation',
               eat_ram=False,
               calc_der=False,
               step = 1e-5,
               tol=1e-12,
               CAP_offset = 1,
               CAP_c=1,
               CAP_scale=1,
               Type='??',
               Manual_R='',
               Manual_L=''):
        print('\n ESTIMATED RAM IN GB:    '+str(np.round(self.RAM(),3))+'\n')
        if self.RAM()>9:
            print('\n Over 9gb ram is estimated to be required.\n Or split calculation over more batches of k-points.')
            if eat_ram==False:
                assert 1==0
        self.se_l   = CZ((self.nk,len(self.E),self.H_dd.shape[0],self.H_dd.shape[1]))
        self.se_r   = CZ((self.nk,len(self.E),self.H_dd.shape[0],self.H_dd.shape[1]))
        self._gsl   = CZ((self.nk,len(self.E),self.H_ll.shape[0],self.H_ll.shape[1]))
        self._gsr   = CZ((self.nk,len(self.E),self.H_rr.shape[0],self.H_rr.shape[1]))
        
        if calc_der==True:
            self.se_l_der = CZ((self.nk,len(self.E),self.H_dd.shape[0],self.H_dd.shape[1]))
            self.se_r_der = CZ((self.nk,len(self.E),self.H_dd.shape[0],self.H_dd.shape[1]))
        
        Il = np.eye(self.H_ll.shape[0])
        Ir = np.eye(self.H_rr.shape[0])
        if method=='decimation':
            self.Calc_Method='decimation'
            print('\n----Calculating surface greens functions by recursion -----\n')
            for j in tqdm(range(self.nk)):
                phase = self.phases[j]
                H_l   = self.H_ll      + self.V_01_ll * phase + self.V_01_ll .T.conj()  * phase.conj()
                H_r   = self.H_rr      + self.V_01_rr * phase + self.V_01_rr .T.conj()  * phase.conj()
                V_lL  = self.V_lL      + self.V_lL_01 * phase + self.V_lL_0m1           * phase.conj()
                V_rR  = self.V_rR      + self.V_rR_01 * phase + self.V_rR_0m1           * phase.conj()
                V_dl  = self.V_00_dl   + self.V_01_dl * phase + self.V_01_ld .T.conj()  * phase.conj()
                V_dr  = self.V_00_dr   + self.V_01_dr * phase + self.V_01_rd .T.conj()  * phase.conj()
                SE_Lead_L = SE_vectorised(self.E, H_l, V_lL ,eta=self.eta, eps=tol)   #[SE(e, H_l, V_lL ,eta=self.eta, eps=tol) for e in self.E]
                SE_Lead_R = SE_vectorised(self.E, H_r, V_rR ,eta=self.eta ,eps=tol)   #[SE(e, H_r, V_rR ,eta=self.eta ,eps=tol) for e in self.E]
                if calc_der == True:
                    SE_Lead_L_PLUS = SE_vectorised(self.E+step, H_l, V_lL ,eta=self.eta, eps=tol)
                    SE_Lead_R_PLUS = SE_vectorised(self.E+step, H_r, V_rR ,eta=self.eta ,eps=tol)
                for i,e in enumerate(self.E):
                    gsl = Inv(Il*(e+1j*self.eta)-H_l-SE_Lead_L[i])
                    gsr = Inv(Ir*(e+1j*self.eta)-H_r-SE_Lead_R[i])
                    
                    self.se_l[j,i,:,:] = UAU_D(V_dl,gsl)
                    self.se_r[j,i,:,:] = UAU_D(V_dr,gsr)
                    # self.se_l_a[j,i,:,:] = UAU_D(V_dl,Dag(gsl))
                    # self.se_r_a[j,i,:,:] = UAU_D(V_dr,Dag(gsr))
                    
                    if calc_der == True:
                        gsl_PLUS  = Inv(Il*(e+step+1j*self.eta)-H_l-SE_Lead_L_PLUS[i])
                        gsr_PLUS  = Inv(Ir*(e+step+1j*self.eta)-H_r-SE_Lead_R_PLUS[i])
                        SE_L_PLUS = UAU_D(V_dl,gsl_PLUS)
                        SE_R_PLUS = UAU_D(V_dr,gsr_PLUS)
                        self.se_l_der[j,i]  = (SE_L_PLUS-self.se_l[j,i,:,:])/step
                        self.se_r_der[j,i]  = (SE_R_PLUS-self.se_r[j,i,:,:])/step
                    
        elif method=='CAP':
            self.Calc_Method='CAP'
            print('\n------ Calculating greens function with CAP potential -----\n')
            nl,nd,nr=self.H_ll.shape[0],self.H_dd.shape[0],self.H_rr.shape[0]
            if self.CAP_ADDED ==False:
                self.Add_CAP(offset=CAP_offset,c=CAP_c,scale=CAP_scale,Type=Type,Manual_L=Manual_L,Manual_R=Manual_R)
            i_l=slice(0,nl)
            i_d=slice(nl,nl+nd)
            i_r=slice(nl+nd,nl+nd+nr)
            #### CAP ER TILFØJET I H_ll og H_rr i "self.Add_CAP"!!!!#####
            H_new=CZ((nl+nd+nr,nl+nd+nr))
            H_new[i_l,i_l] = self.H_ll.copy()
            H_new[i_d,i_d] = self.H_dd.copy()
            H_new[i_r,i_r] = self.H_rr.copy()
            H_new[i_d,i_l] = self.V_00_dl.copy()
            H_new[i_d,i_r] = self.V_00_dr.copy()
            H_new[i_l,i_d] = Dag(self.V_00_dl).copy()
            H_new[i_r,i_d] = Dag(self.V_00_dr).copy()
            self.H_cap=H_new
            
            V_01_new=CZ((nl+nd+nr,nl+nd+nr))
            V_01_new[i_l,i_l] = self.V_01_ll.copy()
            V_01_new[i_d,i_d] = self.V_01_dd.copy()
            V_01_new[i_r,i_r] = self.V_01_rr.copy()
            V_01_new[i_d,i_l] = self.V_01_dl.copy()
            V_01_new[i_d,i_r] = self.V_01_dr.copy()
            V_01_new[i_l,i_d] = self.V_01_ld.copy()
            V_01_new[i_r,i_d] = self.V_01_rd.copy()
            self.V_cap=V_01_new
            self.CAP_LEFT  = CZ(H_new.shape); self.CAP_LEFT [i_l,i_l] += self.Wl.copy()
            self.CAP_RIGHT = CZ(H_new.shape); self.CAP_RIGHT[i_r,i_r] += self.Wr.copy()
    
    def Average_Gamma(self,tol=1e-4,hw = 0,mu_l = 0, mu_r=0,check_hermicity=True):
        from scipy.integrate import simps
        if hw==0:fd = np.ones(len(self.E))
        else:    fd = f(self.E+hw,mu=mu_l)-f(self.E,mu=mu_r)
        
        self.avg_gamma_l_e_v=[]
        self.avg_gamma_r_e_v=[]
        self._avg_gamma_l=[]
        self._avg_gamma_r=[]
        for j in range(self.nk):
            if self.Calc_Method=='decimation':
                avg_l = simps(fd*Gamma(self.se_l[j,:,:,:]).transpose(1,2,0),self.E,even='last')/(self.E.max()-self.E.min())
                avg_r = simps(fd*Gamma(self.se_r[j,:,:,:]).transpose(1,2,0),self.E,even='last')/(self.E.max()-self.E.min())
                self._avg_gamma_l += [avg_l]
                self._avg_gamma_r += [avg_r]
                el,vl = np.linalg.eigh(avg_l)
                er,vr = np.linalg.eigh(avg_r)
                i_l=np.where(el>tol)[0]
                i_r=np.where(er>tol)[0]
                el=el[  i_l]
                vl=vl[:,i_l]
                er=er[  i_r]
                vr=vr[:,i_r]
                self.avg_gamma_l_e_v+=[[el.copy(),vl.copy()]]
                self.avg_gamma_r_e_v+=[[er.copy(),vr.copy()]]
            elif self.Calc_Method=='CAP':
                Gl = Gamma(self.CAP_LEFT.copy() )
                Gr = Gamma(self.CAP_RIGHT.copy())
                self._avg_gamma_l += [Gl]
                self._avg_gamma_r += [Gr]
                el,vl = np.linalg.eigh(Gl)
                er,vr = np.linalg.eigh(Gr)
                i_l=np.where(np.abs(el)>tol)[0]
                i_r=np.where(np.abs(er)>tol)[0]
                el=el[  i_l]
                vl=vl[:,i_l]
                er=er[  i_r]
                vr=vr[:,i_r]
                self.avg_gamma_l_e_v+=[[el.copy(),vl.copy()]]
                self.avg_gamma_r_e_v+=[[er.copy(),vr.copy()]]
    
    def Gen_device_G(self):
        print('\n------------Calculating Device green\'s function------------\n')
        if self.Calc_Method=='decimation':
            V_d = self.V_01_dd.copy()
            h_d = self.H_dd.copy()
        elif self.Calc_Method=='CAP':
            V_d = self.V_cap.copy()
            h_d = self.H_cap.copy()
        s=h_d.shape
        self.GD     = CZ((self.nk,len(self.E    ),s[0],s[1]))
        Is=np.eye(s[0])
        
        for j in range(self.nk):
            phase= self.phases[j]
            H_dev_k = h_d + V_d * phase + V_d.T.conj() * phase.conj()
            if self.Calc_Method=='decimation':
                for i,e in enumerate(self.E):
                    self.GD[j,i,:,:] = Inv( Is*(e+1j*self.eta)
                                               - H_dev_k
                                               - self.se_l[j,i]
                                               - self.se_r[j,i]
                                               )
            elif self.Calc_Method=='CAP':
                for i,e in enumerate(self.E):
                    self.GD[j,i,:,:] = Inv( Is*(e+1j*self.eta)
                                               - H_dev_k
                                               )
    
    def DOS(self,k_resolved=False):
        DOS=np.zeros(len(self.E))
        if k_resolved==False:
            DOS=np.zeros(len(self.E))
            for i in range(len(self.E)):
                for j in range(self.nk):
                    DOS[i] += -np.imag(Tr(self.GD[j,i,:,:]))/self.nk/np.pi
        elif k_resolved==True:
            DOS=np.zeros((self.nk,len(self.E)))
            for i in range(len(self.E)):
                for j in range(self.nk):
                    DOS[j,i] = -np.imag(Tr(self.GD[j,i,:,:]))/np.pi
        return DOS
    
    def Spectral_DOS(self,k_resolved=False):
        if k_resolved==True:
            ADOS=np.zeros((self.nk,len(self.E),2))
        else:
            ADOS=np.zeros((len(self.E),2))
        for j in range(self.nk):
            for i in range(len(self.E)):
                G0 = self.GD[j,i,:,:]
                if k_resolved==False:
                    ADOS[i,0]+= -1/np.pi*np.imag(Tr(G0.dot(Gamma(self.se_l[j,i,:,:])).dot(Dag(G0))))
                    ADOS[i,1]+= -1/np.pi*np.imag(Tr(G0.dot(Gamma(self.se_r[j,i,:,:])).dot(Dag(G0))))
                else:
                    ADOS[j,i,0]=-1/np.pi*np.imag(Tr(self.GD[j,i,:,:].dot(Gamma(self.se_l[j,i,:,:])).dot(self.GD[j,i,:,:].conj().T)))
                    ADOS[j,i,1]=-1/np.pi*np.imag(Tr(self.GD[j,i,:,:].dot(Gamma(self.se_r[j,i,:,:])).dot(self.GD[j,i,:,:].conj().T)))
        return ADOS
    
    def Transport_vectorised(self, hw = 0,k_resolved=False,method='ShetWain_A1',mu = [0,0],kT = [0.025,0.025]):
        T = CZ((self.nk,len(self.inds_cent),2,2))  
        Y = CZ((self.nk,len(self.inds_cent),2))
        
        if hw>0 and method=='ShetWain_A1' and self.Calc_Method=='decimation':
            T1 = CZ((self.nk,len(self.inds_cent),2,2))
            T2 = CZ((self.nk,len(self.inds_cent),2,2))
            T3 = CZ((self.nk,len(self.inds_cent),2,2))
            D1 = CZ((self.nk,len(self.inds_cent),2  ))
            D2 = CZ((self.nk,len(self.inds_cent),2  ))
            
            Tr_G1 = CZ((self.nk,len(self.inds_cent),2  ))
            Tr_G2 = CZ((self.nk,len(self.inds_cent),2  ))
            
            #Man kan også kalde n_2hw for eksempel, bare husk at sætte multiple_absorption til 2 i starten så.
            n_hw = np.round(hw/self.dE,4)
            hw_i,wgt = self.shift_ind(n_hw)
            #i0 is non-shifted indecies
            i0 = self.inds_cent
            
            half_n_hw_p      = np.round( hw/self.dE/2,4)
            half_n_hw_m      = np.round(-hw/self.dE/2,4)
            Hhw_p_i,Hhw_p_w  = self.shift_ind(half_n_hw_p)
            Hhw_m_i,Hhw_m_w  = self.shift_ind(half_n_hw_m)
            def I_ws(Array):
                # Interpolation for E + hw
                # Takes array with energy and orbital indecies ~ A_{E,i,j}
                if wgt[0]==1 and wgt[1]==0:
                    return Array[hw_i[0]]
                elif wgt[1]==1 and wgt[0]==0:
                    return Array[hw_i[1]]
                else:
                    return Array[hw_i[0]]*wgt[0] + Array[hw_i[1]]*wgt[1]
            
            def I_phs(Array):
                w0,w1     = Hhw_p_w[0],Hhw_p_w[1]
                ids0,ids1 = Hhw_p_i[0],Hhw_p_i[1]
                # Interpolation for +hw/2 (Interpolation plus half step)
                # Takes array with energy and orbital indecies ~ A_{E,i,j}
                if w0 == 1 and w1 == 0:
                    return Array[ids0]
                elif w1 == 1 and w0 == 0:
                    return Array[ids1]
                else:
                    return Array[ids0]*w0 + Array[ids1]*w1
            
            def I_mhs(Array):
                w0,w1     = Hhw_m_w[0],Hhw_m_w[1]
                ids0,ids1 = Hhw_m_i[0],Hhw_m_i[1]
                # Interpolation for -hw/2 (Interpolation minus half step)
                # Takes array with energy and orbital indecies ~ A_{E,i,j}
                if   w0 == 1 and w1 == 0:
                    return Array[ids0]
                elif w0 == 0 and w1 == 1:
                    return Array[ids1]
                else:
                    return Array[ids0]*w0 + Array[ids1]*w1
            
            for j in range(self.nk):
                # Conduction current terms eq. A1 i Shevtsov & Waintal
                # Writing A[j] picks out the components A[j,:,:,:]
                # Left to right
                SE={0:self.se_l[j],1:self.se_r[j]}
                G = self.GD[j]
                for m0,m1 in IT.product(range(2),range(2)):
                    T1[j,:,m0,m1] = Tr_Prod( MM(Dag ( SE[m0][i0] )  -  I_ws( SE[m0]     ) , I_ws( G     )  ) ,
                                              MM(I_ws( SE[m1]     )  -  Dag(  SE[m1][i0]  ) , Dag (G[i0] )  )  )
                
                    T2[j,:,m0,m1] = Tr_Prod( MM(       SE[m0][i0]    -  I_ws( SE[m0]     ) , I_ws( G    )  ) ,
                                              MM( I_ws( SE[m1])       -        SE[m1][i0]   ,       G[i0]   )  )
                
                    T3[j,:,m0,m1] = Tr_Prod( MM(      Dag(SE[m0][i0])  -  Dag(I_ws(SE[m0] )   ) ,  Dag(I_ws( G    ) ) ) ,
                                              MM( I_ws(Dag(SE[m1]    )) -  Dag(     SE[m1][i0] ) ,  Dag(      G[i0])   )    )
                
                ## Terms with delta_ij on them 
                for m0 in range(2):
                    D1[j,:,m0]   = (1j*Tr_Prod( I_ws(G    )    -    Dag(G[i0])  ,Gamma( SE[m0][i0]                ) )
                                    +  Tr_Prod(      G[i0]     -    Dag(G[i0])  ,       SE[m0][i0] - I_ws( SE[m0] ) )  )
                    
                    D2[j,:,m0]   = (1j*Tr_Prod( I_ws(G)        -    Dag(G[i0])  ,Gamma(I_ws(SE[m0])) )
                                    +  Tr_Prod( I_ws(G)        -   I_ws(Dag(G)) ,   Dag(SE[m0][i0]) -  Dag(I_ws(SE[m0])) ) )
                
                # Displacement current terms, Calculated from G^{<}_{±1} components (eq. 85) and small first order expansion in (eV_ac)
                # https://photos.app.goo.gl/birPBcCYuignWUhk9
                # Strukturen af alle ledene følger de første fem ligninger herefter
                # ...........
                # Tr[G^<_+1] & Perturbation in left lead:
                Tr_G1[j,:,0]+=Tr_Prod(MMM(    I_phs(self.GD  [j   ])   ,      I_mhs(self.se_l[j]     )        -    I_phs(self.se_l[j]) , 
                                              I_mhs(self.GD  [j   ])    )                                                              ,
                                   MM(Gmmlssr(I_mhs(self.se_l[j   ])   ,      I_mhs(self.E),mu=mu[0],kT=kT[0]),Dag(I_mhs(self.GD  [j]))   ) )
                
                Tr_G1[j,:,0]+=Tr_Prod(
                                    MM(       I_phs(self.GD  [j]  ) ,
                                      Gmmlssr(I_mhs(self.se_l[j]  ) , I_mhs(self.E),mu = mu[0],kT = kT[0])
                                     -Gmmlssr(I_phs(self.se_l[j]  ) , I_phs(self.E),mu = mu[0],kT = kT[0])
                                      )
                                       , 
                                          Dag(I_mhs(self.GD  [j]  ) )
                                     )
                
                Tr_G1[j,:,0]+=Tr_Prod(
                                   MMM(       I_phs(self.GD  [j])                                   ,
                                      Gmmlssr(I_phs(self.se_l[j]),       I_phs(self.E),mu=mu[0],kT=kT[0])  ,
                                          Dag(I_phs(self.GD  [j]) )
                                      )
                                      ,
                                    MM(   Dag(I_mhs(self.se_l[j])) - Dag(I_phs(self.se_l[j])),
                                          Dag(I_mhs(self.GD  [j]))
                                      )
                                     )
                Tr_G1[j,:,0]+=Tr_Prod(MMM(    I_phs(self.GD  [j   ])   ,      I_mhs(self.se_l[j]     )        -    I_phs(self.se_l[j]) , 
                                              I_mhs(self.GD  [j   ])    )                                                              ,
                                   MM(Gmmlssr(I_mhs(self.se_r[j   ])   ,      I_mhs(self.E),mu=mu[1],kT=kT[1]),
                                          Dag(I_mhs(self.GD  [j]))      ) 
                                     )
                
                Tr_G1[j,:,0]+=Tr_Prod(
                                   MMM(       I_phs(self.GD  [j])                                   ,
                                      Gmmlssr(I_phs(self.se_r[j]),       I_phs(self.E),mu=mu[1],kT=kT[1])  ,
                                          Dag(I_phs(self.GD  [j]) )
                                      )
                                      ,
                                    MM(   Dag(I_mhs(self.se_l[j])) - Dag(I_phs(self.se_l[j])),
                                          Dag(I_mhs(self.GD  [j]))
                                      )
                                     )
                # Tr[G^<_-1] & Perturbation in left lead:
                Tr_G2[j,:,0]+=Tr_Prod(MMM(    I_mhs(self.GD  [j   ])   ,      I_mhs(self.se_l[j]     )        -    I_phs(self.se_l[j]) , 
                                              I_phs(self.GD  [j   ])    )                                                              ,
                                   MM(Gmmlssr(I_phs(self.se_l[j   ])   ,      I_phs(self.E),mu=mu[0],kT=kT[0]),Dag(I_phs(self.GD  [j]))   ) )
                
                Tr_G2[j,:,0]+=Tr_Prod(
                                    MM(       I_mhs(self.GD  [j]  ) ,
                                      Gmmlssr(I_mhs(self.se_l[j]  ) , I_mhs(self.E),mu = mu[0],kT = kT[0])
                                     -Gmmlssr(I_phs(self.se_l[j]  ) , I_phs(self.E),mu = mu[0],kT = kT[0])
                                      )
                                       , 
                                          Dag(I_phs(self.GD  [j]  ) )
                                     )
                
                Tr_G2[j,:,0]+=Tr_Prod(
                                   MMM(       I_mhs(self.GD  [j])                                   ,
                                      Gmmlssr(I_mhs(self.se_l[j]),       I_mhs(self.E),mu=mu[0],kT=kT[0])  ,
                                          Dag(I_mhs(self.GD  [j]) )
                                      )
                                      ,
                                    MM(   Dag(I_mhs(self.se_l[j])) - Dag(I_phs(self.se_l[j])),
                                          Dag(I_phs(self.GD  [j]))
                                      )
                                     )
                
                Tr_G2[j,:,0]+=Tr_Prod(MMM(    I_mhs(self.GD  [j   ])   ,      I_mhs(self.se_l[j]     )        -    I_phs(self.se_l[j]) , 
                                              I_phs(self.GD  [j   ])    )                                                              ,
                                   MM(Gmmlssr(I_phs(self.se_r[j   ])   ,      I_phs(self.E),mu=mu[1],kT=kT[1]),
                                          Dag(I_phs(self.GD  [j]))      ) 
                                     )
                
                Tr_G2[j,:,0]+=Tr_Prod(
                                   MMM(       I_mhs(self.GD  [j])                                   ,
                                      Gmmlssr(I_mhs(self.se_r[j]),       I_mhs(self.E),mu=mu[1],kT=kT[1])  ,
                                          Dag(I_mhs(self.GD  [j]) )
                                      )
                                      ,
                                    MM(   Dag(I_mhs(self.se_l[j])) - Dag(I_phs(self.se_l[j])),
                                          Dag(I_phs(self.GD  [j]))
                                      )
                                     )
                
                #######...............########
                # Tr[G^<_+1] & Perturbation in right lead:
                # se_l -> se_r & se_r -> se_l etc.
                
                Tr_G1[j,:,1]+=Tr_Prod(MMM(    I_phs(self.GD  [j   ])   ,      I_mhs(self.se_r[j]     )        -    I_phs(self.se_r[j]) , 
                                              I_mhs(self.GD  [j   ])    )                                                              ,
                                   MM(Gmmlssr(I_mhs(self.se_r[j   ])   ,      I_mhs(self.E),mu=mu[1],kT=kT[1]),Dag(I_mhs(self.GD  [j]))   ) )
                
                Tr_G1[j,:,1]+=Tr_Prod(
                                    MM(       I_phs(self.GD  [j]  ) ,
                                      Gmmlssr(I_mhs(self.se_r[j]  ) , I_mhs(self.E),mu = mu[1],kT = kT[1])
                                     -Gmmlssr(I_phs(self.se_r[j]  ) , I_phs(self.E),mu = mu[1],kT = kT[1])
                                      )
                                        , 
                                          Dag(I_mhs(self.GD  [j]  ) )
                                      )
                
                Tr_G1[j,:,1]+=Tr_Prod(
                                   MMM(       I_phs(self.GD  [j])                                   ,
                                      Gmmlssr(I_phs(self.se_r[j]),       I_phs(self.E),mu=mu[1],kT=kT[1])  ,
                                          Dag(I_phs(self.GD  [j]) )
                                      )
                                      ,
                                    MM(   Dag(I_mhs(self.se_r[j])) - Dag(I_phs(self.se_r[j])),
                                          Dag(I_mhs(self.GD  [j]))
                                      )
                                      )
                Tr_G1[j,:,1]+=Tr_Prod(MMM(    I_phs(self.GD  [j   ])   ,      I_mhs(self.se_r[j]     )        -    I_phs(self.se_r[j]) , 
                                              I_mhs(self.GD  [j   ])    )                                                              ,
                                   MM(Gmmlssr(I_mhs(self.se_l[j   ])   ,      I_mhs(self.E),mu=mu[0],kT=kT[0]),
                                          Dag(I_mhs(self.GD  [j]))      ) 
                                      )
                
                Tr_G1[j,:,1]+=Tr_Prod(
                                   MMM(       I_phs(self.GD  [j])                                   ,
                                      Gmmlssr(I_phs(self.se_l[j]),       I_phs(self.E),mu=mu[0],kT=kT[0])  ,
                                          Dag(I_phs(self.GD  [j]) )
                                      )
                                      ,
                                    MM(   Dag(I_mhs(self.se_r[j])) - Dag(I_phs(self.se_r[j])),
                                          Dag(I_mhs(self.GD  [j]))
                                      )
                                      )
                # # Tr[G^<_-1] & Perturbation in right lead:
                Tr_G2[j,:,1]+=Tr_Prod(MMM(    I_mhs(self.GD  [j   ])   ,      I_mhs(self.se_r[j]     )        -    I_phs(self.se_r[j]) , 
                                              I_phs(self.GD  [j   ])    )                                                              ,
                                    MM(Gmmlssr(I_phs(self.se_r[j   ])   ,      I_phs(self.E),mu=mu[1],kT=kT[1]),Dag(I_phs(self.GD  [j]))   ) )
                
                Tr_G2[j,:,1]+=Tr_Prod(
                                    MM(       I_mhs(self.GD  [j]  ) ,
                                      Gmmlssr(I_mhs(self.se_r[j]  ) , I_mhs(self.E),mu = mu[1],kT = kT[1])
                                     -Gmmlssr(I_phs(self.se_r[j]  ) , I_phs(self.E),mu = mu[1],kT = kT[1])
                                      )
                                        , 
                                          Dag(I_phs(self.GD  [j]  ) )
                                      )
                
                Tr_G2[j,:,1]+=Tr_Prod(
                                    MMM(       I_mhs(self.GD  [j])                                   ,
                                       Gmmlssr(I_mhs(self.se_r[j]),       I_mhs(self.E),mu=mu[1],kT=kT[1])  ,
                                           Dag(I_mhs(self.GD  [j]) )
                                      )
                                      ,
                                     MM(   Dag(I_mhs(self.se_r[j])) - Dag(I_phs(self.se_r[j])),
                                           Dag(I_phs(self.GD  [j]))
                                       )
                                     )
                
                Tr_G2[j,:,1]+=Tr_Prod(MMM(     I_mhs(self.GD  [j   ])   ,      I_mhs(self.se_r[j]     )        -    I_phs(self.se_r[j]) , 
                                               I_phs(self.GD  [j   ])    )                                                              ,
                                    MM(Gmmlssr(I_phs(self.se_l[j   ])   ,      I_phs(self.E),mu=mu[0],kT=kT[0]),
                                          Dag(I_phs(self.GD  [j]))      ) 
                                      )
                
                Tr_G2[j,:,1]+=Tr_Prod(
                                    MMM(       I_mhs(self.GD  [j])                                   ,
                                       Gmmlssr(I_mhs(self.se_l[j]),       I_mhs(self.E),mu=mu[0],kT=kT[0])  ,
                                           Dag(I_mhs(self.GD  [j]) )
                                      )
                                      ,
                                     MM(   Dag(I_mhs(self.se_r[j])) - Dag(I_phs(self.se_r[j])),
                                           Dag(I_phs(self.GD  [j]))
                                       )
                                      )
            
            self.T1    = T1.sum(axis=0)
            self.T2    = T2.sum(axis=0)
            self.T3    = T3.sum(axis=0)
            self.D1    = D1.sum(axis=0)
            self.D2    = D2.sum(axis=0)
            self.Tr_G_plus1  = Tr_G1.sum(axis=0)/2
            self.Tr_G_minus1 = Tr_G2.sum(axis=0)/2
            
            
            
    
    def Transport(self, hw = 0,k_resolved=False,avg_gamma='slow'):
        T = CZ((self.nk,self.E.shape[0],2,2))  
        Y = CZ((self.nk,self.E.shape[0],2))
        if hw == 0:
            #print('\n----- DC Transport ------\n')
            for j in range(self.nk):
                for i,e in enumerate(self.E):
                    if self.Calc_Method=='decimation':
                        G0    =       self.GD  [j,i,:,:]
                        Gl_G  = Gamma(self.se_l[j,i,:,:]).dot(G0)
                        Gr_G  = Gamma(self.se_r[j,i,:,:]).dot(G0)
                        Gl_Gd = Gamma(self.se_l[j,i,:,:]).dot(Dag(G0))
                        Gr_Gd = Gamma(self.se_r[j,i,:,:]).dot(Dag(G0))
                        T[j,i,1,0] = Tr_Prod(Gr_G,Gl_Gd)
                        T[j,i,0,1] = Tr_Prod(Gl_G,Gr_Gd)
                        T[j,i,0,0] = Tr_Prod(Gl_G,Gl_Gd)
                        T[j,i,1,1] = Tr_Prod(Gr_G,Gr_Gd)
                    if self.Calc_Method=='CAP':
                        G0    =       self.GD  [j,i,:,:]
                        Gl_G  = Gamma(self.CAP_LEFT ).dot(G0)
                        Gr_G  = Gamma(self.CAP_RIGHT).dot(G0)
                        Gl_Gd = Gamma(self.CAP_LEFT ).dot(Dag(G0))
                        Gr_Gd = Gamma(self.CAP_RIGHT).dot(Dag(G0))
                        T[j,i,1,0] = Tr_Prod(Gr_G,Gl_Gd)
                        T[j,i,0,1] = Tr_Prod(Gl_G,Gr_Gd)
                        T[j,i,0,0] = Tr_Prod(Gl_G,Gl_Gd)
                        T[j,i,1,1] = Tr_Prod(Gr_G,Gr_Gd)
            if k_resolved==False:
                self.T = T.sum(axis=0)
            else:
                self.T = T
        
        elif hw>0 and avg_gamma=='slow':
            #print('\n----- AC Transport, Gamma averaged, slow version------\n' )
            if self.Calc_Method=='decimation':
                s = self.H_dd.shape[0]
                Is = np.eye(s)
                hop_dev = self.V_01_dd
                h_dd    = self.H_dd
            elif self.Calc_Method=='CAP':
                s = self.H_cap.shape[0]
                Is = np.eye(s)
                hop_dev = self.V_cap
                h_dd    = self.H_cap
            
            for j in range(self.nk):
                phase= self.phases[j]
                H_dev_k = h_dd + hop_dev * phase + hop_dev.T.conj() * phase.conj() 
                for i,e in enumerate(self.E):
                    if self.Calc_Method=='decimation':
                        Sigma=self.se_l[j,i]+self.se_r[j,i]
                    elif self.Calc_Method=='CAP':
                        #Sigma already in H_cap
                        Sigma=CZ(H_dev_k.shape)
                    G0 = self.GD[j,i,:,:]
                    Ghw    = Inv(Is*(e+hw+1j*self.eta)-H_dev_k-Sigma)
                    
                    Ghw_Gl = Ghw.dot    ( self._avg_gamma_l[j] )
                    Ghw_Gr = Ghw.dot    ( self._avg_gamma_r[j] )
                    Gc_Gl  = Dag(G0).dot( self._avg_gamma_l[j] )
                    Gc_Gr  = Dag(G0).dot( self._avg_gamma_r[j] )
                    
                    T[j,i,1,0] =     Tr_Prod(Ghw_Gl,Gc_Gr  )
                    T[j,i,0,1] =     Tr_Prod(Ghw_Gr,Gc_Gl  )
                    T[j,i,0,0] =     Tr_Prod(Ghw_Gl,Gc_Gl  ) - 1j* (Tr(Ghw_Gl) - Tr_Prod(self._avg_gamma_l[j],Dag(G0)))
                    T[j,i,1,1] =     Tr_Prod(Ghw_Gr,Gc_Gr  ) - 1j* (Tr(Ghw_Gr) - Tr_Prod(self._avg_gamma_r[j],Dag(G0)))
                    Y[j,i,0]   = -1j*Tr_Prod(Ghw_Gl,Dag(G0))
                    Y[j,i,1]   = -1j*Tr_Prod(Ghw_Gr,Dag(G0))
            
            self.Y = Y.sum(axis=0)
            self.T = T.sum(axis=0)
        
        elif hw>0 and avg_gamma=='fast':
            if self.Calc_Method=='decimation':
                s = self.H_dd.shape[0]
                Is = np.eye(s)
                hop_dev = self.V_01_dd
                h_dd    = self.H_dd
            elif self.Calc_Method=='CAP':
                s = self.H_cap.shape[0]
                Is = np.eye(s)
                hop_dev = self.V_cap
                h_dd    = self.H_cap
            
            for j in range(self.nk):
                phase= self.phases[j]
                H_dev_k = h_dd + hop_dev * phase + hop_dev.T.conj() * phase.conj() 
                
                el = np.diag(self.avg_gamma_l_e_v[j][0])
                vl =         self.avg_gamma_l_e_v[j][1]
                
                er = np.diag(self.avg_gamma_r_e_v[j][0])
                vr =         self.avg_gamma_r_e_v[j][1]
                
                for i,e in enumerate(self.E):
                    G0   = self.GD[j,i]
                    if self.Calc_Method=='decimation':
                        Sigma = self.se_l[j,i]+self.se_r[j,i]
                    elif self.Calc_Method=='CAP':
                        Sigma = self.CAP_LEFT+self.CAP_RIGHT
                    Ghw  = Inv(Is*(e+hw+1j*self.eta)-H_dev_k)
                    
                    el_vld_Ghw_vr = el.dot( Dag(vl).dot(Ghw).dot(vr) )
                    er_vrd_Ghw_vl = er.dot( Dag(vr).dot(Ghw).dot(vl) )
                    
                    el_vld_G0d_vr = el.dot( Dag(vl).dot(Dag(G0)).dot(vr) )
                    er_vrd_G0d_vl = er.dot( Dag(vr).dot(Dag(G0)).dot(vl) )
                    
                    el_vld_Ghw_vl = el.dot( Dag(vl).dot(Ghw).dot(vl))
                    er_vrd_Ghw_vr = er.dot( Dag(vr).dot(Ghw).dot(vr))
                    
                    el_vld_G0d_vl = el.dot( Dag(vl).dot(Dag(G0)).dot(vl) )
                    er_vrd_G0d_vr = er.dot( Dag(vr).dot(Dag(G0)).dot(vr) )
                    
                    T[j,i,1,0] =     Tr_Prod(el_vld_Ghw_vr,er_vrd_G0d_vl)
                    T[j,i,0,1] =     Tr_Prod(er_vrd_Ghw_vl,el_vld_G0d_vr)
                    T[j,i,0,0] =     Tr_Prod(el_vld_Ghw_vl,el_vld_G0d_vl) - 1j* ( Tr(el_vld_Ghw_vl)
                                                                                 -Tr(el_vld_G0d_vl) )
                    T[j,i,1,1] =     Tr_Prod(er_vrd_Ghw_vr,er_vrd_G0d_vr) - 1j* ( Tr(er_vrd_Ghw_vr)
                                                                                 -Tr(er_vrd_G0d_vr) )
                    
                    Y[j,i,0]   = -1j*Tr_Prod( Ghw.dot(vl) , el.dot(Dag(vl)).dot(Dag(G0)) )
                    Y[j,i,1]   = -1j*Tr_Prod( Ghw.dot(vr) , er.dot(Dag(vr)).dot(Dag(G0)) )
                    
            self.Y = Y.sum(axis=0)
            self.T = T.sum(axis=0)
    
    def Visualise(self,TreD=False,size=3.5):
        if TreD==False:
            fig, ax = plt.subplots()
            ax.scatter(self.pos_d[:,0], self.pos_d[:,1],c=[[1,0,0]]*len(self.pos_d),s=size,label='Left Lead')
            ax.scatter(self.pos_l[:,0], self.pos_l[:,1],c=[[0,1,0]]*len(self.pos_l),s=size,label='Right Lead')
            ax.scatter(self.pos_r[:,0], self.pos_r[:,1],c=[[0,0,1]]*len(self.pos_r),s=size,label='Device')
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.pos_d[:,0],
                       self.pos_d[:,1],
                       self.pos_d[:,2], 
                       marker='*',s=size)
            ax.scatter(self.pos_l[:,0],
                       self.pos_l[:,1],
                       self.pos_l[:,2], 
                       marker='P',s=size)
            ax.scatter(self.pos_r[:,0],
                       self.pos_r[:,1],
                       self.pos_r[:,2], 
                       marker='P',s=size)
            ax.set_xlabel('X ')
            ax.set_ylabel('Y ')
            ax.set_zlabel('Z ')
        
        
        
        
## Gammel kode
# def F_Generator_function(self, Internal_perturbation, samples=5,RAM_cap_generator=5,half_precision=True,skip_warning=False):
#         # Ref:
#         # Numerical toolkit for electronic quantum transport at finite frequency 
#         # Oleksii Shevtsov and Xavier Waintal
#         # Page 12 more or less
#         print('----- Calculating the generating function ------')
        
        
#         if (hasattr(self, 'se_l_der') and hasattr(self, 'se_r_der')) or self.Calc_Method=='CAP':
#             pass
#         else:
#             print('Set calc_der to True in the self-energy calculation!\n')
#             if skip_warning==False:
#                 assert 1==0
#         if half_precision: DTYPE=np.complex64
#         else:              DTYPE=np.complex128
#         if self.Calc_Method=='decimation':
#             N_s = self.H_dd.shape[0]
#             F_V_d     =  self.V_01_dd
#             F_h_d     =  self.H_dd
#         elif self.Calc_Method=='CAP':
#             N_s = self.H_cap.shape[0]
#             F_V_d     =  self.V_cap
#             F_h_d     =  self.H_cap
        
#         RAM_est = 2*self.nk*len(self.E)*samples*N_s**2*(2-half_precision)*8*10**-9
#         if RAM_est>RAM_cap_generator:
#             print('\nMake RAM_cap bigger or download more RAM!\n')
#             assert 1==0
#         Id=np.eye(N_s)
#         def F_sym(i):
#             # We are dealing with a function F symmetric in the sense F(z)=F(z^-1)=F(z*) on the unit circle
#             # i.e F is symmetric around the real axis.
#             # For F sampled on the bottom half (Im(z)<0, but is really arbitrary) we can therefore write
#             i_max=samples-1
#             if 0<= i <= i_max:
#                 return i
#             elif i<0:
#                 if abs(i) < i_max:
#                     return abs(i)
#                 else:
#                     print('error, make the F_sym better!')
#                     assert 1==0
#             elif i>samples-1 and i<2*(samples-1):
#                 diff = i - i_max
#                 if diff<0:
#                     print('error, make the F_sym better!')
#                     assert 1==0
#                 else:
#                     return i_max - diff
        
#         def F_ad(C,j,i):
#             phase   =  self.phases[j]
#             H_dev_k =  F_h_d + F_V_d * phase + F_V_d.T.conj() * phase.conj()
#             Res=[]
#             if self.Calc_Method=='decimation':
#                 Sigma = self.se_l[j,i] + self.se_r[j,i]
#             elif self.Calc_Method=='CAP':
#                 #Sigma ligger allerede i H_cap
#                 Sigma = np.zeros(H_dev_k.shape)
#             for z in C:
#                 G_inv=Inv((self.E[i]+1j*self.eta)*Id-H_dev_k
#                           -Sigma
#                           -Internal_perturbation*(z+z**-1)/2)
#                 Res+=[G_inv]
#             return Res
        
#         # This is actually "tilde T = omega * T from the paper", 
#         T        = np.linspace(0,np.pi,samples)
#         Z_Sample = np.exp(-1j*T)
#         dT       = T[1] - T[0]
#         Neo = np.eye(N_s)
#         self.F_term1 = np.zeros((self.nk,len(self.E),samples,N_s,N_s),dtype=DTYPE)
#         self.F_term2 = np.zeros((self.nk,len(self.E),samples,N_s,N_s),dtype=DTYPE)
#         for j in range(self.nk):
#             for i,e in enumerate(self.E):
#                 brackets = Neo - self.se_l_der[j,i] - self.se_r_der[j,i]
#                 F_adiabatic_samples = F_ad(Z_Sample,j,i)
#                 dFdT_samples = []
#                 for I in range(samples):
#                     four_vals=[ F_adiabatic_samples[F_sym(I-2)],
#                                 F_adiabatic_samples[F_sym(I-1)],
#                                 F_adiabatic_samples[F_sym(I+1)],
#                                 F_adiabatic_samples[F_sym(I+2)] ]
#                     dFdT_samples+=[five_point_stencil(four_vals, dT)]
#                 assert len(dFdT_samples)==len(F_adiabatic_samples)
#                 # hbar*omega missing from second term
#                 Second_term = [-1j*F_adiabatic_samples[P].dot(brackets).dot(dFdT_samples[P]) for P in range(samples)]
#                 #Insert in Matrix.
#                 for I in range(samples):
#                     self.F_term1[j,i,I,:,:] = F_adiabatic_samples[I]
#                     self.F_term2[j,i,I,:,:] = Second_term[I]
#     #def G_l_hwl(self,hw,l):
    
# if hw == 0:
#             print('\n----- DC Transport ------\n')
#             if self.Calc_Method=='decimation':
#                 for j in range(self.nk):
#                     Gl_G  = MM(Gamma(self.se_l[j,self.inds_cent]),    self.GD[j,self.inds_cent] )
#                     Gr_G  = MM(Gamma(self.se_r[j,self.inds_cent]),    self.GD[j,self.inds_cent] )
#                     Gl_Gd = MM(Gamma(self.se_l[j,self.inds_cent]),Dag(self.GD[j,self.inds_cent]))
#                     Gr_Gd = MM(Gamma(self.se_r[j,self.inds_cent]),Dag(self.GD[j,self.inds_cent]))
#                     T[j,:,1,0] = Tr_Prod(Gr_G,Gl_Gd)
#                     T[j,:,0,1] = Tr_Prod(Gl_G,Gr_Gd)
#                     T[j,:,0,0] = Tr_Prod(Gl_G,Gl_Gd)
#                     T[j,:,1,1] = Tr_Prod(Gr_G,Gr_Gd)
#             elif self.Calc_Method=='CAP':
#                 for j in range(self.nk):
#                     Gl_G  = MM(Gamma(self.CAP_LEFT ),    self.GD[j,self.inds_cent] )
#                     Gr_G  = MM(Gamma(self.CAP_RIGHT),    self.GD[j,self.inds_cent] )
#                     Gl_Gd = MM(Gamma(self.CAP_LEFT ),Dag(self.GD[j,self.inds_cent]))
#                     Gr_Gd = MM(Gamma(self.CAP_RIGHT),Dag(self.GD[j,self.inds_cent]))
#                     T[j,:,1,0] = Tr_Prod(Gr_G,Gl_Gd)
#                     T[j,:,0,1] = Tr_Prod(Gl_G,Gr_Gd)
#                     T[j,:,0,0] = Tr_Prod(Gl_G,Gl_Gd)
#                     T[j,:,1,1] = Tr_Prod(Gr_G,Gr_Gd)
#             if k_resolved==False:
#                 self.T = T.sum(axis=0)
#             else:
#                 self.T = T
#         elif hw>0 and method == 'const_gamma' and self.Calc_Method == 'CAP':
#             n_hw = np.round(hw/self.dE,4)
#             hw_inds,weight = self.shift_ind(n_hw)
#             if len(hw_inds) == 1:
#                 interpolated = False
#                 hw_inds=hw_inds[0]
#                 weight = weight[0]
#             else:
#                 interpolated = True
#                 hw_inds_0 = hw_inds[0]
#                 hw_inds_1 = hw_inds[1]
#                 weight_0  = weight[0]
#                 weight_1  = weight[1]
#             for j in range(self.nk):
#                 # Læg mærke til indekserne der bliver puttet ind i self.GD
#                 # (k-index, E-index, orbital_i, orbital_j)
#                 if interpolated == False:
#                     Ghw_Gl = MM(self.GD[j,hw_inds  ]*weight  , self._avg_gamma_l[j] )
#                     Ghw_Gr = MM(self.GD[j,hw_inds  ]*weight  , self._avg_gamma_r[j] )
#                 elif interpolated == True:
#                     Ghw_Gl = MM(self.GD[j,hw_inds_0]*weight_0+self.GD[j,hw_inds_1]*weight_1 , self._avg_gamma_l[j] )
#                     Ghw_Gr = MM(self.GD[j,hw_inds_0]*weight_0+self.GD[j,hw_inds_1]*weight_1 , self._avg_gamma_r[j] )
#                 Gc_Gl  = MM(Dag(self.GD[j,self.inds_cent]) , self._avg_gamma_l[j] )
#                 Gc_Gr  = MM(Dag(self.GD[j,self.inds_cent]) , self._avg_gamma_r[j] )
                
#                 T[j,:,1,0] =     Tr_Prod(Ghw_Gl,Gc_Gr  )
#                 T[j,:,0,1] =     Tr_Prod(Ghw_Gr,Gc_Gl  )
#                 T[j,:,0,0] =     Tr_Prod(Ghw_Gl,Gc_Gl  ) - 1j*(  Tr(Ghw_Gl,axis1=1,axis2=2)
#                                                                - Tr_Prod(self._avg_gamma_l[j],Dag(self.GD[j,self.inds_cent]))  )
#                 T[j,:,1,1] =     Tr_Prod(Ghw_Gr,Gc_Gr  ) - 1j*(  Tr(Ghw_Gr,axis1=1,axis2=2)
#                                                                - Tr_Prod(self._avg_gamma_r[j],Dag(self.GD[j,self.inds_cent]))  )
#                 Y[j,:,0]   = -1j*Tr_Prod(Ghw_Gl,Dag(self.GD[j,self.inds_cent]))
#                 Y[j,:,1]   = -1j*Tr_Prod(Ghw_Gr,Dag(self.GD[j,self.inds_cent]))
#             if k_resolved==False:
#                 self.T = T.sum(axis=0)
#                 self.Y = Y.sum(axis=0)
#             else:
#                 self.T = T
#                 self.Y = Y
# def Band_TB(self, k,pc='R'):
#         k=k[0]*self.b1+k[1]*self.b2+k[2]*self.b3
#         if pc=='R':
#             Phase = np.exp(1j*(k[0]*self.R_bloch[:,:,:,0]+k[1]*self.R_bloch[:,:,:,1]+k[2]*self.R_bloch[:,:,:,2]))
#         elif pc=='r':
#             Phase = np.exp(1j*(k[0]*self.Rarr[:,:,:,0]+k[1]*self.Rarr[:,:,:,1]+k[2]*self.Rarr[:,:,:,2]))
#         ham = (self.Harr*Phase).sum(axis=2)
#         return np.linalg.eigh(ham)

# def JaJb_sum(l,l1,k,x,tol = 1e-10,more=10,Print=False):
#     Sum = (l-2*l1-2*0)**k*jv(l1+0,x)*jv(0,x)
#     tæller = 0
#     Term = 10
#     D= l-2*l1
#     n=1
#     while tæller < more:
#         Tp = (D+2*n)**k*jv(l1+n,x)*jv(n,x)
#         Tm = (D-2*n)**k*jv(l1-n,x)*jv(-n,x)
#         Term = Tm+Tp
#         if (np.abs(Term)<tol).all():
#             tæller+=1
#         Sum+=Term
#         n+=1
#     if Print==True:
#         print('Converged with n = ', str(n))
#     return Sum
