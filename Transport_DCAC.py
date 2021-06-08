#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:41:46 2021

@author: aleks
"""

from Helper import SE_vectorised as SEv
from Helper import CZ, W, Thz, x_min_x_max,f, integrate_f_x,f
from Block_matrices import block_td,block_sparse,test_partition_2d_sparse_matrix,Build_BTD,Build_BS, block_TRACE_different_bs
import numpy as np
#import matplotlib.pyplot as plt
from scipy import sparse as sp
from tqdm import tqdm
from time import time
import gc


Inv = np.linalg.inv

def kpoints(n,direction = 1 ): 
    kvec=np.zeros((n,3)); 
    if direction == 1:
        kvec[:,1] = np.linspace(0,2*np.pi,n+1)[0:n]
        return kvec
    if direction == 0:
        kvec[:,0] = np.linspace(0,2*np.pi,n+1)[0:n]
        return kvec
    if direction == 2:
        kvec[:,2] = np.linspace(0,2*np.pi,n+1)[0:n]
        return kvec

def column(i,j):
    helper_inds = [[0, 0], [-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    return helper_inds.index([i,j])

# def columb_31(i,j):
    

class Gjeniouous_class:
    def __init__(self):
        print('Gjeniouous class initalisd')
        self.lol = 'lol'
    def toarray(self):
        return np.zeros(0)

def pull_region(A,i,j):
    n=A.shape[0]
    tal = column(i,j) # bogstav????
    if tal<=A.shape[1]:
        return A[:,tal*n :  (tal+1)*n]
    else:
        return Gjeniouous_class()

class System:
    def __init__(self, 
                 sisl_dev, 
                 sisl_leads,
                 lead_inds,
                 E_grid,
                 Thz_max, 
                 dE, 
                 buffer_inds = [],
                 k_uv=np.array([0,1,0]),
                 multiple_absorption=0, 
                 eta = 1e-3,
                 pivot = None):
        self.sisl_dev = sisl_dev
        self.sisl_leads = sisl_leads
        below = np.flip(np.arange(E_grid.min()-dE,E_grid.min()-multiple_absorption*Thz_max-2*dE,-dE))
        above = np.arange(E_grid.max()+dE,E_grid.max()+multiple_absorption*Thz_max+2*dE,+dE)
        if multiple_absorption>0:
            self.E_grid = np.hstack([below,
                                     E_grid,
                                     above])
        else:
            self.E_grid = E_grid
        self.fot_E_for_copy=np.arange(E_grid.max()+dE,E_grid.max()+Thz_max, dE)-E_grid.max()
        self.dE = dE
        self.CAP_ADDED=False
        self.lead_inds = lead_inds
        self.buffer_inds = buffer_inds
        self.ne_tot = len(self.E_grid)
        self.k_uv = k_uv
        self.eta = eta
        self.pivot = pivot
    
    def Set_kp(self,k_in):
        self.k_avg = k_in
        self.nk    = len(self.k_avg)
        if self.nk==1 and isinstance(k_in,list) and k_in==[None]:
            print('\n No k-points!\n')
            self.phases = [np.complex128(0+0j)]
            self.kvecs  = None
        else:
            self.phases = [np.exp(1j*self.k_avg[i,:].dot(self.k_uv)) for i in range(self.nk)]
            self.kvecs = k_in
    
    def Organise_and_Check(self):
        Hd = self.sisl_dev['H']
        Sd = self.sisl_dev['S']
        assert Hd.no    == Sd.no
        assert (Hd.cell == Sd.cell).all()
        assert Hd.atoms == Sd.atoms
        lead_inds_orbital = [[] for l in self.lead_inds]
        buffer_inds_orbital = []
        
        it_lead = 0
        for l in self.sisl_leads:
            it_o = 0
            for idx, a in enumerate(Hd.atoms):
                for o in a.orbitals:
                    if idx in self.lead_inds[it_lead]:
                        lead_inds_orbital[it_lead] += [it_o]
                    it_o += 1
            it_lead+=1
        it_o = 0
        for idx, a in enumerate(Hd.atoms):
            for o in a.orbitals:
                if idx in self.buffer_inds:
                    buffer_inds_orbital += [it_o]
                it_o+=1
        
        del it_lead, it_o, idx
        
        self._old_buffer_inds = self.buffer_inds.copy()
        self.buffer_inds      = buffer_inds_orbital
        self._old_lead_inds   = self.lead_inds.copy()
        self.lead_inds        = lead_inds_orbital
        
        print('Checking consistency of positions and number of orbitals with the given indices')
        for i in range(len(self.sisl_leads)):
            assert  self.sisl_leads[i]['H'].no    == len(self.lead_inds[i] )
            assert  self.sisl_leads[i]['S'].no    == len(self.lead_inds[i] )
            assert  np.isclose(self.sisl_leads[i]['S'].cell ,   self.sisl_leads[i]['H'].cell).all()
            assert  self.sisl_leads[i]['S'].atoms ==     self.sisl_leads[i]['H'].atoms
        count = 0
        for lead in self.sisl_leads:
            assert np.isclose(lead['H'].xyz,self.sisl_dev['H'].xyz[self._old_lead_inds[count]]).all()
            count+=1
        
        no = Hd.no
        inds=[]
        #Find indecies of device part of the Hamiltonian containing device and 1 UC of the leads
        nl = len(self.lead_inds)
        for i in range(no):
            it = 0
            for lead_ind in self.lead_inds:
                if i in lead_ind or i in self.buffer_inds:
                    pass
                else:
                    it+=1
            if it==nl:
                inds+=[i]
        
        self.dev_inds = inds
        self._input_pivot = self.pivot.copy()
        self.pivot = [self.dev_inds.index(p) for p in self.pivot]
    
    def Gen_SE_decimation(self,dirs =[(-1,0),(1,0)],tol = 1e-12):
        pdir=np.where(self.k_uv!=0)[0]
        # self.pdir = pdir
        self.SE = []
        self.GS = []
        self.lead_directions = dirs.copy()
        self.l_to_d_inds=[]
        self.l_to_d_inds_dev_only=[]
        it_lead=0
        for lead in self.sisl_leads:
            pdir = dirs[it_lead].index(0)
            if pdir == 1:
                way = dirs[it_lead][0]
            if pdir == 0:
                way = dirs[it_lead][1]
            
            H_l = lead['H']; S_l = lead['S']
            H_l_sparse  = H_l.tocsr()
            S_l_sparse  = S_l.tocsr()
            
            no = H_l.no
            List_H = []; List_S = []
            H = {}     ; S = {}
            it = 0
            print('lead-lead couplings....\n')
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    Hrr = pull_region(H_l_sparse,i,j)
                    Srr = pull_region(S_l_sparse,i,j)
                    
                    hsub = Hrr.toarray()
                    ssub = Srr.toarray()
                    List_H+=[hsub.copy()];
                    List_S+=[ssub.copy()]
                    
                    H.update({(i,j):List_H[it]})
                    S.update({(i,j):List_S[it]})
                    
                    it+=1
            
            print('surface greens function....\n')
            
            gs = CZ((self.nk,self.ne_tot,no,no))
            for i in range(self.nk):
                phase = self.phases[i]
                
                if pdir == 1:
                    V_LEAD   = H[(way,0)]  +  phase*H[(way,1)] + phase.conj()*H[(way,-1)]
                    S01_LEAD = S[(way,0)]  +  phase*S[(way,1)] + phase.conj()*S[(way,-1)]
                    H_LEAD   = H[(0,  0)]  +  phase*H[(0,  1)] + phase.conj()*H[(0,  -1)]
                    S00_LEAD = S[(0,  0)]  +  phase*S[(0,  1)] + phase.conj()*S[(0,  -1)]
                
                if pdir == 0:
                    V_LEAD   = H[(0,way)]  +  phase*H[(1,way)] + phase.conj()*H[(-1,way)]
                    S01_LEAD = S[(0,way)]  +  phase*S[(1,way)] + phase.conj()*S[(-1,way)]
                    H_LEAD   = H[(0,  0)]  +  phase*H[(1,0  )] + phase.conj()*H[(-1, 0 )]
                    S00_LEAD = S[(0,  0)]  +  phase*S[(1,0  )] + phase.conj()*S[(-1, 0 )]
                print(pdir, way)
                
                
                se_lead = SEv( self.E_grid,    H_LEAD, V_LEAD, 
                               S00 = S00_LEAD, S01 = S01_LEAD, 
                               eps = tol,      eta = self.eta )
                for j in range(self.ne_tot):
                    z = (self.E_grid[j]+1j*self.eta)
                    gs[i,j,:,:] = Inv( S00_LEAD*z - H_LEAD - se_lead[j] )
            
            self.GS+=[gs]
            no_d = len(self.dev_inds)
            print('lead-device couplings....\n')
            V_0_dl  = CZ((no_d,no))
            V_1_dl  = CZ((no_d,no))
            V_m1_dl = CZ((no_d,no))
            S_0_dl  = CZ((no_d,no))
            S_1_dl  = CZ((no_d,no))
            S_m1_dl = CZ((no_d,no))
            tup0 = (0,0)
            
            if pdir == 1:
                tup1  = (0, 1)
                tupm1 = (0,-1)
            if pdir == 0:
                tup1  = ( 1, 0)
                tupm1 = (-1, 0)
            print(tup1, tupm1, pdir )
            di = self.dev_inds
            li = self.lead_inds[it_lead]
            V_0_dl [:,:] = pull_region(self.sisl_dev['H'].tocsr(),tup0 [0] , tup0[1] )[di,:][:,li].toarray()
            V_1_dl [:,:] = pull_region(self.sisl_dev['H'].tocsr(),tup1 [0] , tup1[1] )[di,:][:,li].toarray()
            V_m1_dl[:,:] = pull_region(self.sisl_dev['H'].tocsr(),tupm1[0], tupm1[1] )[di,:][:,li].toarray()
            S_0_dl [:,:] = pull_region(self.sisl_dev['S'].tocsr(),tup0 [0] , tup0[1] )[di,:][:,li].toarray()
            S_1_dl [:,:] = pull_region(self.sisl_dev['S'].tocsr(),tup1 [0] , tup1[1] )[di,:][:,li].toarray()
            S_m1_dl[:,:] = pull_region(self.sisl_dev['S'].tocsr(),tupm1[0], tupm1[1] )[di,:][:,li].toarray()
            
            
            if self.nk == 1 and self.phases[0]==0:
                ME_list = [V_0_dl,S_0_dl]
            else:
                ME_list = [V_0_dl,V_1_dl,V_m1_dl,S_0_dl,S_1_dl,S_m1_dl]
            
            l_to_d_coupling_inds = []
            l_to_d_coupling_inds_dev_only = []
            
            for i in range(no_d):
                if sum([(q[i,:]==0).all() for q in ME_list])==len(ME_list):
                    pass
                else:
                    l_to_d_coupling_inds_dev_only += [i]
                    l_to_d_coupling_inds          += [self.dev_inds[i]]
            
            self.l_to_d_inds+=[l_to_d_coupling_inds]
            self.l_to_d_inds_dev_only+=[l_to_d_coupling_inds_dev_only]
            
            V_0_dl  = V_0_dl [l_to_d_coupling_inds_dev_only,:]
            V_1_dl  = V_1_dl [l_to_d_coupling_inds_dev_only,:]
            V_m1_dl = V_m1_dl[l_to_d_coupling_inds_dev_only,:]
            S_0_dl  = S_0_dl [l_to_d_coupling_inds_dev_only,:]
            S_1_dl  = S_1_dl [l_to_d_coupling_inds_dev_only,:]
            S_m1_dl = S_m1_dl[l_to_d_coupling_inds_dev_only,:]
            
            ncoup = len(V_0_dl)
            se = CZ((self.nk,self.ne_tot,ncoup,ncoup))
            for i in range(self.nk):
                phase = self.phases[i]
                vk = V_0_dl + phase * V_1_dl + phase.conj() * V_m1_dl
                sk = S_0_dl + phase * S_1_dl + phase.conj() * S_m1_dl
                for j in range(self.ne_tot):
                    ej = self.E_grid[j]
                    Uu = vk-ej*sk
                    se[i,j,:,:] = Uu.dot(gs[i,j,:,:]).dot(Uu.conj().T)
            
            self.SE+=[se.copy()]
            
            it_lead+=1
    
    def Gen_SE_CAP(self,dirs =[(-1,0),(1,0)]):
        pdir=np.where(self.k_uv!=0)[0]
        self.pdir = pdir
        self.lead_directions = dirs.copy()
    
    def Block_Setup_decimation(self,P,tol = 1e-10, force_continue=False, test_interval = 10):
        dev_inds  = self.dev_inds
        
        Hd = self.sisl_dev['H'].tocsr()
        Sd = self.sisl_dev['S'].tocsr()
        assert (self.sisl_dev['H'].nsc== np.array([3,3,1])).all()
        
        H_0_dd = pull_region(Hd,0,0)[dev_inds,:][:,dev_inds]
        S_0_dd = pull_region(Sd,0,0)[dev_inds,:][:,dev_inds]
        if np.where(self.k_uv==1)[0][0] == 0:
            pdir = 0
        if np.where(self.k_uv==1)[0][0] == 1:
            pdir = 1
        else:
            pdir = None
        
        if pdir == None:
            H_1_dd  = np.zeros(H_0_dd.shape).asdtype(complex)
            H_m1_dd = np.zeros(H_0_dd.shape).asdtype(complex)
            S_1_dd  = np.zeros(H_0_dd.shape).asdtype(complex)
            S_m1_dd = np.zeros(H_0_dd.shape).asdtype(complex)
        
        
        elif pdir == 1:
            H_1_dd = pull_region(Hd,0, 1)[dev_inds,:][:,dev_inds]
            H_m1_dd= pull_region(Hd,0,-1)[dev_inds,:][:,dev_inds]
            S_1_dd = pull_region(Sd,0, 1)[dev_inds,:][:,dev_inds]
            S_m1_dd= pull_region(Sd,0,-1)[dev_inds,:][:,dev_inds]
        elif pdir == 0:
            H_1_dd = pull_region(Hd, 1,0)[dev_inds,:][:,dev_inds]
            H_m1_dd= pull_region(Hd,-1,0)[dev_inds,:][:,dev_inds]
            S_1_dd = pull_region(Sd, 1,0)[dev_inds,:][:,dev_inds]
            S_m1_dd= pull_region(Sd,-1,0)[dev_inds,:][:,dev_inds]
        n_diags = len(P)-1
        # P is the partition used for the BTD setup of the system´
        assert P[-1] == len(dev_inds)         #ends at index as large as #deviceindecies
        assert P[ 0] == 0                     #starts at zero
        assert P == sorted(P)                 #ordered
        assert P == sorted(list(set(P)))      #no duplicates
        nk = self.nk
        ne = self.ne_tot
        no_d = len(dev_inds)
        num_leads = len(self.lead_inds)
        
        Ia = [i for i in range(n_diags  )]
        Ib = [i for i in range(n_diags-1)]
        Ic = [i for i in range(n_diags-1)]
        
        Al = [CZ((nk,ne,P[i+1]-P[i  ],P[i+1]-P[i  ])) for i in range(n_diags  )]
        Bl = [CZ((nk,ne,P[i+2]-P[i+1],P[i+1]-P[i  ])) for i in range(n_diags-1)]
        Cl = [CZ((nk,ne,P[i+1]-P[i  ],P[i+2]-P[i+1])) for i in range(n_diags-1)]
        
        Gli = []
        SEli = []
        print('\n Building ES - H - Self Energies \n')
        self.iGreens = block_td(Al,Bl,Cl,Ia,Ib,Ic,diagonal_zeros=False,E_grid = self.E_grid)
        del Al, Bl, Cl
        gc.collect()
        
        for i in tqdm(range(nk)):
            phase = self.phases[i]
            hk    = H_0_dd + phase * H_1_dd + phase.conj() * H_m1_dd 
            sk    = S_0_dd + phase * S_1_dd + phase.conj() * S_m1_dd
            Glj = []
            SElj= []
            for j in range(ne):
                z  = self.E_grid[j]+1j*self.eta
                se_list    = []
                Gamma_list = []
                
                for it_lead in range(num_leads):
                    se_sparse = sp.csr_matrix((no_d,no_d),dtype=complex)
                    inds_coupling   = self.l_to_d_inds_dev_only[it_lead]
                    se_sub          = self.SE[it_lead][i,j,:,:]
                    iv = []
                    jv = []
                    for ii in inds_coupling:
                        for jj in inds_coupling:
                            #se_sparse[ii,jj] += se_sub[count_ii,count_jj]
                            iv+=[ii]
                            jv+=[jj]
                    
                    se_sparse[iv,jv] = se_sub.ravel()
                    se_list   +=[    se_sparse.copy()                    ]
                    Gamma_list+=[1j*(se_sparse-se_sparse.conj().T).copy()]
                
                iG = sk*z - hk - sum(se_list)
                if self.pivot is not None:
                    iG = iG[self.pivot , :][: , self.pivot]
                
                if np.mod(j,test_interval) == 0:
                    f,S=test_partition_2d_sparse_matrix(iG,P)
                    if f<1-tol:
                        print('\n-------------------------------------------------------------------\n Matrix elements lost during partitioning. Choose less restrictive partitioning\n-----------------------------------------------------------------\n')
                        print('\n size of elements: ' +  str(f))
                        
                        if force_continue==True:
                            print('Matrix elements lost, but continuing\n')
                        else:
                            assert 1 == 0
                
                al,bl,cl,ia,ib,ic=Build_BTD(iG,S)
                if self.pivot is None:
                    Glj+=[[Build_BS(Gamma_list[QQ],P) for QQ in range(num_leads)]]
                    SElj+=[[Build_BS(se_list[QQ],P)   for QQ in range(num_leads)]]
                else:
                    Glj+=[ [Build_BS(Gamma_list[QQ][self.pivot , :][: , self.pivot],P) 
                            for QQ in range(num_leads)] ]
                    SElj+=[ [ Build_BS(se_list[QQ]  [self.pivot , :][: , self.pivot],P)   
                              for QQ in range(num_leads) ] ]
                
                for b in range(n_diags):
                    self.iGreens.Al[b][i,j,:,:] += al[b]
                    if b<n_diags-1:
                        self.iGreens.Bl[b][i,j,:,:] += bl[b]
                        self.iGreens.Cl[b][i,j,:,:] += cl[b]
                
            Gli+=[Glj]
            SEli+=[SElj]
        
        #Building Gamma blocks
        ## Get block structure of the Gammas
        representative_inds = [[] for i in range(num_leads)]
        
        for Glj in Gli:
            for Gl in Glj:
                it=0
                for g in Gl:
                    Ids = g[0]
                    for ids in Ids:
                        if ids not in representative_inds[it]:
                            representative_inds[it]+=[ids]
                    it+=1
        self._representative_inds =  representative_inds
        bs_list = []
        se_list = []
        for l in range(num_leads):
            vals =  [ CZ((nk,ne,P[ids[0]+1]-P[ids[0]],P[ids[1]+1]-P[ids[1]])) for ids in representative_inds[l] ]
            inds =  representative_inds[l]
            vals_se=[ CZ((nk,ne,P[ids[0]+1]-P[ids[0]],P[ids[1]+1]-P[ids[1]])) for ids in representative_inds[l] ]
            for i in range(nk):
                for j in range(ne):
                    for count in range(len(inds)):
                        vals   [count][i,j,:,:] = Gli [i][j][l][1][count]
                        vals_se[count][i,j,:,:] = SEli[i][j][l][1][count]
            
            bs = block_sparse(inds,vals,(n_diags,n_diags),E_grid  = self.E_grid)
            bs_list+=[bs.copy()]
            bs2=block_sparse(inds,vals_se,(n_diags,n_diags),E_grid = self.E_grid)
            se_list+=[bs2.copy()]
        self.Gammas = bs_list
        self.SelfEnergies = se_list
        del Gli,bs,bs2
        gc.collect()
    
    def Block_Setup_CAP(self, P, test_interval = 10, 
                        Manual_se = None,
                        CAP_Dir = 0,CAP_c = 1, 
                        CAP_Type = '??',
                        CAP_move = 1,
                        tol  = 1e-10,
                        force_continue=False):
        whole_inds = [i for i in range(self.sisl_dev['H'].no) if i not in self.buffer_inds]
        Hd = self.sisl_dev['H'].tocsr()
        Sd = self.sisl_dev['S'].tocsr()
        no_tot = Hd.shape[0]
        assert (self.sisl_dev['H'].nsc== np.array([3,3,1])).all()
        
        H_0_dd = pull_region(Hd,0,0)[whole_inds,:][:,whole_inds]
        S_0_dd = pull_region(Sd,0,0)[whole_inds,:][:,whole_inds]
        
        if self.pdir == 1:
            t1 = (0, 1)
            tm1= (0,-1)
        elif self.pdir == 0:
            t1 = ( 1,0)
            tm1= (-1,0)
        
        H_1_dd = pull_region(Hd, t1[0], t1[1])[whole_inds,:][:,whole_inds]
        H_m1_dd= pull_region(Hd,tm1[0],tm1[1])[whole_inds,:][:,whole_inds]
        S_1_dd = pull_region(Sd, t1[0], t1[1])[whole_inds,:][:,whole_inds]
        S_m1_dd= pull_region(Sd,tm1[0],tm1[1])[whole_inds,:][:,whole_inds]
        
        ######
        ###### Very similar to the same snippet from "Block_Setup_decimation"
        n_diags = len(P)-1
        # P is the partition used for the BTD setup of the system´
        assert P[-1] == len(whole_inds)         #ends at index as large as # total indecies
        assert P[ 0] == 0                       #starts at zero
        assert P == sorted(P)                   #ordered
        assert P == sorted(list(set(P)))        #no duplicates
        nk = self.nk
        ne = self.ne_tot
        no_orb = len(whole_inds)
        num_leads = len(self.lead_inds)
        
        Ia = [i for i in range(n_diags  )]
        Ib = [i for i in range(n_diags-1)]
        Ic = [i for i in range(n_diags-1)]
        
        Al = [CZ((nk,ne,P[i+1]-P[i  ],P[i+1]-P[i  ])) for i in range(n_diags  )]
        Bl = [CZ((nk,ne,P[i+2]-P[i+1],P[i+1]-P[i  ])) for i in range(n_diags-1)]
        Cl = [CZ((nk,ne,P[i+1]-P[i  ],P[i+2]-P[i+1])) for i in range(n_diags-1)]
        print('\n Building ES - H - Self Energies \n')
        self.iGreens = block_td(Al,Bl,Cl,Ia,Ib,Ic,diagonal_zeros=False,E_grid = self.E_grid)
        del Al, Bl, Cl
        gc.collect()
        
        CAP_SEs =    []
        CAP_gammas = []
        
        
        #######
        #######
        for i in tqdm(range(nk)):
            phase = self.phases[i]
            hk    = H_0_dd + phase * H_1_dd + phase.conj() * H_m1_dd 
            sk    = S_0_dd + phase * S_1_dd + phase.conj() * S_m1_dd
            
            for it_lead in range(num_leads):
                li = self.lead_inds[it_lead]
                pos_lead = self.sisl_dev['H'].xyz[li,CAP_Dir]
                way  = self.lead_directions[it_lead][CAP_Dir]
                if way == -1:
                    pmin = pos_lead.max() + CAP_move
                    pmax = pos_lead.min() - CAP_move
                if way ==  1:
                    pmin = pos_lead.min() - CAP_move
                    pmax = pos_lead.max() + CAP_move
                w_diag = 1j*W(pos_lead, pmin, pmax, c = CAP_c, Type = CAP_Type)
                #Take out lead parts ......
                # i_ox, j_ox, overlap = sp.find(sk[li,:][:,li])
                # np_arr_li = np.array(li)
                
                #filter to lead only
                
                # we evaluate the offdiagonal matrix elements of the CAP by evaluating W in the average position of
                # the overlapping orbitals and multiplying by the overlap (i.e we assume W(x) changes sufficiently slowly,
                # which we can get it we make the absorping region long enough).
                
                # x_av  = (self.sisl_leads[it_lead]['S'].xyz[i_ox,0] + self.sisl_leads[it_lead]['S'].xyz[j_ox,0]) / 2
                # wij   = 1j * W(x_av, pmin, pmax, c = CAP_c, Type = CAP_Type) * overlap
                
                # ..... and map back to "global" matrix indecies
                # i_ox = np_arr_li[i_ox]
                # j_ox = np_arr_li[j_ox]
                
                if Manual_se == None:
                    wm = sp.csr_matrix((no_orb,no_orb),dtype=complex)
                    wm[li,li] = w_diag
                    # wm[i_ox,j_ox] = wij
                
                else:
                    wm = Manual_se[it_lead]
                
                CAP_SEs    += [wm.copy()                 ]
                CAP_gammas += [1j*(wm-wm.conj().T).copy()]
            
            SE_sum = sum(CAP_SEs)
            for j in range(ne):
                z  = self.E_grid[j]+1j*self.eta
                iG = sk*z - hk - SE_sum
                
                if np.mod(j,test_interval) == 0:
                    f,S=test_partition_2d_sparse_matrix(iG,P)
                    if f<1-tol:
                        print('\n-------------------------------------------------------------------\n Matrix elements lost during partitioning. Choose less restrictive partitioning\n-----------------------------------------------------------------\n')
                        if force_continue==True:
                            print(f)
                            print('Matrix elements lost, but continuing\n')
                        else:
                            print(f)
                            print('Matrix_elements lost: ')
                            assert 1 == 0
                
                al,bl,cl,ia,ib,ic=Build_BTD(iG,S)
                
                for b in range(n_diags):
                    self.iGreens.Al[b][i,j,:,:] += al[b]
                    if b<n_diags-1:
                        self.iGreens.Bl[b][i,j,:,:] += bl[b]
                        self.iGreens.Cl[b][i,j,:,:] += cl[b]
        BS_SEs    = []
        BS_Gammas = []
        
        for it_lead in range(num_leads):
            inds_se,vals_se = Build_BS(CAP_SEs   [it_lead],P)
            inds_ga,vals_ga = Build_BS(CAP_gammas[it_lead],P)
            bs_se=block_sparse(inds_se,vals_se,(n_diags,n_diags),E_grid = self.E_grid)
            bs_ga=block_sparse(inds_ga,vals_ga,(n_diags,n_diags),E_grid = self.E_grid)
            BS_SEs    += [bs_se.copy()]
            BS_Gammas += [bs_ga.copy()]
        
        self.SelfEnergies = BS_SEs
        self.Gammas       = BS_Gammas
        self.iGreens.Find_Duplicates()
        

def Simple_calc(Greens,Gl,Gr,Eg, fot_E ):
    ne = len(Eg)
    nw = len(fot_E)
    nk = Greens.Block(0,0).shape[0]
    
    
    Int_T = CZ((len(fot_E),2,2))
    Int_Y = CZ((len(fot_E),2))
    Int_T = CZ((len(fot_E),2,2))
    Int_Y = CZ((len(fot_E),2))
    Int_TE = CZ((len(fot_E),ne,2,2))
    Int_YE = CZ((len(fot_E),ne,2))
    
    BZero = block_sparse([],[],Greens.Block_shape,E_grid = Greens.E_grid,FoRcE_dTypE = np.complex128)
    
    for i in tqdm(range(nw)):
        
        w = fot_E[i]
        M_Ghw_Gl  = Greens.BDot(Gl,Ei1 = Eg + w, 
                                   Ei2 = Eg
                                   )
        M_Ghw_Gr  = Greens.BDot(Gr,Ei1 = Eg + w, 
                                   Ei2 = Eg
                                   )
        fd = f(Eg)-f(Eg+w)
        
        Greens.do_dag()
        M_Gd_Gl = Greens.BDot(Gl,  Ei1 = Eg, Ei2 = Eg)
        M_Gd_Gr = Greens.BDot(Gr,  Ei1 = Eg, Ei2 = Eg)
        Gd_sub  = Greens.Add(BZero,Ei1 = Eg, Ei2 = Eg)
        Greens.do_dag()
        
        # sum(axis=0) sums over k-points
        div = w*nk
        TE01 =  M_Ghw_Gl.TrProd(M_Gd_Gr).sum(axis=0)
        TE10 =  M_Ghw_Gr.TrProd(M_Gd_Gl).sum(axis=0)
        TE11 = (M_Ghw_Gr.TrProd(M_Gd_Gr).sum(axis=0) - 1j*( M_Ghw_Gr.Tr().sum(axis = 0) - M_Gd_Gr.Tr().sum(axis=0)))
        TE00 = (M_Ghw_Gl.TrProd(M_Gd_Gl).sum(axis=0) - 1j*( M_Ghw_Gl.Tr().sum(axis = 0) - M_Gd_Gl.Tr().sum(axis=0)))
        
        #ugly way to pick out the indecies corresponding to Eg in Greens
        
        YE0  = - 1j * M_Ghw_Gr.TrProd(Gd_sub).sum(axis=0)
        YE1  = - 1j * M_Ghw_Gl.TrProd(Gd_sub).sum(axis=0)
        
        Int_T[i,0,0] += integrate_f_x(TE00*fd,Eg)/div
        Int_T[i,0,1] += integrate_f_x(TE01*fd,Eg)/div
        Int_T[i,1,0] += integrate_f_x(TE10*fd,Eg)/div
        Int_T[i,1,1] += integrate_f_x(TE11*fd,Eg)/div
        Int_Y[i,0] += integrate_f_x(YE0*fd,Eg)/nk
        Int_Y[i,1] += integrate_f_x(YE1*fd,Eg)/nk
        
        # Int_TE[i,:,0,0] = TE00
        # Int_TE[i,:,0,1] = TE01
        # Int_TE[i,:,1,0] = TE10
        # Int_TE[i,:,1,1] = TE11
        # Int_YE[i,:,0  ] = YE0
        # Int_YE[i,:,1  ] = YE1
        
    Pl_1 = -Int_T[:,0,:].sum(axis=1)/( Int_Y.sum(axis=1))
    Pr_1 = -Int_T[:,1,:].sum(axis=1)/( Int_Y.sum(axis=1))
    G_1  =  Int_T[:,0,1]  +  Pl_1*Int_Y[:,1]
    return G_1, Int_T, Int_Y



def less_simple_calc(Greens, iGreens, Gl, Gr, Eg, fot_E, GGd_blocks):
    ne = len(Eg)
    nw = len(fot_E)
    nk = Greens.Block(0,0).shape[0]
    print('nk is: ', nk)
    Int_T = CZ((len(fot_E),2,2))
    Int_Y = CZ((len(fot_E),2))
    Int_TE = CZ((len(fot_E),ne,2,2))
    Int_YE = CZ((len(fot_E),ne,2))
    
    BZero = block_sparse([],[],Greens.Block_shape,E_grid = Greens.E_grid,FoRcE_dTypE = np.complex128)
    
    for i in tqdm(range(nw)):
        
        w = fot_E[i]
        M_Ghw_Gl  = Greens.BDot(Gl,Ei1 = Eg + w, 
                                   Ei2 = Eg
                                   )
        M_Ghw_Gr  = Greens.BDot(Gr,Ei1 = Eg + w, 
                                   Ei2 = Eg
                                   )
        fd = f(Eg)-f(Eg+w)
        
        Greens.do_dag()
        M_Gd_Gl = Greens.BDot(Gl,  Ei1 = Eg, Ei2 = Eg)
        M_Gd_Gr = Greens.BDot(Gr,  Ei1 = Eg, Ei2 = Eg)
        Greens.do_dag()
        
        i_Ghw_Gd = iGreens.A_Adag(Ei1 = Eg + w, Ei2 = Eg )
        i_Ghw_Gd = i_Ghw_Gd.Make_BTD()
        
        N_ggdb = i_Ghw_Gd.Block_shape[0]
        sb = []
        
        for I in range(GGd_blocks):
            sb += [I]
        for I in range(GGd_blocks):
            sb += [N_ggdb-I-1]
        sb = sorted(sb)
        print(sb, i_Ghw_Gd.shape)
        diagGG = i_Ghw_Gd.Inverse_Diag_of_Diag(sb)
        diag_Gr  = Gr.Get_diagonal(iGreens.all_slices)
        diag_Gl  = Gl.Get_diagonal(iGreens.all_slices)
        YE0 = np.sum(diag_Gr *  diagGG, axis = (-1,0))
        YE1 = np.sum(diag_Gl *  diagGG, axis = (-1,0))
        
        div = w*nk
        TE01 =  M_Ghw_Gl.TrProd(M_Gd_Gr).sum(axis=0)
        TE10 =  M_Ghw_Gr.TrProd(M_Gd_Gl).sum(axis=0)
        TE11 = (M_Ghw_Gr.TrProd(M_Gd_Gr).sum(axis=0) - 1j*( M_Ghw_Gr.Tr().sum(axis = 0) - M_Gd_Gr.Tr().sum(axis=0)))
        TE00 = (M_Ghw_Gl.TrProd(M_Gd_Gl).sum(axis=0) - 1j*( M_Ghw_Gl.Tr().sum(axis = 0) - M_Gd_Gl.Tr().sum(axis=0)))
        
        Int_T[i,0,0] += integrate_f_x(TE00*fd,Eg)/div
        Int_T[i,0,1] += integrate_f_x(TE01*fd,Eg)/div
        Int_T[i,1,0] += integrate_f_x(TE10*fd,Eg)/div
        Int_T[i,1,1] += integrate_f_x(TE11*fd,Eg)/div
        Int_Y[i,0]   += integrate_f_x(YE0*fd,Eg)/nk
        Int_Y[i,1]   += integrate_f_x(YE1*fd,Eg)/nk
        
        
        
        Int_TE[i,:,0,0] = TE00
        Int_TE[i,:,0,1] = TE01
        Int_TE[i,:,1,0] = TE10
        Int_TE[i,:,1,1] = TE11
        Int_YE[i,:,0  ] = YE0
        Int_YE[i,:,1  ] = YE1
        
    Pl_1 = -Int_T[:,0,:].sum(axis=1)/( Int_Y.sum(axis=1))
    Pr_1 = -Int_T[:,1,:].sum(axis=1)/( Int_Y.sum(axis=1))
    G_1  =  Int_T[:,0,1]  +  Pl_1*Int_Y[:,1]
    return G_1, [Int_TE,Int_YE]


def serial_k_calc(System_object, kvec, Partition,Eg, hw, tol = 1e-15):
    
    ne = len(Eg)
    nw = len(hw)
    nk = len(kvec)
    print('nk is: ', nk)
    Int_T   = CZ((nw,2,2))
    Int_Y   = CZ((nw,2))
    Int_TE  = CZ((nw,ne,2,2))
    Int_YE  = CZ((nw,ne,2))
    it_k = 0
    for k in kvec:
        print('k-point ',str(it_k+1), ' out of ',  len(kvec), '\n')
        if k is None:
            k = [None]
        else:
            k = np.array([k]) # lav fra (3,) til (1,3) shape, det er det Set_kp spiser
        
        System_object.Set_kp(k)
        System_object.Organise_and_Check()
        System_object.Gen_SE_CAP()
        System_object.Block_Setup_CAP( P = Partition,CAP_move = -0.5, CAP_c = 0.3)
        iG = System_object.iGreens
        Gl = System_object.Gammas[0]
        Gr = System_object.Gammas[1]
        N = iG.Block_shape[0]
        for b in range(N):
            b1 = Gl.Block(b,b)
            b2 = Gr.Block(N-b-1, N-b-1)
            if b1 is None:
                b1 = np.zeros(2)
            else:
                b1 = np.abs(b1)
            if b2 is None:
                b2 = np.zeros(2)
            else:
                b2 = np.abs(b2)
            if (b1 < tol).all() and (b2 < tol).all():
                break
        G = iG.Invert('*\*'+str(b)) # Greens function on an extented grid
                                    # so that interpolation with hw is possible
        #BZero = block_sparse([],[],G.Block_shape,E_grid = G.E_grid,FoRcE_dTypE = np.complex128)
        for i in range(nw):
            w = hw[i]
            M_Ghw_Gl  = G.BDot(Gl,Ei1 = Eg + w, 
                                  Ei2 = Eg
                              )
            M_Ghw_Gr  = G.BDot(Gr,Ei1 = Eg + w, 
                                  Ei2 = Eg
                              )
            fd = f(Eg)-f(Eg+w)
            
            G.do_dag()
            M_Gd_Gl = G.BDot(Gl,  Ei1 = Eg, Ei2 = Eg)
            M_Gd_Gr = G.BDot(Gr,  Ei1 = Eg, Ei2 = Eg)
            G.do_dag()
            
            #Ghw_Gd is inverse of actual Ghw_Gd, a bit confusing
            Ghw_Gd = iG.A_Adag(Ei1 = Eg + w, Ei2 = Eg )
            Ghw_Gd = Ghw_Gd.Make_BTD()
            rough_bs = Ghw_Gd.all_slices.copy()
            N_ggdb = Ghw_Gd.Block_shape[0]
            sb = []#''
            
            for I in range(b//2+1):
                sb += [I]#str(I)+' '
            for I in range(b//2+1-1,-1,-1):
                sb += [N_ggdb-I-1]#str(N_ggdb-I-1)+' '
            # Ghw_Gd = Ghw_Gd.Invert(BW = 'diag ' + sb)
            sb = sorted(sb)
            print(sb, Ghw_Gd.Block_shape)
            diagGG = Ghw_Gd.Inverse_Diag_of_Diag(sb)
            diag_Gr  = Gr.Get_diagonal(iG.all_slices)
            diag_Gl  = Gl.Get_diagonal(iG.all_slices)
            
            # Ghw_Gd and Gr/Gl/G/iG does not have the same "block-structure" since we made the 5-diagonal into a 3-diagonal
            # a couple of lines above
            YE0 = block_TRACE_different_bs(Ghw_Gd, Gl, rough_bs, iG.all_slices ).sum(axis = 0) #np.sum(diag_Gr *  diagGG, axis = (-1,0))
            YE1 = block_TRACE_different_bs(Ghw_Gd, Gr, rough_bs, iG.all_slices ).sum(axis = 0) #np.sum(diag_Gl *  diagGG, axis = (-1,0))
            
            div = w*nk
            TE01 =  M_Ghw_Gl.TrProd(M_Gd_Gr).sum(axis=0)
            TE10 =  M_Ghw_Gr.TrProd(M_Gd_Gl).sum(axis=0)
            TE11 = (M_Ghw_Gr.TrProd(M_Gd_Gr).sum(axis=0) - 1j*( M_Ghw_Gr.Tr().sum(axis = 0) - M_Gd_Gr.Tr().sum(axis=0)))
            TE00 = (M_Ghw_Gl.TrProd(M_Gd_Gl).sum(axis=0) - 1j*( M_Ghw_Gl.Tr().sum(axis = 0) - M_Gd_Gl.Tr().sum(axis=0)))
            
            Int_T[i,0,0] += integrate_f_x(TE00*fd,Eg)/div
            Int_T[i,0,1] += integrate_f_x(TE01*fd,Eg)/div
            Int_T[i,1,0] += integrate_f_x(TE10*fd,Eg)/div
            Int_T[i,1,1] += integrate_f_x(TE11*fd,Eg)/div
            Int_Y[i,0]   += integrate_f_x(YE0*fd,Eg)/nk
            Int_Y[i,1]   += integrate_f_x(YE1*fd,Eg)/nk
            
            Int_TE[i,:,0,0] += TE00/nk
            Int_TE[i,:,0,1] += TE01/nk
            Int_TE[i,:,1,0] += TE10/nk
            Int_TE[i,:,1,1] += TE11/nk
            Int_YE[i,:,0  ] += YE0/nk
            Int_YE[i,:,1  ] += YE1/nk
        it_k+=1
    
    
    
    Pl_1 = -Int_T[:,0,:].sum(axis=1)/( Int_Y.sum(axis=1))
    Pr_1 = -Int_T[:,1,:].sum(axis=1)/( Int_Y.sum(axis=1))
    G_1  =  Int_T[:,0,1]  +  Pl_1*Int_Y[:,1]
    return G_1, [Int_TE,Int_YE]
     




        
        




import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size





