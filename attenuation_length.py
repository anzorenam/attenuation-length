#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ROOT
import root_numpy as rnp

sns.set(rc={"figure.figsize":(8,4)})
sns.set_context('paper',font_scale=1.5,rc={'lines.linewidth':1.0})
sns.set_style('ticks')
mat.rc('text',usetex=True)
mat.rc('font',family='DejaVu Serif',serif='palatino')
mat.rcParams['text.latex.preamble']=[r'\usepackage[utf8]{inputenc}',r'\usepackage[T1]{fontenc}',r'\usepackage[spanish]{babel}',r'\usepackage[scaled]{helvet}',r'\renewcommand\familydefault{\sfdefault}',r'\usepackage{amsmath,amsfonts,amssymb}',r'\usepackage{siunitx}']

mips=np.loadtxt('ymip_950V.dat')
x=np.arange(0,320.0,0.5)
distance=21.4*np.array([0,1,2,4,5,6,7,8,9,10,11,12,13])+40.7

atlen=ROOT.TF1('lx','[0]*(exp(-1.0*x*[1])+[2]*exp(-1.0*(660.0-x)*[1]))',10,290)
atgraph=ROOT.TGraph()

fit=True
s=np.all(mips>0,axis=1)
mips=mips[np.nonzero(s)]
K=np.size(mips,0)

p0=226.084,0.00314745,0.549437
p1=192.661,0.00246944,0.498449
ysim0=1.0*(np.exp(-1.0*x*p0[1])+p0[2]*np.exp(-1.0*(660.0-x)*p0[1]))
ysim1=1.0*(np.exp(-1.0*x*p1[1])+p1[2]*np.exp(-1.0*(660.0-x)*p1[1]))

atpars=np.zeros((K,3))
ateror=np.zeros((K,3))
if fit==True:
  for b in range(0,K):
    aT=np.transpose(np.array([distance,mips[b,:]]))
    rnp.fill_graph(atgraph,aT)
    atlen.SetParLimits(2,0,1.0)
    atgraph.Fit(atlen,'q')
    p=atlen.GetParameters()
    ateror[b,0]=atlen.GetParError(0)
    ateror[b,1]=atlen.GetParError(1)
    ateror[b,2]=atlen.GetParError(2)
    atpars[b,:]=p[0],p[1],p[2]
lamb=1.0/atpars[:,1]
lamb_e=np.power(ateror[:,1]/np.power(lamb,2.0),2.0)
lsel=np.logical_and(lamb>100,lamb<500.0)
lamb=lamb[lsel]
lamb_e=lamb_e[lsel]
Natt=np.size(lsel,0)
l_std=lamb_e
reflec=atpars[:,2]
test=np.logical_and(reflec>0.35,reflec<0.99)
reflec=reflec[test]
r_std=np.power(ateror[test,2],2.0)
Nref=np.size(reflec,0)
lw=np.sum(lamb*(1.0/l_std))/np.sum(1.0/l_std)
ew0=np.sum((1.0/l_std)*np.power(lamb-lw,2.0))/(((Natt-1)*np.sum(1.0/l_std)))
rw=np.sum(reflec*(1.0/r_std))/np.sum(1.0/r_std)
ew1=np.sum((1.0/r_std)*np.power(reflec-rw,2.0))/(((Nref-1)*np.sum(1.0/r_std)))

lx=1.0*(np.exp(-1.0*x/lw)+rw*np.exp(-1.0*(660.0-x)/lw))
print(lw,np.sqrt(ew0))
print(rw,np.sqrt(ew1))
c=sns.cubehelix_palette(3,rot=-0.5,dark=0.1,light=0.8,reverse=True)
sns.set_palette(c)

fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True)
h,n,e=ax.hist(reflec+0.15,bins=np.arange(0,1.0,0.05),histtype='stepfilled')
print(np.mean(reflec+0.15),np.std(reflec+0.15))
plt.xlabel('Reflectance',x=0.9,horizontalalignment='right')
plt.ylabel('Counts')
plt.xlim(0.2,1.0)
plt.tight_layout(pad=1.0)
plt.savefig('reflectance.pdf')
fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True)
ax.plot(x,lx*(1.0/np.amax(lx)),lw=1.5,ls=':')
ax.plot(x,ysim0*(1.0/np.amax(ysim0)),lw=1.0,ls='--')
ax.plot(x,ysim1*(1.0/np.amax(ysim1)),lw=2.0)
plt.xlabel(r'Distance from MAPMT $[\si{\cm}]$',x=0.9,horizontalalignment='right')
plt.ylabel(r'Photons $[\si{normalized}]$')
plt.xlim(0,350)
plt.ylim(0.5,1.1)
plt.tight_layout(pad=1.0)
plt.savefig('data_atlength.pdf')
plt.show()
