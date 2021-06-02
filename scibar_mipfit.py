#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ROOT
import root_numpy as rnp

sns.set(rc={"figure.figsize":(8,4)})
sns.set_context('paper',font_scale=1.5,rc={'lines.linewidth':0.5})
sns.set_style('ticks')
mat.rc('text',usetex=True)
mat.rc('font',family='DejaVu Serif',serif='palatino')
mat.rcParams['text.latex.preamble']=[r'\usepackage[utf8]{inputenc}',r'\usepackage[T1]{fontenc}',r'\usepackage[spanish]{babel}',r'\usepackage[scaled]{helvet}',r'\renewcommand\familydefault{\sfdefault}',r'\usepackage{amsmath,amsfonts,amssymb}',r'\usepackage{siunitx}']

f=open('scibar-ay950V.dat','r')
Kbars=np.array([0,1,2,4,5,6,7,8,9,10,11,12,13])
Ksize=13
FEB0=np.hstack((np.arange(32,40),np.arange(48,56)))
FEB1=np.hstack((np.arange(16,24),np.arange(32,40),np.arange(48,56)))+64
FEB2=np.hstack((np.arange(0,8),np.arange(16,24),np.arange(32,40),np.arange(48,56)))+128
FEB3=np.hstack((np.arange(0,8),np.arange(16,24)))+192
FEBs=np.hstack((FEB0,FEB1,FEB2,FEB3))
N=np.size(FEBs,0)
Mbins=2048
bins=np.arange(0,Mbins)
scibar=np.zeros((896,2048,14))
m=0
ksel=0
N=462
Nsel=np.zeros(N,dtype=np.uint16)
for line in f:
  scibar[m,:,:]=np.array(line.split(),dtype=np.uint16).reshape(2048,14)
  if np.amax(scibar[m,:,:])>=20.0:
    Nsel[ksel]=m
    ksel+=1
  m+=1
f.close()

plot=True
fit=False

bar_test=scibar[Nsel,:,:]
distance=10.0*np.array([2,4,10,12,14,16,18,20,22,24,26,28])
mips=np.zeros((N,Ksize))
if fit==True:
  for k in range(74,75):
    m=0
    for b in Kbars:
      bar_dist=ROOT.TF1('Name','landau')
      bar_graph=ROOT.TGraph()
      barT=np.transpose(np.array([bins,bar_test[k,:,b]]))
      rnp.fill_graph(bar_graph,barT)
      bar_graph.Fit(bar_dist)
      bar_pars=bar_dist.GetParameters()
      mips[k,m]=bar_pars[1]
      m+=1
  np.savetxt('ymip_950V.dat',mips,fmt='%1.4f',newline='\n')
  fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True)
  ax.plot(distance,np.transpose(mips))
  plt.yscale('log')
  plt.xlim(0,300)
  plt.show()

pbar=bar_test[74,0::2,:]+bar_test[74,1::2,:]
if plot==True:
  m=64
  c=sns.color_palette(sns.cubehelix_palette(16,start=.5,rot=-.75,reverse=True))
  sns.set_palette(c)
  fig,ax=plt.subplots(nrows=1,ncols=1,sharex=True,sharey=True)
  for ax in axes.flatten():
    ax.plot(bins,bar_test[m,:,:],ds='steps-mid')
    plt.yscale('log')
    plt.xlim(0,600)
    m+=1
  ax.fill_between(bins[0::2],2.5*pbar[:,1],0,step='mid',color=c[0])
  ax.fill_between(bins[0::2],2.0*pbar[:,12],0,step='mid',color=c[5])
  plt.xlabel(r'ADC value',x=0.9,horizontalalignment='right')
  plt.ylabel(r'$\log_{10}\left(\text{Counts}\right)$')
  plt.yscale('log')
  plt.xlim(0,1600)
  plt.ylim(1e0,2e2)
  plt.tight_layout(pad=1.0)
  plt.savefig('adc_attenuation.pdf')
  plt.show()
