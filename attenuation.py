#!/usr/bin/env python3.7
# -*- coding: utf8 -*-

import time
import datetime
import subprocess
import io
import skimage.morphology as morph
import numpy as np
import argparse
import os

def vertical_dist(home,nfile,all_bebs,ped,pscibar,adc_position):
  fs=[]
  for yx in all_bebs:
    name='{0}/{1}.{2}.gz'.format(home,nfile[:-7],yx)
    p=subprocess.Popen(['/usr/bin/zcat',name],stdout=subprocess.PIPE)
    f=io_method(p.communicate()[0])
    assert p.returncode==0
    fs.append(f)
  adc_thr=70
  ls=['0','0','0','0','0','0']
  eventnum=0
  muons=0
  yxhits=np.zeros([64,4],dtype=np.int16)
  for sl in fs[0]:
    ls[0],ls[1],ls[2]=fs[1].readline(),fs[2].readline(),fs[3].readline()
    ls[3],ls[4],ls[5]=fs[4].readline(),fs[5].readline(),fs[6].readline()
    ls_test=len(sl)+len(ls[0])+len(ls[1])+len(ls[2])+len(ls[3])+len(ls[4])+len(ls[5])
    if ls_test>9100:
      dbeb20=np.fromstring(sl[15:],dtype=np.int16,count=256,sep=' ').reshape(64,4)[pscibar]
      dbeb21=np.fromstring(ls[0][15:],dtype=np.int16,count=256,sep=' ').reshape(64,4)[pscibar]
      dbeb22=np.fromstring(ls[1][15:],dtype=np.int16,count=256,sep=' ').reshape(64,4)[pscibar]
      dbeb23=np.fromstring(ls[2][15:],dtype=np.int16,count=256,sep=' ').reshape(64,4)[pscibar]
      dbeb24=np.fromstring(ls[3][15:],dtype=np.int16,count=256,sep=' ').reshape(64,4)[pscibar]
      dbeb25=np.fromstring(ls[4][15:],dtype=np.int16,count=256,sep=' ').reshape(64,4)[pscibar]
      dbeb26=np.fromstring(ls[5][15:],dtype=np.int16,count=256,sep=' ').reshape(64,4)[pscibar]
      yhits0=dbeb20-ped[0,:,:]
      yhits1=dbeb21-ped[1,:,:]
      yhits2=dbeb22-ped[0,:,:]
      yxhits[:,:2]=dbeb23[:,:2]-ped[1,:,:2]
      yxhits[:,2:]=dbeb23[:,2:]-ped[3,:,2:]
      xhits1=dbeb24-ped[2,:,:]
      xhits2=dbeb25-ped[3,:,:]
      xhits3=dbeb26-ped[4,:,:]
      nbeb_y0=np.transpose(yhits0).reshape(4,8,8)
      nbeb_y1=np.transpose(yhits1).reshape(4,8,8)
      nbeb_y2=np.transpose(yhits2).reshape(4,8,8)
      nbeb_y3=np.transpose(yxhits[:,:2]).reshape(2,8,8)
      nbeb_x0=np.transpose(yxhits[:,2:]).reshape(2,8,8)
      nbeb_x1=np.transpose(xhits1).reshape(4,8,8)
      nbeb_x2=np.transpose(xhits2).reshape(4,8,8)
      nbeb_x3=np.transpose(xhits3).reshape(4,8,8)
      yside=np.vstack((nbeb_y0,nbeb_y1,nbeb_y2,nbeb_y3))
      xside=np.vstack((nbeb_x0,nbeb_x1,nbeb_x2,nbeb_x3))
      yside[yside<0]=0
      xside[xside<0]=0
      ybars=True*(yside>adc_thr)
      xbars=True*(xside>adc_thr)
      ymask=morph.remove_small_objects(ybars,min_size=2,connectivity=2)
      xmask=morph.remove_small_objects(xbars,min_size=2,connectivity=2)
      num_yhits=np.sum(ymask)
      num_xhits=np.sum(xmask)
      min_bars=num_yhits<=16.0 and num_xhits<=16.0
      if min_bars:
        eventnum+=1
        ylays=np.any(ymask>0,axis=2)
        xlays=np.any(xmask>0,axis=2)
        muon_y=np.sum(ylays,axis=1)
        muon_x=np.sum(xlays,axis=1)
        muon_trgy=np.any(muon_y>=1) and np.sum(muon_y>0)==1
        muon_trgx=np.any(muon_x>=2) and np.sum(muon_x>0)==1
        muon_singleh=muon_trgx and muon_trgy
        if muon_singleh:
          muons+=1
          px_trg=np.argmax(muon_x)
          a,b,c=np.nonzero(ymask)
          py_trg=c+8*a+112*b
          adc_position[yside[ymask],py_trg,px_trg]+=1.0
  return eventnum,muons

parser=argparse.ArgumentParser()
parser.add_argument('home', help='home',type=str)

args=parser.parse_args()
home=args.home
io_method=io.BytesIO

t0=time.time()

npos='pscibar.txt'
f=open(npos,'r')
pos=np.array(f.readline()[4:-1].split(),dtype=np.uint8)
f.close()
rot=True

if rot==True:
  pos_90=np.rot90(pos.reshape(8,8),1)
  pos=np.ravel(pos_90)

all_bebs=[20,21,22,23,24,25,26]
pedestal=np.zeros([7,64,4],dtype=np.int16)
nbeb=0
for beb in all_bebs:
  nped='ped-muon-950-{0}.txt'.format(beb)
  p=np.loadtxt(nped,delimiter=None,dtype=np.int16)+90
  pedestal[nbeb,:,:]=np.transpose(p.reshape(4,64))[pos]
  nbeb+=1

mfile='muon_files.txt'
fmuon=open(mfile,'r')
time_stamp=fmuon.readlines()
fmuon.close()

nbins=2**11
adc_position=np.zeros([nbins,896,14])

name='scibar-ay950V.dat'
fdat=open(name,'w')
for times in time_stamp:
  print('Fecha {0}'.format(times[5:-9]))
  ibento,total_mu=vertical_dist(home,times,all_bebs,pedestal,pos,adc_position)

  print('Eventos totales: {0}'.format(ibento))
  print('Muones totales: {0}'.format(total_mu))

for j in range(0,896):
  np.savetxt(fdat,adc_position[:,j,:],fmt='%d',newline=' ')
  fdat.write('\n')

fdat.close()
t1=time.time()
dt=str(datetime.timedelta(seconds=(t1-t0)))[0:7]
print('Tiempo total {0} hrs.'.format(dt))
