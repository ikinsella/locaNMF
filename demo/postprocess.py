#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:37:53 2020

@author: shreyasaxena
"""
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import numpy as np
import os

def parse_areanames(areanames_):
    areainds=[]; areanames=[];
    for i,area in enumerate(areanames_[0][0]):
        areainds.append(area[0][0])
        areanames.append(areanames_[0][0].dtype.descr[i][0])
    sortvec=np.argsort(np.abs(areainds))
    areanames=[areanames[i] for i in sortvec]
    areainds=[areainds[i] for i in sortvec]
    return areainds,areanames

def plot_components(A,C,areas,atlas,areanames_,outputfolder):
   # Make figures directory
    figdir=os.path.join(outputfolder,'Figures')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    
    areainds,areanames = parse_areanames(areanames_)

    print('plotting components',flush=True)
    # One figure per component
    if not os.path.exists(os.path.join(figdir,'AllComponents')):
        os.makedirs(os.path.join(figdir,'AllComponents'))

    for i,area in enumerate(areas):
        try:
            areadir=os.path.join(figdir,'AllComponents',areanames[areainds.index(area)])
            if not os.path.exists(areadir):
                os.makedirs(areadir)
            fig=plt.figure()
            ax1 = fig.add_subplot(2,2,1)
            ax1.imshow(atlas==area);
            ax1.set_title('Atlas '+areanames[areainds.index(area)])
            ax1.axis('off')
            
            ax2 = fig.add_subplot(2,2,2)
            ax2.imshow(A[:,:,i])
            ax2.set_title('A [%s]'%(i+1))
            ax2.axis('off')
            
            ax3 = fig.add_subplot(2,1,2)
            ax3.plot(C[i,:min(1000,C.shape[1])])
            ax3.set_title('C [%s]'%(i+1))
            fig.savefig(os.path.join(areadir,'Component_'+str(i+1)+'.png'))
            plt.close()
        except:
            pass

    # Spatial and Temporal Components: Summary
    atlascolor=np.zeros((atlas.shape[0],atlas.shape[1],4))
    A_color=np.zeros((A.shape[0],A.shape[1],4))
    cmap=plt.cm.get_cmap('jet')
    colors=cmap(np.arange(len(areainds))/len(areainds))
    for i,area_i in enumerate(areainds):
        if area_i not in areas:
            continue
        atlascolor[atlas==area_i,:]=colors[i,:]
        C_area=C[np.where(areas==area_i)[0],:]
        for j in np.arange(colors.shape[1]):
            A_color[:,:,j]=A_color[:,:,j]+colors[i,j]*A[:,:,np.where(areas==area_i)[0][0]]
    fig=plt.figure(figsize=(10,10))
    ax1=fig.add_subplot(2,2,1)
    ax1.imshow(atlascolor)
    ax1.set_title('Atlas Regions')
    ax1.axis('off')
    ax2=fig.add_subplot(2,2,3)
    ax2.imshow(A_color)
    ax2.set_title('Spatial Components (One per region)')
    ax2.axis('off')
    ax3=fig.add_subplot(1,2,2)
    axvar=0
    for i,area_i in enumerate(areainds):
        if area_i not in areas:
            continue
        C_area=C[np.where(areas==area_i)[0][0],:min(1000,C.shape[1])]
        ax3.plot(1.5*axvar+C_area/np.nanmax(np.abs(C_area)),color=colors[i,:])
        axvar+=1
    ax3.set_title('Temporal Components (One per region)')
    ax3.axis('off')
    fig.savefig(os.path.join(figdir,'SummaryComponents.png'))
    plt.close()
    
    return figdir

def plot_correlations(A,C,areas,atlas,areanames_,outputfolder):
    # Correlation plot
   # Make figures directory
    figdir=os.path.join(outputfolder,'Figures')
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    
    areainds,areanames = parse_areanames(areanames_)

    # Preprocess C to remove nans
    keepinds=np.nonzero(np.sum(np.isfinite(C),axis=0))[0]
    C=C[:,keepinds]
    corrmat=np.zeros((len(areainds),len(areainds)))
    skipinds=[]
    for i,area_i in enumerate(areainds):
        for j,area_j in enumerate(areainds):
            if i==0 and area_j not in areas:
                skipinds.append(j)
            C_i=C[np.where(areas==area_i)[0],:].T
            C_j=C[np.where(areas==area_j)[0],:].T
            if i not in skipinds and j not in skipinds:
                cca=CCA(n_components=1)
                cca.fit(C_i,C_j)
                C_i_cca,C_j_cca=cca.transform(C_i,C_j)
                try:
                    C_i_cca=C_i_cca[:,0]
                except:
                    pass
                try:
                    C_j_cca=C_j_cca[:,0]
                except:
                    pass               
                corrmat[i,j]=np.corrcoef(C_i_cca,C_j_cca)[0,1]
    corrmat=np.delete(corrmat,skipinds,axis=0); 
    corrmat=np.delete(corrmat,skipinds,axis=1);
    corr_areanames=np.delete(areanames,skipinds)
    print('plotting correlations',flush=True)
    fig=plt.figure(figsize=(16,16))
    plt.imshow(corrmat,cmap=plt.cm.get_cmap('jet')); plt.colorbar(shrink=0.8)
    plt.get_cmap('jet')
    plt.xticks(ticks=np.arange(len(areainds)-len(skipinds)),labels=corr_areanames,rotation=90); 
    plt.yticks(ticks=np.arange(len(areainds)-len(skipinds)),labels=corr_areanames); 
    plt.title('CCA between all regions',fontsize=36)
    plt.xlabel('Region i',fontsize=30)
    plt.ylabel('Region j',fontsize=30)
    fig.savefig(os.path.join(figdir,'CorrelationPlot.png'))
    plt.close()
    
    
    return figdir
