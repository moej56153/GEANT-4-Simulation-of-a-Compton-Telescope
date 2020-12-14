import numpy as np
import matplotlib.pyplot as plt
import reading_data as rd
import ang_tof as at
import att_func as af
import pos_dis as pd
import en_dis as ed
import mono as mo
import par_cuts as pc
from scipy.optimize import minimize
from scipy.integrate import quad
from matplotlib.lines import Line2D
import matplotlib.image as mpimg



p_data='C:\\Users\\moell\\Desktop\\COMPTEL_Simulation_Project\\Werkstudent\\Data_from_Simulations/'
p_plots='C:\\Users\\moell\\Desktop\\COMPTEL_Simulation_Project\\Werkstudent\\Plots/'
p_werner_cuts="C:\\Users\moell\Desktop\COMPTEL_Simulation_Project\Werkstudent\Werner_Cuts/"



def E1_E2_2D_hist():
    fig=plt.figure(figsize=(8,12))
    grid=plt.GridSpec(4, 2,hspace=0.15,wspace=0.30)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("E1 [MeV]",labelpad=30)
    pa.set_ylabel("E2 [MeV]",labelpad=40)

    rows={0:"1500",1:"5000",2:"15000",3:"50000"}
    energies={0:"1.5",1:"5",2:"15",3:"50"}
    columns={0:True,1:False}
    
    letters=np.array([["a","b"],
                    ["c","d"],
                    ["e","f"],
                    ["g","h"]])
    
    num_bins=100
    
    for row in range(len(rows)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            d=rd.read_h5py_file(p_data+"Thesis_Simulations/"+"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
            d=rd.cut_selection(d,9,0,columns[column])
            if columns[column]:
                E1=rd.extract_parameter(d, 3)/1000
                E2=rd.extract_parameter(d, 1)/1000
            else:
                E1=rd.extract_parameter(d, 1)/1000
                E2=rd.extract_parameter(d, 3)/1000
                
            logbins1 = np.geomspace(E1.min(), E1.max(), num_bins)
            logbins2 = np.geomspace(E2.min(), E2.max(), num_bins)
            
            counts, _, _ = np.histogram2d(E1, E2, bins=(logbins1, logbins2))

            temp.pcolormesh(logbins1, logbins2, counts.T)
            temp.plot()
            temp.set_xscale('log')
            temp.set_yscale('log')
            
            
            temp.plot(logbins1,float(energies[row])-logbins1,color="C1",lw=1.0)
            
            
            plt.ylim(logbins2[0],logbins2[-1])
            if column==1:
                if row==1:
                    plt.ylim(1.5,logbins2[-1])
                elif row==2:
                    plt.ylim(5,logbins2[-1])
                elif row==3:
                    plt.ylim(10,logbins2[-1])
            
            if row==0:
                if column==0:
                    temp.set_xlabel("Forward Photons")
                    temp.xaxis.set_label_position("top")
                elif column==1:
                    temp.set_xlabel("Backward Photons")
                    temp.xaxis.set_label_position("top")

            if column==1:
                temp.set_ylabel(energies[row]+"MeV")
                temp.yaxis.set_label_position("right")
            

            
            
            if column==0:
                angs=(1,10,20,30,40,50)
                ang_lines=np.zeros((len(angs),num_bins))
                
                for ang in range(len(angs)):
                    for b in range(num_bins):
                        e_sca=logbins2[b]
                        wl_sca=6.625e-34*3e8/(1.602e-19*float(e_sca)*1e6)
                        wl_ini=wl_sca-2.426e-12*(1-np.cos(angs[ang]*np.pi/180))
                        e_ini=6.625e-34*3e8/(1.602e-19*wl_ini)*1e-6
                        ang_lines[ang,b]=e_ini-e_sca if e_ini-e_sca>0 else float("NaN")

                for ang in range(len(angs)):
                    temp.plot(ang_lines[ang],logbins2,color="C6",lw=1)
                    for b in range(num_bins):
                        if ang_lines[ang,b]>E1.min():
                            plt.text(ang_lines[ang,b],logbins2[b],str(angs[ang])+"$^\circ$",color="bisque")
                            break
                        
            
            else:
                mult=10
                logbinse=np.geomspace(E1.min(), E1.max(), num_bins*mult)
                angs=(180,140)
                ang_lines=np.zeros((len(angs),num_bins*mult))
                
                for ang in range(len(angs)):
                    for b in range(num_bins*mult):
                        e_sca=logbinse[b]
                        wl_sca=6.625e-34*3e8/(1.602e-19*float(e_sca)*1e6)
                        wl_ini=wl_sca-2.426e-12*(1-np.cos(angs[ang]*np.pi/180))
                        e_ini=6.625e-34*3e8/(1.602e-19*wl_ini)*1e-6
                        ang_lines[ang,b]=e_ini-e_sca if e_ini-e_sca>0 else float("NaN")

                left_shift=((3/4,1),
                            (2/3,1),
                            (3/5,1),
                            (1/2,1))
                for ang in range(len(angs)):
                    temp.plot(logbinse,ang_lines[ang],color="C6",lw=1)
                    for b in range(num_bins*mult):
                        if ang_lines[ang,b]>E2.min():
                            plt.text(logbinse[b]*left_shift[row][ang],ang_lines[ang,b],str(angs[ang])+"$^\circ$",color="bisque")
                            break
                    
            plt.xlim(E1.min(),np.amax(E1))
            plt.ylim(E2.min(),np.amax(E2))
            
            
            
            temp.text(-0.10,0.97,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='center')
                
        plt.savefig(p_plots+'E1_E2_2D_hist.pdf',bbox_inches='tight')


def crab_outlier_distributions():
    fig=plt.figure(figsize=(10,13))
    grid1=plt.GridSpec(1, 3,hspace=0.2,wspace=0.25,bottom=0.73)
    grid2=plt.GridSpec(3,3,hspace=0.05,wspace=0.05,top=0.65)
    
    
    
    d=rd.read_h5py_file(p_data+"Thesis_Simulations/Simulation_Batch_3/","Simulation1")
    d=rd.cut_selection(d,9,0,True)

    c=2.998e8
    m=511/c**2
    rang=0.08
    dc=d[np.abs( np.minimum( np.abs(np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])*np.sign(d[:,11]-d[:,5]) - np.arccos( 1 - (d[:,3]/d[:,1]*m*c**2/(d[:,3]+d[:,1])) ) ) ,
                       np.abs(np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])*np.sign(d[:,11]-d[:,5]) + np.arccos( 1 - (d[:,3]/d[:,1]*m*c**2/(d[:,3]+d[:,1])) ) )
                       ))>rang]
    
    temp=fig.add_subplot(grid1[0,0])
    n1,bins1=pc.create_ARM_plot(dc,200,1,True)
    n2,bins2=pc.create_ARM_plot(d,200,1,False)
    temp.axhline(0,c="black",lw=0.8)
    plt.plot(bins1,n2,c="C2")
    plt.plot(bins1,n1-n2,c="C3")
    temp.set_xlabel("Angle of Incidence [rad]")
    temp.set_ylabel("Frequency [Counts/Bin]")
    temp.text(0.01,0.99,"(a)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid1[0,1])
    bins1,n1=pc.plot_flight_angle(dc,False,50)
    bins2,n2=pc.plot_flight_angle(d,True,50)
    n2=n2/np.sum(n2)*np.sum(n1)
    temp.axhline(0,c="black",lw=0.8)
    plt.plot(bins1,n2,c="C2")
    plt.plot(bins1,n1-n2,c="C3")
    temp.set_xlabel("Flight Angle [rad]")
    secax=temp.secondary_xaxis("top",functions=(lambda d:d*180/np.pi,lambda r:r*np.pi/180))
    secax.set_xlabel("Flight Angle [degrees]")
    temp.text(0.01,0.99,"(b)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid1[0,2])
    dct=dc[np.logical_and(dc[:,17]>4e-9,dc[:,17]<7.5e-9)]
    dt=d[np.logical_and(d[:,17]>4e-9, d[:,17]<7.5e-9)]
    bins1,n1=pc.plot_ToF(dct,False,50)
    bins2,n2=pc.plot_ToF(dt,True,50)
    n2=n2/np.sum(n2)*np.sum(n1)
    temp.axhline(0,c="black",lw=0.8)
    plt.plot(bins1,n2,c="C2")
    plt.plot(bins1,n1-n2,c="C3")
    temp.set_xlabel("Time of Flight [ns]")
    temp.text(0.01,0.99,"(c)",transform=temp.transAxes,ha='left', va='top')
    
    emp=fig.add_subplot(grid2[2,0])
    emp.spines['top'].set_color('none')
    emp.spines['bottom'].set_color('none')
    emp.spines['left'].set_color('none')
    emp.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    custom_lines = [Line2D([0], [0], color="C0", lw=10),
                            Line2D([0], [0], color="C2", lw=1.5),
                            Line2D([0], [0], color="C3", lw=1.5),]
    plt.legend(custom_lines, ['Crab after Cut', 'Crab before Cut' , 'Residual'],loc="lower left")
    
    
    
    E1_min=0.04
    E2_min=0.4
    E_max=20
    
    temp=fig.add_subplot(grid2[0:2,0])
    bins1,n1=pc.plot_E2_vert(dc,False,50)
    bins2,n2=pc.plot_E2_vert(d,True,50)
    n2=n2/np.sum(n2)*np.sum(n1)
    temp.axvline(0,c="black",lw=0.8)
    plt.plot(n2,bins1,c="C2")
    plt.plot(n1-n2,bins1,c="C3")
    plt.ylim(E2_min,E_max)
    temp.set_ylabel("E2 [MeV]")
    temp.set_xlabel("Frequency\n[Counts/Bins]")
    temp.text(0.01,0.99,"(d)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid2[2,1:3])
    bins1,n1=pc.plot_E1(dc,False,50)
    bins2,n2=pc.plot_E1(d,True,50)
    n2=n2/np.sum(n2)*np.sum(n1)
    temp.axhline(0,c="black",lw=0.8)
    plt.plot(bins1,n2,c="C2")
    plt.plot(bins1,n1-n2,c="C3")
    temp.set_xlabel("E1 [MeV]")
    temp.set_ylabel("Frequency\n[Counts/Bins]")
    temp.text(0.01,0.99,"(f)",transform=temp.transAxes,ha='left', va='top')
    
    plt.xlim(E1_min,E_max)
    temp=fig.add_subplot(grid2[0:2,1:3])
    pcmap=pc.plot_E1_E2(d,temp,label_pos_left=False)
    plt.colorbar(pcmap,ax=emp,orientation="horizontal",).set_label("Frequency [Counts/Bin]")
    plt.xlim(E1_min,E_max)
    plt.ylim(E2_min,E_max)
    temp.xaxis.tick_top()
    temp.yaxis.tick_right()
    temp.xaxis.set_label_position("top")
    temp.yaxis.set_label_position("right")
    temp.set_xlabel("E1 [MeV]")
    temp.set_ylabel("E2 [MeV]")
    temp.text(0,1,"(e)",transform=temp.transAxes,ha='left', va='top',bbox=dict(facecolor='white', edgecolor='none'))
    

    
    
    
    
    
    
    plt.savefig(p_plots+'Crab_ARM_Cut.pdf',bbox_inches='tight')


def werner_cut_comparison():
    fig=plt.figure(figsize=(10,10))
    
    cut1_plot=mpimg.imread(p_werner_cuts+"Cut8.jpg")
    plt.imshow(cut1_plot)
    
    d=rd.read_h5py_file(p_data,"Thesis_Simulations\\Real_COMPTEL_data"+"\\h5_vp426_09-43")
    dc=rd.read_h5py_file(p_data,"Thesis_Simulations\\Real_COMPTEL_data"+"\\h5_vp426_09-43_c")
    
    
    
    E_e=dc[:,3]
    E_g=dc[:,1]
    c=2.998e8
    m=511/c**2
    phibar=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    phigeo=np.arccos( -( -dc[:,9]+dc[:,15] ) / dc[:,24])
    ARM=phibar-phigeo
    
    for i in range(20):
        print(i)
        print("Phibar:", phibar[i]*180/np.pi,d[i,5])
        print("ARM:",ARM[i]*180/np.pi,d[i,14])
    





