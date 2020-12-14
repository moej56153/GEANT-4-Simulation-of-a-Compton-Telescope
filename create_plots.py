import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import reading_data as rd
import ang_tof as at
import att_func as af
import pos_dis as pd
import en_dis as ed
import mono as mo
import par_cuts as pc
from scipy.optimize import minimize
from scipy.integrate import quad




def plot_scattering_angle_with_removed_outliers():
    fig=plt.figure(figsize=(12,13))
    grid=plt.GridSpec(3, 2,hspace=0.15,wspace=0.15)
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    pa.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    pa.set_xlabel("Scattering Angle [rad]",labelpad=9)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=15)
    
    p1=fig.add_subplot(grid[0,0])
    pd.plot_ang_dist(True,100)
    p1.text(0.05,0.95,"(a)",transform=p1.transAxes,ha='center', va='center')
    p1.set_xlabel("Forward Photons")
    p1.xaxis.set_label_position("top")
    
    p2=fig.add_subplot(grid[0,1])
    p2.text(0.05,0.95,"(b)",transform=p2.transAxes,ha='center', va='center')
    p2.set_xlabel("Backward Photons")
    p2.xaxis.set_label_position("top")
    p2.set_ylabel("No Cuts")
    p2.yaxis.set_label_position("right")
    pd.plot_ang_dist(False,30)
    
    d3=mo.remove_vel_out(True,1.2*10**9)
    d4=mo.remove_vel_out(False,1.2*10**9)
    d5=mo.remove_sa_out(True,0.025)
    d6=mo.remove_sa_out(False,0.025)
    a3,a4,a5,a6=rd.calculate_angles(d3),rd.calculate_angles(d4),rd.calculate_angles(d5),rd.calculate_angles(d6)
    
    p3=fig.add_subplot(grid[1,0])
    pd.plot_ang_dist(True,50,a3)
    p3.text(0.05,0.95,"(c)",transform=p3.transAxes,ha='center', va='center')
    p4=fig.add_subplot(grid[1,1])
    p4.text(0.05,0.95,"(d)",transform=p4.transAxes,ha='center', va='center')
    p4.set_ylabel("Velocity Cuts")
    p4.yaxis.set_label_position("right")
    pd.plot_ang_dist(False,30,a4)
    
    p5=fig.add_subplot(grid[2,0])
    pd.plot_ang_dist(True,50,a5)
    p5.text(0.05,0.95,"(e)",transform=p5.transAxes,ha='center', va='center')
    p6=fig.add_subplot(grid[2,1])
    p6.text(0.05,0.95,"(f)",transform=p6.transAxes,ha='center', va='center')
    p6.set_ylabel("Scattering Angle Cuts")
    p6.yaxis.set_label_position("right")
    pd.plot_ang_dist(False,15,a6)

    plt.savefig(rd.p2+'SAD_wo_out.pdf',bbox_inches='tight')


def plot_parameter_cut_example():
    y_lim=2.9
    y_lim2=1.1
    
    x=np.linspace(-1,1,100)
    x2=np.linspace(-0.9,1.1,100)
    x3=np.linspace(-1.2,0.8,100)
    y_source=at.normal_dis(x,0.25)
    y_background1=np.array([1 for i in x])
    y_background2=np.linspace(2,0,100)
    y_both1=y_source+y_background1
    y_both2=y_source+y_background2
    y_ex=(at.normal_dis(x2,0.15)/1.2+np.linspace(1,2,100))/1.5
    y_ex2=(at.normal_dis(x3,0.05)/6+np.sin(7*x)/5+np.array([2.5 for i in x]))/1.5
    y_ex3=y_source-at.normal_dis(x2,0.02)/20+at.normal_dis(x3,0.03)/15
    
    norm1=y_source/np.amax(y_source)
    norm2=y_source/y_both2
    norm2=norm2/np.amax(norm2)
    norm3=norm1*norm2
    norm3=norm3/np.amax(norm3)
    norm4=np.sqrt(norm1)*np.sqrt(norm2)
    norm4=norm4/np.amax(norm4)
    
    
    fig=plt.figure(figsize=(9,11))
    grid=plt.GridSpec(9, 7,hspace=0.10,wspace=0.10)
    
    p1=fig.add_subplot(grid[0,1:])
    p1.spines['top'].set_color('none')
    p1.spines['bottom'].set_color('none')
    p1.spines['left'].set_color('none')
    p1.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    p1.set_xlabel("Parameter Cut")
    p1.xaxis.set_label_position("top")
    
    p2=fig.add_subplot(grid[1:,0])
    p2.spines['top'].set_color('none')
    p2.spines['bottom'].set_color('none')
    p2.spines['left'].set_color('none')
    p2.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    p2.set_ylabel("Parameter Distribution")
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    

    
    if True:
        N1=fig.add_subplot(grid[0,1])
        plt.plot(x,norm1,color="#2ca02c")
        plt.ylim(0,y_lim2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(a)",transform=N1.transAxes,va='center')
        
        N2=fig.add_subplot(grid[0,2])
        plt.plot(x,norm2,color="#2ca02c")
        plt.ylim(0,y_lim2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(b)",transform=N2.transAxes,va='center')
        
        N3=fig.add_subplot(grid[0,3])
        plt.plot(x,norm3,color="#2ca02c")
        plt.ylim(0,y_lim2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(c)",transform=N3.transAxes,va='center')
        
        N4=fig.add_subplot(grid[0,4])
        plt.plot(x,norm4,color="#2ca02c")
        plt.ylim(0,y_lim2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(d)",transform=N4.transAxes,va='center')
        
        N5=fig.add_subplot(grid[0,5])
        plt.plot(x,norm1,color="C4")
        plt.ylim(0,y_lim2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(e)",transform=N5.transAxes,va='center')
        
        N6=fig.add_subplot(grid[0,6])
        plt.plot(x,norm1,color="C6")
        plt.ylim(0,y_lim2)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(f)",transform=N6.transAxes,va='center')
        
        D1=fig.add_subplot(grid[1,0])
        plt.plot(x,y_source,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(i)",transform=D1.transAxes,va='center')
        
        D2=fig.add_subplot(grid[2,0])
        plt.plot(x,y_background1,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(ii)",transform=D2.transAxes,va='center')
        
        D3=fig.add_subplot(grid[3,0])
        plt.plot(x,y_background2,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(iii)",transform=D3.transAxes,va='center')
        
        D4=fig.add_subplot(grid[4,0])
        plt.plot(x,y_both1,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(iv)",transform=D4.transAxes,va='center')
        
        D5=fig.add_subplot(grid[5,0])
        plt.plot(x,y_both2,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(v)",transform=D5.transAxes,va='center')
        
        D6=fig.add_subplot(grid[6,0])
        plt.plot(x,y_ex,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(vi)",transform=D6.transAxes,va='center')
        
        D7=fig.add_subplot(grid[7,0])
        plt.plot(x,y_ex2,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(vii)",transform=D7.transAxes,va='center')
        
        D8=fig.add_subplot(grid[8,0])
        plt.plot(x,y_ex3,color="#ff7f0e")
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.text(0.01,0.89,"(viii)",transform=D8.transAxes,va='center')
    
    norm={1:norm1,2:norm2,3:norm3,4:norm4}
    dist={1:y_source,2:y_background1,3:y_background2,4:y_both1,5:y_both2,6:y_ex,7:y_ex2,8:y_ex3}
    
    for row in range(1,len(dist)+1):
        for column in range(1,len(norm)+1):
            fig.add_subplot(grid[row,column])
            plt.plot(x,dist[row]*norm[column])
            plt.ylim(0,y_lim)
            plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        fig.add_subplot(grid[row,5])
        a=minimize(rd.fit_func,1,(dist[row],y_source)).x
        plt.plot(x,np.minimum(dist[row],a*y_source))
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        
        fig.add_subplot(grid[row,6])
        a=np.min(dist[row]/y_source)
        plt.plot(x,np.minimum(dist[row],y_source*a))
        plt.ylim(0,y_lim)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    
    pa.axvline(0.1382,lw=4.5,c="gray")
    pa.axhline(0.8929,lw=4.5,c="gray")
    plt.savefig(rd.p2+'pce.pdf',bbox_inches='tight')


def plot_scattering_angle_via_energies():
    fig=plt.figure(figsize=(12,9))
    grid=plt.GridSpec(2, 2,hspace=0.15,wspace=0.15)
    
    p1=fig.add_subplot(grid[:,:])
    p1.spines['top'].set_color('none')
    p1.spines['bottom'].set_color('none')
    p1.spines['left'].set_color('none')
    p1.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    p1.set_xlabel("Scattering Angle via Energies [rad]",labelpad=25)

    p1=fig.add_subplot(grid[0,0])
    ed.SAD_E(True)
    p1.text(0.05,0.95,"(a)",transform=p1.transAxes,ha='center', va='center')
    p1.set_ylabel("Frequency [Counts/Bin]")
    p1.set_xlabel("Forward Photons")
    p1.xaxis.set_label_position("top")
    p2=fig.add_subplot(grid[0,1])
    p2.text(0.05,0.95,"(b)",transform=p2.transAxes,ha='center', va='center')
    p2.set_xlabel("Backward Photons")
    p2.xaxis.set_label_position("top")
    ed.SAD_E(False,30)
    p3=fig.add_subplot(grid[1,0])
    p3.text(0.05,0.95,"(c)",transform=p3.transAxes,ha='center', va='center',bbox=dict(facecolor='white', edgecolor='none'))
    ed.SAD_E_vs_Pos(True)
    #plt.xlim(None,1)
    p3.set_ylabel("Scattering Angle via Positions [rad]")
    p4=fig.add_subplot(grid[1,1])
    p4.text(0.05,0.95,"(d)",transform=p4.transAxes,ha='center', va='center',bbox=dict(facecolor='white', edgecolor='none'))
    ed.SAD_E_vs_Pos(False,50)
    
    plt.savefig(rd.p2+'SAD_E.pdf',bbox_inches='tight')


def plot_velocity_distribution():
    fig=plt.figure(figsize=(12,5))
    grid=plt.GridSpec(1, 2,hspace=0.15,wspace=0.15)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Velocity [cm/s]",labelpad=40)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=40)
    
    c=2.998e10
    sig_t=3.53553e-10
    sig_D1=rd.D1_depth/2/np.sqrt(3)
    sig_D2=rd.D2_depth/2/np.sqrt(3)
    
    E_dist=np.geomspace(0.7,50,100)
    s1f=0
    s2=0
    for i in range(len(E_dist)):
        temp1f=E_dist[i]*at.energy_distribution(E_dist[i]*1000)*mo.calculate_total_distance(E_dist[i],True)
        temp2=E_dist[i]*at.energy_distribution(E_dist[i]*1000)
        if not np.isnan(temp1f):
            s1f+=temp1f
            s2+=temp2
    av_df=s1f/s2
    av_velf=rd.D1_D2_Distance*c/av_df
    #av_velf=av_df/rd.D1_D2_Distance*c
    
    s1b=0
    s2=0
    for i in range(len(E_dist)):
        temp1b=E_dist[i]*at.energy_distribution(E_dist[i]*1000)*mo.calculate_total_distance(E_dist[i],False)
        temp2=E_dist[i]*at.energy_distribution(E_dist[i]*1000)
        if not np.isnan(temp1b):
            s1b+=temp1b
            s2+=temp2
    av_db=s1b/s2
    av_velb=rd.D1_D2_Distance*c/av_db
    #av_velb=av_db/rd.D1_D2_Distance*c

    sig=np.sqrt( (sig_D1*c/rd.D1_D2_Distance)**2 + (sig_D2*c/rd.D1_D2_Distance)**2 + (sig_t*c**2/rd.D1_D2_Distance)**2 )
    
    
    d=rd.read_h5py_file()
    df=rd.cut_selection(d, 9, 0, True)
    db=rd.cut_selection(d, 9, 0, False)
    vf=rd.calculate_velocity(df)
    vb=rd.calculate_velocity(db)
    vf=vf[vf[:]>0]
    vf=vf[vf[:]<5*10e9]
    vb=vb[vb[:]>0]
    vb=vb[vb[:]<5*10e9]
    
    
    p1=fig.add_subplot(grid[0,0])
    p1.set_xlabel("Forward Photons")
    p1.xaxis.set_label_position("top")
    n,bins,patches=plt.hist(vf,200)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    velf=np.zeros(len(x_dist))
    for i in range(len(velf)):
        velf[i]=at.normal_dis(av_velf-(x_dist[i]),sig)*1e20
    a=minimize(rd.fit_func,np.amax(n)/np.amax(velf)*0.5,(n,velf)).x
    plt.plot(x_dist,velf*a,lw=2)
    p1.text(0.05,0.95,"(a)",transform=p1.transAxes,ha='center', va='center')
    plt.xlim(1.75*10e9,4.25*10e9)
    
    vf='{:.3f}'.format(round(av_velf/1e10, 3))
    sigma='{:.3f}'.format(round(sig/1e10, 3))
    p1.text(0.80,0.90,"v="+vf+"e10[cm/s]\n $\Delta$v="+sigma+"e10[cm/s]",transform=p1.transAxes,ha='center', va='center')
    
    
    
    p2=fig.add_subplot(grid[0,1])
    p2.set_xlabel("Backward Photons")
    p2.xaxis.set_label_position("top")
    n,bins,patches=plt.hist(vb,100)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    velb=np.zeros(len(x_dist))
    for i in range(len(velb)):
        velb[i]=at.normal_dis(av_velb-(x_dist[i]),sig)*1e20
    a=minimize(rd.fit_func,np.amax(n)/np.amax(velb)*0.5,(n,velb)).x
    plt.plot(x_dist,velb*a,lw=2)
    p2.text(0.05,0.95,"(b)",transform=p2.transAxes,ha='center', va='center')
    plt.xlim(1.75*10e9,4.25*10e9)
    
    vb='{:.3f}'.format(round(av_velb/1e10, 3))
    sigma='{:.3f}'.format(round(sig/1e10, 3))
    p2.text(0.80,0.90,"v="+vb+"e10cm/s\n $\Delta$v="+sigma+"e10cm/s",transform=p2.transAxes,ha='center', va='center')
    
    plt.savefig(rd.p2+'vel.pdf',bbox_inches='tight')
    

def plot_scattering_angle_for_monochromatic_sources_forward():
    fig=plt.figure(figsize=(12,12))
    grid=plt.GridSpec(4, 4,hspace=0.30,wspace=0.30)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Scattering Angle [rad]",labelpad=30)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=40)
    
    rows={0:"1500",1:"5000",2:"15000",3:"50000"}
    columns={0:0,1:1,2:2,3:3}
    
    cuts=np.array([[0,1*10**9,40,0.025],
                  [0,1*10**9,200,0.025],
                  [0,1*10**9,1000,0.025],
                  [0,1*10**9,20000,0.025]])
    
    letters=np.array([["a","b","c","d"],
                    ["e","f","g","h"],
                    ["i","j","k","l"],
                    ["m","n","o","p"]])
    
    for row in range(len(rows)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            s=30
            if column==0:
                d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
                d=rd.cut_selection(d,9,0,True)
                s=100
                if row==0:
                    temp.set_xlabel("No Cuts")
                    temp.xaxis.set_label_position("top")
            elif column==1:
                d=mo.remove_vel_out(True,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
                if row==0:
                    temp.set_xlabel("Velocity Cut")
                    temp.xaxis.set_label_position("top")
            elif column==2:
                d=mo.remove_e_out(True,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5",int(rows[row]))
                if row==0:
                    temp.set_xlabel("Energy Cut")
                    temp.xaxis.set_label_position("top")
            elif column==3:
                d=mo.remove_sa_out(True,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
                if row==0:
                    temp.set_xlabel("Scattering Angle Cut")
                    temp.xaxis.set_label_position("top")
                    temp.set_ylabel("1.5 MeV")
                    temp.yaxis.set_label_position("right")
                elif row==1:
                    temp.set_ylabel("5 MeV")
                    temp.yaxis.set_label_position("right")
                elif row==2:
                    temp.set_ylabel("15 MeV")
                    temp.yaxis.set_label_position("right")
                elif row==3:
                    temp.set_ylabel("50 MeV")
                    temp.yaxis.set_label_position("right")
                
            mo.plot_ang_dist_mono(True,d,int(rows[row]),s)
            temp.text(0.01,0.95,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='center', fontsize="small")
            
    plt.savefig(rd.p2+'SAD_mono_f.pdf',bbox_inches='tight')


def plot_scattering_angle_for_monochromatic_sources_backward():
    fig=plt.figure(figsize=(12,9))
    grid=plt.GridSpec(2, 2,hspace=0.20,wspace=0.20)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Scattering Angle [rad]",labelpad=30)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=40)

    p1=fig.add_subplot(grid[0,0])
    d=rd.read_h5py_file("Mono_1500/","Mono_1500_h5")
    d=rd.cut_selection(d,9,0,False)
    mo.plot_ang_dist_mono(False,d,1500,30)
    p1.set_xlabel("1.5MeV")
    p1.xaxis.set_label_position("top")
    p1.text(0.05,0.95,"(a)",transform=p1.transAxes,ha='center', va='center')
    
    p2=fig.add_subplot(grid[0,1])
    d=rd.read_h5py_file("Mono_5000/","Mono_5000_h5")
    d=rd.cut_selection(d,9,0,False)
    mo.plot_ang_dist_mono(False,d,5000,30)
    p2.set_xlabel("5MeV")
    p2.xaxis.set_label_position("top")
    p2.text(0.05,0.95,"(b)",transform=p2.transAxes,ha='center', va='center')
    
    p3=fig.add_subplot(grid[1,0])
    d=rd.read_h5py_file("Mono_15000/","Mono_15000_h5")
    d=rd.cut_selection(d,9,0,False)
    mo.plot_ang_dist_mono(False,d,15000,30)
    p3.set_xlabel("15MeV")
    p3.xaxis.set_label_position("top")
    p3.text(0.05,0.95,"(c)",transform=p3.transAxes,ha='center', va='center')
    
    p4=fig.add_subplot(grid[1,1])
    d=rd.read_h5py_file("Mono_50000/","Mono_50000_h5")
    d=rd.cut_selection(d,9,0,False)
    mo.plot_ang_dist_mono(False,d,50000,30)
    p4.set_xlabel("50MeV")
    p4.xaxis.set_label_position("top")
    p4.text(0.05,0.95,"(d)",transform=p4.transAxes,ha='center', va='center')
    
    plt.savefig(rd.p2+'SAD_mono_b.pdf',bbox_inches='tight')


def plot_scattering_angle_via_energies_monochromatic():
    fig=plt.figure(figsize=(18,8))
    grid=plt.GridSpec(2, 4,hspace=0.10,wspace=0.15)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Scattering Angle via Energies [rad]",labelpad=25)
    pa.set_ylabel("Scattering Angle via Positions [rad]",labelpad=30)
    
    columns={0:"1500",1:"5000",2:"15000",3:"50000"}
    energies={0:"1.5",1:"5",2:"15",3:"50"}
    rows={0:True,1:False}
    
    letters=np.array([["a","b","c","d"],
                    ["e","f","g","h"]])
    
    for row in range(len(rows)):
        for column in range(len(columns)):
            d=rd.read_h5py_file("Mono_"+columns[column]+"/","Mono_"+columns[column]+"_h5")
            d=rd.cut_selection(d,9,0,rows[row])
            temp=fig.add_subplot(grid[row,column])
            
            ed.SAD_E_vs_Pos(rows[row],100,d)
            
            temp.text(0.00,1,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top',bbox=dict(facecolor='white', edgecolor='none'))

            if row==0:
                plt.xlim(None,0.8)

            if row==0:
                temp.set_xlabel(energies[column]+"MeV")
                temp.xaxis.set_label_position("top")

                    
            if column==3:
                if row==0:
                    temp.set_ylabel("Forward Photons")
                    temp.yaxis.set_label_position("right")
                elif row==1:
                    temp.set_ylabel("Backward Photons")
                    temp.yaxis.set_label_position("right")

    plt.savefig(rd.p2+'SAD_E_mono.pdf',bbox_inches='tight')


def plot_time_of_flight():
    fig=plt.figure(figsize=(12,10))
    grid=plt.GridSpec(2, 2,hspace=0.15,wspace=0.15)
    
    
    E_dist=np.geomspace(0.7,50,100)
    s1f=0
    s2=0
    for i in range(len(E_dist)):
        temp1f=E_dist[i]*at.energy_distribution(E_dist[i]*1000)*mo.calculate_total_distance(E_dist[i],True)
        temp2=E_dist[i]*at.energy_distribution(E_dist[i]*1000)
        if not np.isnan(temp1f):
            s1f+=temp1f
            s2+=temp2
    av_df=s1f/s2
    
    
    s1b=0
    s2=0
    for i in range(len(E_dist)):
        temp1b=E_dist[i]*at.energy_distribution(E_dist[i]*1000)*mo.calculate_total_distance(E_dist[i],False)
        temp2=E_dist[i]*at.energy_distribution(E_dist[i]*1000)
        if not np.isnan(temp1b):
            s1b+=temp1b
            s2+=temp2
    av_db=s1b/s2
    
    c=2.998e10
    sig_t=3.53553e-10
    sig_D1=rd.D1_depth/2/np.sqrt(3)
    sig_D2=rd.D2_depth/2/np.sqrt(3)
    
    sig=np.sqrt(sig_t**2+(sig_D1/c)**2+(sig_D2/c)**2)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Time of Flight [s]",labelpad=40)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=40)
    
    d=rd.read_h5py_file()
    d=rd.slice_selection(d,17,3.5e-9,9e-9)
    df=rd.cut_selection(d, 9, 0)
    tf=rd.extract_parameter(df, 17)
    db=rd.cut_selection(d, 9, 0, False)
    tb=rd.extract_parameter(db, 17)
    
    
    
    
    p1=fig.add_subplot(grid[0,0])
    n,bins,patches=plt.hist(tf,100)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    f_dist=at.forward_ToF(x_dist,sig,av_df,1e-13)
    a=minimize(rd.fit_func,np.amax(n)/np.amax(f_dist)*0.5,(n,f_dist)).x
    p1.set_xlabel("Forward Photons")
    p1.xaxis.set_label_position("top")
    p1.text(0.05,0.95,"(a)",transform=p1.transAxes,ha='center', va='center')
    plt.plot(x_dist,f_dist*a,lw=2)
    
    disf='{:.1f}'.format(round(av_df, 1))
    sigma='{:.3f}'.format(round(sig*1e10, 3))
    p1.text(0.80,0.90,"d="+disf+"e10cm\n $\sigma$="+sigma+"e-10s",transform=p1.transAxes,ha='center', va='center')
    
    
    p2=fig.add_subplot(grid[0,1])
    n,bins,patches=plt.hist(tb,50)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    b_dist=at.backward_ToF(x_dist,sig,av_df,1e-13)
    a=minimize(rd.fit_func,np.amax(n)/np.amax(b_dist)*0.5,(n,b_dist)).x
    p2.set_xlabel("Backward Photons")
    p2.xaxis.set_label_position("top")
    p2.text(0.05,0.95,"(b)",transform=p2.transAxes,ha='center', va='center')
    plt.plot(x_dist,b_dist*a,lw=2)
    
    disb='{:.1f}'.format(round(av_db, 1))
    sigma='{:.3f}'.format(round(sig*1e10, 3))
    p2.text(0.80,0.90,"d="+disb+"e10cm\n $\sigma$="+sigma+"e-10s",transform=p2.transAxes,ha='center', va='center')
    
    p3=fig.add_subplot(grid[1,0])
    popt,pcov=at.optimize_ToF_parameters(True,100)
    sig='{:.3f}'.format(round(popt[0]*1e10, 3))
    sig_u='{:.3f}'.format(round(pcov[0,0]**0.5*1e10, 3))
    d='{:.1f}'.format(round(popt[1], 1))
    d_u='{:.1f}'.format(round(pcov[1,1]**0.5, 1))
    p3.text(0.80,0.90,"d=("+d+"$\pm$"+d_u+")cm\n$\sigma$=("+sig+"$\pm$"+sig_u+")e-10s",transform=p3.transAxes,ha='center', va='center')
    p3.text(0.05,0.95,"(c)",transform=p3.transAxes,ha='center', va='center')
    
    p4=fig.add_subplot(grid[1,1])
    popt,pcov=at.optimize_ToF_parameters(False,50)
    sig='{:.3f}'.format(round(popt[0]*1e10, 3))
    sig_u='{:.3f}'.format(round(pcov[0,0]**0.5*1e10, 3))
    d='{:.1f}'.format(round(popt[1], 1))
    d_u='{:.1f}'.format(round(pcov[1,1]**0.5, 1))
    p4.text(0.80,0.90,"d=("+d+"$\pm$"+d_u+")cm\n$\sigma$=("+sig+"$\pm$"+sig_u+")e-10s" ,transform=p4.transAxes,ha='center', va='center')
    p4.text(0.05,0.95,"(d)",transform=p4.transAxes,ha='center', va='center')
    
    
    plt.savefig(rd.p2+'ToF.pdf',bbox_inches='tight')


def plot_time_of_flight_monochromotic_forward():
    fig=plt.figure(figsize=(15,15))
    grid=plt.GridSpec(4, 4,hspace=0.30,wspace=0.30)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Time of Flight [s]",labelpad=40)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)

    rows={0:"1500",1:"5000",2:"15000",3:"50000"}
    columns={0:0,1:1,2:2,3:3}
    
    cuts=np.array([[0,1*10**9,40,0.025],
                  [0,1*10**9,200,0.025],
                  [0,1*10**9,1000,0.025],
                  [0,1*10**9,20000,0.025]])
    xlims=np.array([[(3.5e-9,8.0e-9),(4.5e-9,7.0e-9),(3.5e-9,7.5e-9),(3.5e-9,7.5e-9)],
                    [(3.5e-9,8.0e-9),(4.5e-9,7.0e-9),(3.5e-9,7.5e-9),(3.5e-9,7.5e-9)],
                    [(2.0e-9,8.0e-9),(4.5e-9,7.0e-9),(3.5e-9,7.5e-9),(3.5e-9,7.5e-9)],
                    [(1.0e-9,8.0e-9),(4.5e-9,7.0e-9),(3.5e-9,7.5e-9),(2.5e-9,7.5e-9)]])
    
    letters=np.array([["a","b","c","d"],
                    ["e","f","g","h"],
                    ["i","j","k","l"],
                    ["m","n","o","p"]])
    
    
    for row in range(len(rows)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            s=40
            d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
            d=rd.cut_selection(d,9,0,True)
            d=rd.slice_selection(d,17,1.0e-9,9e-9)
            if column==0:
                
                s=100
                if row==0:
                    temp.set_xlabel("No Cuts")
                    temp.xaxis.set_label_position("top")
            elif column==1:
                d=mo.remove_vel_out(True,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5",d)
                if row==0:
                    temp.set_xlabel("Velocity Cut")
                    temp.xaxis.set_label_position("top")
            elif column==2:
                d=mo.remove_e_out(True,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5",int(rows[row]),d)
                if row==0:
                    temp.set_xlabel("Energy Cut")
                    temp.xaxis.set_label_position("top")
            elif column==3:
                d=mo.remove_sa_out(True,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5",d)
                if row==0:
                    temp.set_xlabel("Scattering Angle Cut")
                    temp.xaxis.set_label_position("top")
                    temp.set_ylabel("1.5MeV")
                    temp.yaxis.set_label_position("right")
                elif row==1:
                    temp.set_ylabel("5MeV")
                    temp.yaxis.set_label_position("right")
                elif row==2:
                    temp.set_ylabel("15MeV")
                    temp.yaxis.set_label_position("right")
                elif row==3:
                    temp.set_ylabel("50MeV")
                    temp.yaxis.set_label_position("right")
            plt.xlim(xlims[row,column][0],xlims[row,column][1])
            
            vel_cut=True if column==1 else False
            
            av_d,sig=mo.plot_ToF_mono(True,d,s,int(rows[row]),vel_cut,xlims[row,column])
            
            disb='{:.1f}'.format(round(av_d, 1))
            sigma='{:.3f}'.format(round(sig*1e10, 3))
            temp.text(0.90,0.5,"d="+disb+"e10cm\n $\sigma$="+sigma+"e-10s",transform=temp.transAxes,ha='center', va='center',rotation="vertical")

            temp.text(0.01,0.95,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='center')

    plt.savefig(rd.p2+'ToF_mono_f.pdf',bbox_inches='tight')


def plot_time_of_flight_monochromotic_backward():
    fig=plt.figure(figsize=(15,15))
    grid=plt.GridSpec(4, 4,hspace=0.30,wspace=0.30)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Time of Flight [s]",labelpad=40)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)

    rows={0:"1500",1:"5000",2:"15000",3:"50000"}
    columns={0:0,1:1,2:2,3:3}
    
    cuts=np.array([[0,1*10**9,40,0.5],
                  [0,1*10**9,200,0.5],
                  [0,1*10**9,1000,0.5],
                  [0,1*10**9,5000,0.5]])
    xlims=np.array([[(3.5e-9,9.5e-9),(4.5e-9,7.0e-9),(3.5e-9,9.5e-9),(3.5e-9,9.5e-9)],
                    [(3.5e-9,9.5e-9),(4.5e-9,7.0e-9),(3.5e-9,9.5e-9),(3.5e-9,9.5e-9)],
                    [(3.5e-9,9.5e-9),(4.5e-9,7.0e-9),(3.5e-9,9.5e-9),(3.5e-9,9.5e-9)],
                    [(3.5e-9,9.5e-9),(4.5e-9,7.0e-9),(3.5e-9,9.5e-9),(3.5e-9,9.5e-9)]])
    
    letters=np.array([["a","b","c","d"],
                    ["e","f","g","h"],
                    ["i","j","k","l"],
                    ["m","n","o","p"]])
    
    for row in range(len(rows)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            s=40
            d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
            d=rd.cut_selection(d,9,0,False)
            d=rd.slice_selection(d,17,1.0e-9,10e-9)
            if column==0:
                
                s=100
                if row==0:
                    temp.set_xlabel("No Cuts")
                    temp.xaxis.set_label_position("top")
            elif column==1:
                d=mo.remove_vel_out(False,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5",d)
                if row==0:
                    temp.set_xlabel("Velocity Cut")
                    temp.xaxis.set_label_position("top")
            elif column==2:
                d=mo.remove_e_out(False,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5",int(rows[row]),d)
                if row==0:
                    temp.set_xlabel("Energy Cut")
                    temp.xaxis.set_label_position("top")
            elif column==3:
                d=mo.remove_sa_out(False,cuts[row,column],"Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5",d)
                if row==0:
                    temp.set_xlabel("Scattering Angle Cut")
                    temp.xaxis.set_label_position("top")
                    temp.set_ylabel("1.5MeV")
                    temp.yaxis.set_label_position("right")
                elif row==1:
                    temp.set_ylabel("5MeV")
                    temp.yaxis.set_label_position("right")
                elif row==2:
                    temp.set_ylabel("15MeV")
                    temp.yaxis.set_label_position("right")
                elif row==3:
                    temp.set_ylabel("50MeV")
                    temp.yaxis.set_label_position("right")
            plt.xlim(xlims[row,column][0],xlims[row,column][1])
            vel_cut=True if column==1 else False
            
            av_d,sig=mo.plot_ToF_mono(False,d,s,int(rows[row]),vel_cut,xlims[row,column])
            
            disb='{:.1f}'.format(round(av_d, 1))
            sigma='{:.3f}'.format(round(sig*1e10, 3))
            temp.text(0.90,0.5,"d="+disb+"e10cm\n $\sigma$="+sigma+"e-10s",transform=temp.transAxes,ha='center', va='center',rotation="vertical")

            temp.text(0.01,0.95,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='center')

    plt.savefig(rd.p2+'ToF_mono_b.pdf',bbox_inches='tight')


def plot_energy_distribution():
    fig=plt.figure(figsize=(12,5))
    grid=plt.GridSpec(1, 2,hspace=0.20,wspace=0.15)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Total Detected Energy [MeV]",labelpad=30)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=40)
    
    d=rd.read_h5py_file()
    
    
    df=rd.cut_selection(d, 9, 0, True)
    Ef=(rd.extract_parameter(df, 1) + rd.extract_parameter(df, 3))/1000
    logbinsf = np.geomspace(Ef.min(), Ef.max(), 100)
    p1=fig.add_subplot(grid[0,0])
    plt.xscale("log")
    p1.set_xlabel("Forward Photons")
    p1.xaxis.set_label_position("top")
    p1.text(0.05,0.95,"(a)",transform=p1.transAxes,ha='center', va='center')
    y2f=np.array([quad(ed.integrand1,at.fsad_min,at.fsad_max,(i,True,at.fsad_min,at.fsad_max,at.fsad),0,10e-5,10e-5)[0]*i for i in logbinsf*1000])
    n,bins,patches=plt.hist(Ef, bins=logbinsf)
    yf=np.zeros(len(y2f))
    for i in range(len(y2f)):
        if i==0:
            yf[i]=n[i]
        elif i==len(y2f)-1:
            yf[i]=n[i-1]
        else:
            yf[i]=(n[i-1]+n[i])/2
    a=minimize(rd.fit_func,5e2,(yf,y2f)).x
    plt.plot(logbinsf,y2f*a,lw=2.0)
    
    db=rd.cut_selection(d, 9, 0, False)
    Eb=(rd.extract_parameter(db, 1) + rd.extract_parameter(db, 3))/1000
    p2=fig.add_subplot(grid[0,1])
    logbinsb = np.geomspace(Eb.min(), Eb.max(), 50)
    plt.xscale("log")
    p2.set_xlabel("Backward Photons")
    p2.xaxis.set_label_position("top")
    p2.text(0.05,0.95,"(b)",transform=p2.transAxes,ha='center', va='center')
    y2b=np.array([quad(ed.integrand2,at.bsad_min,at.bsad_max,(i,False,at.bsad_min,at.bsad_max,at.bsad),0,10e-5,10e-5)[0]*i for i in logbinsb*1000])##########
    n,bins,patches=plt.hist(Eb, bins=logbinsb)
    yb=np.zeros(len(y2b))
    for i in range(len(y2b)):
        if i==0:
            yb[i]=n[i]
        elif i==len(y2b)-1:
            yb[i]=n[i-1]
        else:
            yb[i]=(n[i-1]+n[i])/2
    a=minimize(rd.fit_func,5e2,(yb,y2b)).x
    #a=1e5
    plt.plot(logbinsb,y2b*a,lw=2.0)
    
    plt.savefig(rd.p2+'E.pdf',bbox_inches='tight')


def plot_energy_distribution_per_detector_layer():
    fig=plt.figure(figsize=(15,10))
    grid=plt.GridSpec(2, 3,hspace=0.20,wspace=0.20)
    
    pa=fig.add_subplot(grid[:,:2])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Detected Energy [MeV]",labelpad=30)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=40)
    
    d=rd.read_h5py_file()
    
    df=rd.cut_selection(d, 9, 0, True)
    
    E=rd.extract_parameter(df, 3)/1000
    logbins = np.geomspace(E.min(), E.max(), 100)
    p1=fig.add_subplot(grid[0,0])
    plt.xscale("log")
    p1.set_xlabel("Forward: E1")
    p1.xaxis.set_label_position("top")
    p1.text(0.01,0.99,"(a)",transform=p1.transAxes,ha='left', va='top')
    n,bins,patches=plt.hist(E, bins=logbins)
    
    E=rd.extract_parameter(df, 1)/1000
    logbins = np.geomspace(E.min(), E.max(), 100)
    p2=fig.add_subplot(grid[0,1])
    plt.xscale("log")
    p2.set_xlabel("Forward: E2")
    p2.xaxis.set_label_position("top")
    
    p2.text(0.01,0.99,"(b)",transform=p2.transAxes,ha='left', va='top')
    n,bins,patches=plt.hist(E, bins=logbins)
    
    db=rd.cut_selection(d, 9, 0, False)

    E=rd.extract_parameter(db, 1)/1000
    logbins = np.geomspace(E.min(), E.max(), 50)
    p3=fig.add_subplot(grid[1,0])
    plt.xscale("log")
    p3.set_xlabel("Backward: E1")
    p3.xaxis.set_label_position("top")
    p3.text(0.01,0.99,"(d)",transform=p3.transAxes,ha='left', va='top')
    n,bins,patches=plt.hist(E, bins=logbins)
    
    E=rd.extract_parameter(db, 3)/1000
    logbins = np.geomspace(E.min(), E.max(), 50)
    p4=fig.add_subplot(grid[1,1])
    plt.xscale("log")
    p4.set_xlabel("Backward: E2")
    p4.xaxis.set_label_position("top")
    plt.ylim(None,1150)
    
    p4.text(0.01,0.99,"(e)",transform=p4.transAxes,ha='left', va='top')
    n,bins,patches=plt.hist(E, bins=logbins)
    
    E1f=rd.extract_parameter(df, 3)/1000
    E2f=rd.extract_parameter(df, 1)/1000
    E1b=rd.extract_parameter(db, 1)/1000
    E2b=rd.extract_parameter(db, 3)/1000
    
    num_bins=100
    
    logbins1f = np.geomspace(E1f.min(), E1f.max(), num_bins)
    logbins2f = np.geomspace(E2f.min(), E2f.max(), num_bins)
    
    logbins1b = np.geomspace(E1b.min(), E1b.max(), num_bins)
    logbins2b = np.geomspace(E2b.min(), E2b.max(), num_bins)
    
    p5=fig.add_subplot(grid[0,2])
    counts, _, _ = np.histogram2d(E1f, E2f, bins=(logbins1f, logbins2f))

    p5.pcolormesh(logbins1f, logbins2f, counts.T)
    p5.plot()
    p5.set_xscale('log')
    p5.set_yscale('log')
    
    p5.set_xlabel("E1 [MeV]")
    p5.set_ylabel("E2 [MeV]",labelpad=-10)
    p7=plt.twinx(p5)
    p7.set_ylabel("Forward Photons",labelpad=10)
    p7.yaxis.set_label_position("right")
    p7.axes.get_yaxis().set_ticks([])

    
    p5.text(0.01,0.99,"(c)",transform=p5.transAxes,ha='left', va='top',bbox=dict(facecolor='white', edgecolor='none'))
    
    p6=fig.add_subplot(grid[1,2])
    counts, _, _ = np.histogram2d(E1b, E2b, bins=(logbins1b, logbins2b))

    p6.pcolormesh(logbins1b, logbins2b, counts.T)
    p6.plot()
    p6.set_xscale('log')
    p6.set_yscale('log')
    
    p6.set_xlabel("E1 [MeV]")
    p6.set_ylabel("E2 [MeV]",labelpad=-10)
    p8=plt.twinx(p6)
    p8.set_ylabel("Backward Photons",labelpad=10)
    p8.yaxis.set_label_position("right")
    p8.axes.get_yaxis().set_ticks([])
    
    p6.text(0.01,0.99,"(f)",transform=p6.transAxes,ha='left', va='top',bbox=dict(facecolor='white', edgecolor='none'))

    plt.savefig(rd.p2+'E_layer.pdf',bbox_inches='tight')


def plot_energy_distribution_monochromatic():
    fig=plt.figure(figsize=(10,12))
    grid=plt.GridSpec(4, 2,hspace=0.15,wspace=0.20)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Total Detected Energy [MeV]",labelpad=30)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)

    rows={0:"1500",1:"5000",2:"15000",3:"50000"}
    energies={0:"1.5",1:"5",2:"15",3:"50"}
    columns={0:True,1:False}
    
    letters=np.array([["a","b"],
                    ["c","d"],
                    ["e","f"],
                    ["g","h"]])

    for row in range(len(rows)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
            d=rd.cut_selection(d,9,0,columns[column])
            E=(rd.extract_parameter(d, 1) + rd.extract_parameter(d, 3))/1000
            logbins = np.geomspace(E.min(), E.max(), 100)
            plt.xscale("log")
            temp.text(0.05,0.95,"("+letters[row,column]+")",transform=temp.transAxes,ha='center', va='center')
            n,bins,patches=plt.hist(E, bins=logbins)
            plt.axvline(float(energies[row]),color="C1",lw=2.0)
            if row==0:
                if column==0:
                    temp.set_xlabel("Forward Photons")
                    temp.xaxis.set_label_position("top")
                else:
                    temp.set_xlabel("Backward Photons")
                    temp.xaxis.set_label_position("top")
            if column==1:
                temp.set_ylabel(energies[row]+"MeV")
                temp.yaxis.set_label_position("right")
            
    plt.savefig(rd.p2+'E_mono.pdf',bbox_inches='tight')


def plot_energy_distribution_monochromatic_layer():
    fig=plt.figure(figsize=(12,12))
    grid=plt.GridSpec(4, 4,hspace=0.15,wspace=0.30)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Detected Energy [MeV]",labelpad=30)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=40)

    rows={0:"1500",1:"5000",2:"15000",3:"50000"}
    energies={0:"1.5",1:"5",2:"15",3:"50"}
    columns={0:(True,3),1:(True,1),2:(False,1),3:(False,3)}
    
    num_bins=100
    
    letters=np.array([["a","b","c","d"],
                    ["e","f","g","h"],
                    ["i","j","k","l"],
                    ["m","n","o","p"]])

    intensities=np.zeros(num_bins-1)
    
    x_dists=np.zeros(num_bins-1)

    amps=np.array([[1,1.45,1,1],
                    [1.05,1.7,1,1],
                    [1.2,1.5,1,1],
                    [1,1,1,1]])

    for row in range(len(rows)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
            d=rd.cut_selection(d,9,0,columns[column][0])
            E=rd.extract_parameter(d, columns[column][1])/1000
            logbins = np.geomspace(E.min(), E.max(), num_bins)
            plt.xscale("log")
            temp.text(0.01,0.95,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='center')
            n,bins,patches=plt.hist(E, bins=logbins)
            
            
            for i in range(len(x_dists)):
                x_dists[i]=np.e**( (np.log(logbins[i])+np.log(logbins[i+1]))/2 )
            
            intensities=n
            
            if column==0 or column==2:
                e_min=0.05
            else:
                e_min=0.5
            e_dist=np.geomspace(e_min,float(energies[row]),5000)
            
            exp_dist=mo.energy_layer_dist(columns[column][0],columns[column][1],int(rows[row]),e_dist*1000)*e_dist
            
            a=1*np.amax(intensities)/np.amax(exp_dist)
            plt.plot(e_dist,exp_dist*a*amps[row,column],lw=2.0)
            
            
            wl1=6.625e-34*3e8/(1.602e-19*float(energies[row])*1e6)
            wl2=wl1+2*2.426e-12
            eline1=6.625e-34*3e8/(1.602e-19*wl2)*1e-6

            if column==0:
                if row==2 or row==3:
                    plt.axvline(eline1,color="C2",lw=2.0)

                
            if row==0:
                if column==0:
                    temp.set_xlabel("Forward Photons: E1")
                    temp.xaxis.set_label_position("top")
                elif column==1:
                    temp.set_xlabel("Forward Photons: E2")
                    temp.xaxis.set_label_position("top")
                elif column==2:
                    temp.set_xlabel("Backward Photons: E1")
                    temp.xaxis.set_label_position("top")
                else:
                    temp.set_xlabel("Backward Photons: E2")
                    temp.xaxis.set_label_position("top")
            if column==3:
                temp.set_ylabel(energies[row]+"MeV")
                temp.yaxis.set_label_position("right")
            
            if n[0]/np.amax(n)>0.9 and not row==2:
                plt.ylim(None,np.amax(n)*1.1)
                
            if row>1 and column==3:
                plt.xlim(1e0)

            
    plt.savefig(rd.p2+'E_mono_layer.pdf',bbox_inches='tight')
        

def plot_energy_distribution_monochromatic_2D_histogram():
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
            d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
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
            #plt.axvline(0.511)
            
            wl1=6.625e-34*3e8/(1.602e-19*float(energies[row])*1e6)
            wl2=wl1+2*2.426e-12
            eline1=6.625e-34*3e8/(1.602e-19*wl2)*1e-6
            
            if column==1 or row==3:
                plt.axvline(eline1,color="C6")
            
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
                
            
            temp.text(-0.10,0.97,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='center')
                
        plt.savefig(rd.p2+'E_mono_2D_hist.pdf',bbox_inches='tight')


def plot_positional_distribution():
    fig=plt.figure(figsize=(18,8.2))
    grid=plt.GridSpec(2, 4,hspace=0.10,wspace=0.15)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("X Coordinate [cm]",labelpad=30)
    pa.set_ylabel("Y Coordinate [cm]",labelpad=35)
    
    
    letters=np.array([["a","b","c","d"],
                    ["e","f","g","h"]])
    
    for row in range(2):
        for column in range(4):
            temp=fig.add_subplot(grid[row,column])
            text=True if row==1 else False
            forward=True if (column==0 or column==1) else False
            D1=True if (column==0 or column==2) else False
            pd.events_per_cell(forward,D1,text)
            
            temp.text(0.00,1,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top',bbox=dict(facecolor='white', edgecolor='none'))
            if row==0:
                if column==0:
                    temp.set_xlabel("Forward Photons: D1")
                    temp.xaxis.set_label_position("top")
                elif column==1:
                    temp.set_xlabel("Forward Photons: D2")
                    temp.xaxis.set_label_position("top")
                elif column==2:
                    temp.set_xlabel("Backward Photons: D1")
                    temp.xaxis.set_label_position("top")
                elif column==3:
                    temp.set_xlabel("Backward Photons: D2")
                    temp.xaxis.set_label_position("top")
                
    plt.savefig(rd.p2+'Pos.pdf',bbox_inches='tight')


def plot_positional_distribution_monochromatic():
    fig=plt.figure(figsize=(18,16.5))
    grid=plt.GridSpec(4, 4,hspace=0.10,wspace=0.15)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("X Coordinate [cm]",labelpad=30)
    pa.set_ylabel("Y Coordinate [cm]",labelpad=35)
    
    rows={0:"1500",1:"5000",2:"15000",3:"50000"}
    energies={0:"1.5",1:"5",2:"15",3:"50"}
    columns={0:(True,True),1:(True,False),2:(False,True),3:(False,False)}
    
    letters=np.array([["a","b","c","d"],
                    ["e","f","g","h"],
                    ["i","j","k","l"],
                    ["m","n","o","p"]])
    
    for row in range(len(rows)):
        for column in range(len(columns)):
            d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
            d=rd.cut_selection(d,9,0,columns[column][0])
            temp=fig.add_subplot(grid[row,column])
            mo.events_per_cell_mono(columns[column][0],columns[column][1],d,int(rows[row]))
            
            temp.text(0.00,1,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top',bbox=dict(facecolor='white', edgecolor='none'))

            if row==0:
                if column==0:
                    temp.set_xlabel("Forward Photons: D1")
                    temp.xaxis.set_label_position("top")
                elif column==1:
                    temp.set_xlabel("Forward Photons: D2")
                    temp.xaxis.set_label_position("top")
                elif column==2:
                    temp.set_xlabel("Backward Photons: D1")
                    temp.xaxis.set_label_position("top")
                elif column==3:
                    temp.set_xlabel("Backward Photons: D2")
                    temp.xaxis.set_label_position("top")
                    
            if column==3:
                temp.set_ylabel(energies[row]+"MeV")
                temp.yaxis.set_label_position("right")

    plt.savefig(rd.p2+'Pos_mono.pdf',bbox_inches='tight')


def plot_positional_distribution_50_MeV():
    fig=plt.figure(figsize=(18,3.9))
    grid=plt.GridSpec(1, 4,hspace=0.10,wspace=0.15)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("X Coordinate [cm]",labelpad=30)
    pa.set_ylabel("Y Coordinate [cm]",labelpad=35)
    
    rows={0:"50000"}
    energies={0:"50"}
    columns={0:(True,True),1:(True,False),2:(False,True),3:(False,False)}
    letters=np.array([["a","b","c","d"]])
    
    for row in range(1):
        for column in range(4):
            d=rd.read_h5py_file("Mono_"+rows[row]+"/","Mono_"+rows[row]+"_h5")
            d=rd.cut_selection(d,9,0,columns[column][0])
            temp=fig.add_subplot(grid[row,column])

            mo.events_per_cell_mono(columns[column][0],columns[column][1],d,int(rows[row]),False)
            
            temp.text(0.00,1,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top',bbox=dict(facecolor='white', edgecolor='none'))
            if row==0:
                if column==0:
                    temp.set_xlabel("Forward Photons: D1")
                    temp.xaxis.set_label_position("top")
                elif column==1:
                    temp.set_xlabel("Forward Photons: D2")
                    temp.xaxis.set_label_position("top")
                elif column==2:
                    temp.set_xlabel("Backward Photons: D1")
                    temp.xaxis.set_label_position("top")
                elif column==3:
                    temp.set_xlabel("Backward Photons: D2")
                    temp.xaxis.set_label_position("top")
            
            if column==3:
                temp.set_ylabel(energies[row]+"MeV")
                temp.yaxis.set_label_position("right")
                
    plt.savefig(rd.p2+'Pos_50.pdf',bbox_inches='tight')


def plot_attenuation_function():
    fig=plt.figure(figsize=(12,5))
    grid=plt.GridSpec(1, 2,hspace=0.20,wspace=0.15)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Photon Energy [MeV]",labelpad=30)
    pa.set_ylabel("Mass Attenuation Coefficient [cm$^2$/g]",labelpad=35)
    
    p1=fig.add_subplot(grid[0,0])
    af.plot_all()
    
    p2=fig.add_subplot(grid[0,1])
    af.plot_mass_att_coe()

    plt.savefig(rd.p2+'att_coe.pdf',bbox_inches='tight')


def plot_interaction_probabilty():
    fig=plt.figure(figsize=(12,10))
    grid=plt.GridSpec(2, 2,hspace=0.10,wspace=0.20)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("Photon Energy [MeV]",labelpad=30)
    pa.set_ylabel("Average Travel Distance in each Detector Layer [cm]",labelpad=30)

    p1=fig.add_subplot(grid[0,1])
    af.plot_abs_prob()
    p1.text(0.01,0.99,"(b)",transform=p1.transAxes,ha='left', va='top')
    plt.ylim(None,1.06)
    
    p2=fig.add_subplot(grid[0,0])
    af.plot_av_dis()
    E_dist=np.geomspace(0.5,20,100)
    p2.text(0.01,0.99,"(a)",transform=p2.transAxes,ha='left', va='top')

    plt.plot(E_dist,mo.FD2,label="D2: Forward")
    plt.plot(E_dist,mo.BD1,label="D1: Backward")
    

    plt.legend()
    
    
    p3=fig.add_subplot(grid[1,0])
    E_dist=np.geomspace(0.05,20,100)
    d1inf=np.array([mo.calculate_travel_distance_inf(True,i*1000) for i in E_dist])
    d2inf=np.array([mo.calculate_travel_distance_inf(False,i*1000) for i in E_dist])
    plt.plot(E_dist,d1inf,label="D1: Infinite")
    plt.plot(E_dist,d2inf,label="D2: Infinite")
    plt.xscale("log")
    plt.yscale("log")
    p3.text(0.01,0.99,"(c)",transform=p3.transAxes,ha='left', va='top')
    
    plt.legend(loc="lower right")

    p4=fig.add_subplot(grid[1,1])
    totdf=np.array([mo.calculate_total_distance(i,True) for i in E_dist])
    totdb=np.array([mo.calculate_total_distance(i,False) for i in E_dist])
    plt.plot(E_dist,totdf,label="Forward")
    plt.plot(E_dist,totdb,label="Backward")
    plt.xscale("log")
    plt.legend()
    plt.ylabel("Average Total Travel Distance [cm]")
    p4.text(0.01,0.99,"(d)",transform=p4.transAxes,ha='left', va='top')
    
    plt.savefig(rd.p2+'tra_dis.pdf',bbox_inches='tight')


def plot_ARM():
    fig=plt.figure(figsize=(15,8))
    grid=plt.GridSpec(2, 3,hspace=0.20,wspace=0.20)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_xlabel("ARM [rad]",labelpad=30)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)
    
    p1=fig.add_subplot(grid[0,0])
    d=rd.read_h5py_file(rd.p3+"Mono_1500/","Mono_1500_h5")
    d=d=rd.cut_selection(d,9,0,True)
    pc.create_ARM_plot(d)
    p1.set_xlabel("1.5MeV: Forward")
    p1.xaxis.set_label_position("top")
    p1.text(0.01,0.99,"(a)",transform=p1.transAxes,ha='left', va='top')
    
    p2=fig.add_subplot(grid[0,1])
    d=rd.read_h5py_file(rd.p3+"Mono_5000/","Mono_5000_h5")
    d=d=rd.cut_selection(d,9,0,True)
    pc.create_ARM_plot(d)
    p2.set_xlabel("5MeV: Forward")
    p2.xaxis.set_label_position("top")
    p2.text(0.01,0.99,"(b)",transform=p2.transAxes,ha='left', va='top')
    
    p3=fig.add_subplot(grid[1,0])
    d=rd.read_h5py_file(rd.p3+"Mono_15000/","Mono_15000_h5")
    d=d=rd.cut_selection(d,9,0,True)
    pc.create_ARM_plot(d)
    p3.set_xlabel("15MeV: Forward")
    p3.xaxis.set_label_position("top")
    p3.text(0.01,0.99,"(d)",transform=p3.transAxes,ha='left', va='top')
    
    p4=fig.add_subplot(grid[1,1])
    d=rd.read_h5py_file(rd.p3+"Mono_50000/","Mono_50000_h5")
    d=d=rd.cut_selection(d,9,0,True)
    pc.create_ARM_plot(d)
    p4.set_xlabel("50MeV: Forward")
    p4.xaxis.set_label_position("top")
    p4.text(0.01,0.99,"(e)",transform=p4.transAxes,ha='left', va='top')
    
    p5=fig.add_subplot(grid[0,2])
    d=rd.read_h5py_file()
    d=d=rd.cut_selection(d,9,0,True)
    pc.create_ARM_plot(d)
    p5.set_xlabel("Crab: Forward")
    p5.xaxis.set_label_position("top")
    p5.text(0.01,0.99,"(c)",transform=p5.transAxes,ha='left', va='top')
    
    p6=fig.add_subplot(grid[1,2])
    d=rd.read_h5py_file()
    d=d=rd.cut_selection(d,9,0,False)
    pc.create_ARM_plot(d,50,2)
    p6.set_xlabel("Crab: Backward")
    p6.xaxis.set_label_position("top")
    p6.text(0.01,0.99,"(f)",transform=p6.transAxes,ha='left', va='top')
    
    
    plt.savefig(rd.p2+'ARM.pdf',bbox_inches='tight')
    

def plot_pure_background_angles():
    fig=plt.figure(figsize=(12,20))
    grid=plt.GridSpec(6, 3,hspace=0.15,wspace=0.25)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)
    
    rows=[rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cra0,pc.h5_cra0),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cra15,pc.h5_cra15),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cos,pc.h5_cos),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_alb,pc.h5_alb),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_act,pc.h5_act),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_act,pc.h5_act_f),9,0,True)]
    od=rd.read_h5py_file()
    od=rd.cut_selection(od,9,0,True)
    
    ToF_cuts=[(4e-9,7.5e-9),
              (4e-9,7.5e-9),
              (4e-9,7.5e-9),
              (0e-9,8e-9),
              (1e-9,8e-9),
              (0e-9,8e-9)]
    
    letters=np.array([["a","b","c"],
                 ["d","e","f"],
                 ["g","h","i"],
                 ["j","k","l"],
                 ["m","n","o"],
                 ["p","q","r"]])
    
    labels=["Crab without CGRO","Residual"]
    lab=[None,None]
    for row in range(6):
        s=50 if not row==5 else 15
        
        
        column=0
        temp=fig.add_subplot(grid[row,0])
        d=rows[row]
        d=rd.slice_selection(d, 17, ToF_cuts[row][0], ToF_cuts[row][1])
        x1,y1=pc.plot_ToF(d,False,s)
        
        od_t=rd.slice_selection(od, 17, ToF_cuts[row][0], ToF_cuts[row][1])
        x2,y2=pc.plot_ToF(od_t,True,s)
        y2=y2*np.amax(y1)/np.amax(y2)
        a=minimize(rd.fit_func,1,(y1,y2)).x
        temp.axhline(0,c="black")
        plt.plot(x2,y2*a,lw=2.5)
        plt.plot(x2,y1-y2*a,lw=2.5,c="C3")
        plt.xlim(x1[0],x1[-1])
        temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==0:
            temp.set_xlabel("Time of Flight")
            temp.xaxis.set_label_position("top")
        elif row==5:
            temp.set_xlabel("Time of Flight [ns]")
            
        if y1[1]/np.amax(y1)>0.9 or y1[2]/np.amax(y1)>0.9:
            plt.ylim(None,np.amax(y1)*1.15)
        
        column=1
        temp=fig.add_subplot(grid[row,1])
        d=rows[row]
        x1,y1=pc.plot_flight_angle(d,False,s)
        
        x2,y2=pc.plot_flight_angle(od,True,s)
        y2=y2*np.amax(y1)/np.amax(y2)
        a=minimize(rd.fit_func,1,(y1,y2)).x
        temp.axhline(0,c="black")
        plt.plot(x2,y2*a,lw=2.5)
        plt.plot(x2,y1-y2*a,lw=2.5,c="C3")
        plt.xlim(x1[0],x1[-1])
        temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==0:
            temp.set_xlabel("Flight Angle")
            temp.xaxis.set_label_position("top")
        elif row==5:
            temp.set_xlabel("Flight Angle [rad]")
        
        column=2
        if row==4:
            lab=labels
        
        temp=fig.add_subplot(grid[row,2])
        d=rows[row]
        x1,y1=pc.plot_ARM(d,False,s)
        
        x2,y2=pc.plot_ARM(od,True,s)
        y2=y2*np.amax(y1)/np.amax(y2)
        a=minimize(rd.fit_func,1,(y1,y2)).x
        temp.axhline(0,c="black")
        plt.plot(x2,y2*a,lw=2.5,label=lab[0])
        plt.plot(x2,y1-y2*a,lw=2.5,c="C3",label=lab[1])
        plt.xlim(x1[0],x1[-1])
        temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==0:
            temp.set_xlabel("ARM")
            temp.xaxis.set_label_position("top")
        elif row==5:
            temp.set_xlabel("ARM [rad]")
        if row==0:
            temp.set_ylabel("Crab at 0rad")
            temp.yaxis.set_label_position("right")
        elif row==1:
            temp.set_ylabel("Crab at 0.26rad")
            temp.yaxis.set_label_position("right")
        elif row==2:
            temp.set_ylabel("Cosmic Photons")
            temp.yaxis.set_label_position("right")
        elif row==3:
            temp.set_ylabel("Albedo Photons")
            temp.yaxis.set_label_position("right")
        elif row==4:
            temp.set_ylabel("Primary Proton Activation: Slow")
            temp.yaxis.set_label_position("right")
        elif row==5:
            temp.set_ylabel("Primary Proton Activation: Fast")
            temp.yaxis.set_label_position("right")
            plt.legend(ncol=2,loc="lower right",bbox_to_anchor=(1,-0.35))
        
    plt.savefig(rd.p2+'Back_Ang.pdf',bbox_inches='tight')


def plot_pure_background_energies():
    fig=plt.figure(figsize=(12,20))
    grid=plt.GridSpec(6, 3,hspace=0.15,wspace=0.25)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)
    
    rows=[rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cra0,pc.h5_cra0),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cra15,pc.h5_cra15),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cos,pc.h5_cos),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_alb,pc.h5_alb),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_act,pc.h5_act),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_act,pc.h5_act_f),9,0,True)]
    od=rd.read_h5py_file()
    od=rd.cut_selection(od,9,0,True)
    
    
    letters=np.array([["a","b","c"],
                 ["d","e","f"],
                 ["g","h","i"],
                 ["j","k","l"],
                 ["m","n","o"],
                 ["p","q","r"]])
    
    labels=["Crab without CGRO","Residual"]
    lab=[None,None]
    for row in range(6):
        s=50 if not row==5 else 15
        column=0
        temp=fig.add_subplot(grid[row,0])
        d=rows[row]
        x1,y1=pc.plot_E1(d,False,s)
        
        x2,y2=pc.plot_E1(od,True,s)
        y2=y2*np.amax(y1)/np.amax(y2)
        a=minimize(rd.fit_func,1,(y1,y2)).x
        temp.axhline(0,c="black")
        plt.plot(x2,y2*a,lw=2.5)
        plt.plot(x2,y1-y2*a,lw=2.5,c="C3")
        plt.xlim(x1[0],x1[-1])
        temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
        
        if row==0:
            temp.set_xlabel("E1")
            temp.xaxis.set_label_position("top")
        elif row==5:
            temp.set_xlabel("E1 [MeV]")

        if y1[3]/np.amax(y1)>0.9 or y1[4]/np.amax(y1)>0.9 or y2[1]/np.amax(y2)>0.9/a or y2[2]/np.amax(y2)>0.9/a:
            plt.ylim(None,max(np.amax(y1),a*np.amax(y2))*1.15)
        
        column=1
        temp=fig.add_subplot(grid[row,1])
        d=rows[row]
        x1,y1=pc.plot_E2(d,False,s)
        
        x2,y2=pc.plot_E2(od,True,s)
        y2=y2*np.amax(y1)/np.amax(y2)
        a=minimize(rd.fit_func,1,(y1,y2)).x
        temp.axhline(0,c="black")
        plt.plot(x2,y2*a,lw=2.5)
        plt.plot(x2,y1-y2*a,lw=2.5,c="C3")
        plt.xlim(x1[0],x1[-1])
        temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
        
        if row==0:
            temp.set_xlabel("E2")
            temp.xaxis.set_label_position("top")
        elif row==5:
            temp.set_xlabel("E2 [MeV]")
        
        if y1[3]/np.amax(y1)>0.9 or y1[4]/np.amax(y1)>0.9 or y2[1]/np.amax(y2)>0.9/a or y2[2]/np.amax(y2)>0.9/a:
            plt.ylim(None,max(np.amax(y1),a*np.amax(y2))*1.15)
        
        column=2
        if row==5:
            lab=labels
        
        temp=fig.add_subplot(grid[row,2])
        d=rows[row]
        x1,y1=pc.plot_ET(d,False,s)
        
        x2,y2=pc.plot_ET(od,True,s)
        y2=y2*np.amax(y1)/np.amax(y2)
        a=minimize(rd.fit_func,1,(y1,y2)).x
        temp.axhline(0,c="black")
        plt.plot(x2,y2*a,lw=2.5,label=lab[0])
        plt.plot(x2,y1-y2*a,lw=2.5,c="C3",label=lab[1])
        plt.xlim(x1[0],x1[-1])
        temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
        
        if row==0:
            temp.set_xlabel("E1+E2")
            temp.xaxis.set_label_position("top")
        elif row==5:
            temp.set_xlabel("E1+E2 [MeV]")

        if y1[3]/np.amax(y1)>0.9 or y1[4]/np.amax(y1)>0.9 or y2[1]/np.amax(y2)>0.9/a or y2[2]/np.amax(y2)>0.9/a:
            plt.ylim(None,max(np.amax(y1),a*np.amax(y2))*1.15)

        if row==0:
            temp.set_ylabel("Crab at 0rad")
            temp.yaxis.set_label_position("right")
        elif row==1:
            temp.set_ylabel("Crab at 0.26rad")
            temp.yaxis.set_label_position("right")
        elif row==2:
            temp.set_ylabel("Cosmic Photons")
            temp.yaxis.set_label_position("right")
        elif row==3:
            temp.set_ylabel("Albedo Photons")
            temp.yaxis.set_label_position("right")
        elif row==4:
            temp.set_ylabel("Primary Proton Activation: Slow")
            temp.yaxis.set_label_position("right")
        elif row==5:
            temp.set_ylabel("Primary Proton Activation: Fast")
            temp.yaxis.set_label_position("right")
            plt.legend(ncol=2,loc="lower right",bbox_to_anchor=(1,-0.40))
        
        
    plt.savefig(rd.p2+'Back_En.pdf',bbox_inches='tight')


def plot_background_angles():
    fig=plt.figure(figsize=(12,18))
    grid=plt.GridSpec(5, 3,hspace=0.15,wspace=0.25)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)
    

    od=rd.read_h5py_file()
    od=rd.cut_selection(od,9,0,True)
    
    
    letters=np.array([["a","b","c"],
                 ["d","e","f"],
                 ["g","h","i"],
                 ["j","k","l"],
                 ["m","n","o"]])
    
    d=rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cra0,pc.h5_cra0),9,0,True)
    
    temp=fig.add_subplot(grid[0,0])
    x1,y1=pc.plot_ToF_sum(False)
    
    t=rd.extract_parameter(d, 17)*1e9
    x=np.linspace(0,8,50)
    n,bins=np.histogram(t,bins=x)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    y2=n*pc.t_act_f/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_xlabel("Time of Flight")
    temp.xaxis.set_label_position("top")
    temp.text(0.01,0.99,"(a)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[1,0])
    x1,y1=pc.plot_ToF_sum(True)
    y2=n*pc.t_alb/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.text(0.01,0.99,"(d)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[0,1])
    x1,y1=pc.plot_flight_angle_sum(False)
    a=rd.calculate_angles(d)
    x=np.linspace(0,0.7,50)
    n,bins=np.histogram(a,bins=x)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    y2=n*pc.t_act_f/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_xlabel("Flight Angle")
    temp.xaxis.set_label_position("top")
    temp.text(0.01,0.99,"(b)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[1,1])
    x1,y1=pc.plot_flight_angle_sum(True)
    y2=n*pc.t_alb/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.text(0.01,0.99,"(e)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[0,2])
    x1,y1=pc.plot_ARM_sum(False)
    E_e=d[:,3]
    E_g=d[:,1]
    c=2.998e8
    m=511/c**2
    phibar=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    phigeo=np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])
    ARM=phibar-phigeo
    x=np.linspace(-0.5,0.75,50)
    n,bins=np.histogram(ARM,bins=x)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    y2=n*pc.t_act_f/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_xlabel("ARM")
    temp.xaxis.set_label_position("top")
    temp.set_ylabel("Total Simulated Distribution")
    temp.yaxis.set_label_position("right")
    temp.text(0.01,0.99,"(c)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[1,2])
    x1,y1=pc.plot_ARM_sum(True)
    y2=n*pc.t_alb/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_ylabel("Total Simulated Distribution\n without Fast Decays")
    temp.yaxis.set_label_position("right")
    temp.text(0.01,0.99,"(f)",transform=temp.transAxes,ha='left', va='top')
    
    
    od= pc.load_od()
    for row in range(3):
        column=0
        temp=fig.add_subplot(grid[row+2,0])
        t=rd.extract_parameter(od[row], 17)*1e9
        x=np.linspace(0,8,33)
        n,bins=np.histogram(t,bins=x)
        n2,bins,patches=plt.hist(bins[:-1],bins=x,weights=n)
        temp.text(0.01,0.99,"("+letters[row+2,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==2:
            temp.set_xlabel("Time of Flight [ns]")
        
        column=1
        temp=fig.add_subplot(grid[row+2,1])
        pc.plot_flight_angle(od[row])
        temp.text(0.01,0.99,"("+letters[row+2,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==2:
            temp.set_xlabel("Flight Angle [rad]")
        
        column=2
        temp=fig.add_subplot(grid[row+2,2])
        pc.plot_ARM(od[row])
        temp.text(0.01,0.99,"("+letters[row+2,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==0:
            temp.set_ylabel("Real Data: No Cuts")
            temp.yaxis.set_label_position("right")
        elif row==1:
            temp.set_ylabel("Real Data: PSD Cut")
            temp.yaxis.set_label_position("right")
        elif row==2:
            temp.set_xlabel("ARM [rad]")
            temp.set_ylabel("Real Data: Standard Cuts")
            temp.yaxis.set_label_position("right")
        
    
    
    plt.savefig(rd.p2+'Back_Ang_Real.pdf',bbox_inches='tight')

    
def plot_background_energies():
    fig=plt.figure(figsize=(12,18))
    grid=plt.GridSpec(5, 3,hspace=0.15,wspace=0.25)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)
    

    od=rd.read_h5py_file()
    od=rd.cut_selection(od,9,0,True)
    
    
    letters=np.array([["a","b","c"],
                 ["d","e","f"],
                 ["g","h","i"],
                 ["j","k","l"],
                 ["m","n","o"]])
    
    d=rd.cut_selection(rd.read_h5py_file(rd.p3+pc.p_cra0,pc.h5_cra0),9,0,True)
    
    temp=fig.add_subplot(grid[0,0])
    x1,y1=pc.plot_E1_sum(False)
    
    t=rd.extract_parameter(d, 3)/1000
    x=np.geomspace(0.04,30,50)
    n,bins=np.histogram(t,bins=x)
    x_dist=np.zeros(len(n))
    for i in range(len(n)):
        x_dist[i]=np.exp( (np.log(bins[i])+np.log(bins[i+1]))/2 )
    y2=n*pc.t_act_f/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_xlabel("E1")
    temp.xaxis.set_label_position("top")
    temp.text(0.01,0.99,"(a)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[1,0])
    x1,y1=pc.plot_E1_sum(True)
    y2=n*pc.t_alb/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.text(0.01,0.99,"(d)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[0,1])
    x1,y1=pc.plot_E2_sum(False)
    a=rd.extract_parameter(d, 3)/1000
    x=np.geomspace(0.4,30,50)
    n,bins=np.histogram(a,bins=x)
    x_dist=np.zeros(len(n))
    for i in range(len(n)):
        x_dist[i]=np.exp( (np.log(bins[i])+np.log(bins[i+1]))/2 )
    y2=n*pc.t_act_f/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_xlabel("E2")
    temp.xaxis.set_label_position("top")
    temp.text(0.01,0.99,"(b)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[1,1])
    x1,y1=pc.plot_E2_sum(True)
    y2=n*pc.t_alb/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.text(0.01,0.99,"(e)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[0,2])
    x1,y1=pc.plot_ET_sum(False)
    or_a=rd.extract_parameter(d, 1)/1000+rd.extract_parameter(d, 3)/1000
    x=np.geomspace(0.4,40,50)
    n,bins=np.histogram(or_a,bins=x)
    x_dist=np.zeros(len(n))
    for i in range(len(n)):
        x_dist[i]=np.exp( (np.log(bins[i])+np.log(bins[i+1]))/2 )
    y2=n*pc.t_act_f/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_xlabel("E1+E2")
    temp.xaxis.set_label_position("top")
    temp.set_ylabel("Total Simulated Distribution")
    temp.yaxis.set_label_position("right")
    temp.text(0.01,0.99,"(c)",transform=temp.transAxes,ha='left', va='top')
    
    temp=fig.add_subplot(grid[1,2])
    x1,y1=pc.plot_ET_sum(True)
    y2=n*pc.t_alb/pc.t_cra0
    plt.plot(x_dist,y2,lw=2.5)
    temp.set_ylabel("Total Simulated Distribution\n without Fast Decays")
    temp.yaxis.set_label_position("right")
    temp.text(0.01,0.99,"(f)",transform=temp.transAxes,ha='left', va='top')
    
    
    od= pc.load_od()
    for row in range(3):
        column=0
        temp=fig.add_subplot(grid[row+2,0])
        pc.plot_E1(od[row])
        temp.text(0.01,0.99,"("+letters[row+2,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==2:
            temp.set_xlabel("E1 [MeV]")
        
        column=1
        temp=fig.add_subplot(grid[row+2,1])
        pc.plot_E2(od[row])
        temp.text(0.01,0.99,"("+letters[row+2,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==2:
            temp.set_xlabel("E2 [MeV]")
        
        column=2
        temp=fig.add_subplot(grid[row+2,2])
        pc.plot_ET(od[row])
        temp.text(0.01,0.99,"("+letters[row+2,column]+")",transform=temp.transAxes,ha='left', va='top')
        if row==0:
            temp.set_ylabel("Real Data: No Cuts")
            temp.yaxis.set_label_position("right")
        elif row==1:
            temp.set_ylabel("Real Data: PSD Cut")
            temp.yaxis.set_label_position("right")
        elif row==2:
            temp.set_xlabel("E1+E2 [MeV]")
            temp.set_ylabel("Real Data: Standard Cuts")
            temp.yaxis.set_label_position("right")
        
    
    
    plt.savefig(rd.p2+'Back_En_Real.pdf',bbox_inches='tight')
        

def plot_final_cuts():
    fig=plt.figure(figsize=(12,16))
    grid=plt.GridSpec(6, 3,hspace=0.15,wspace=0.25)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)
    pa.set_xlabel("Angle of Incidence [rad]",labelpad=30)

    columns=((True,True),
             (True,False),
             (False,True))
    rows1=((100,100,1,1,1),
          (100,1,100,1,1),
          (100,1,1,100,1),
          (100,1,1,1,100),
          (1,1,1,100,100),
          (100,1,1,100,100))
    
    rows2=((1,100,100,1,1),
          (100,100,100,1,1),
          (1,100,100,1,100),
          (1,10,10,1,10))
          
    letters=np.array([["a","b","c"],
                 ["d","e","f"],
                 ["g","h","i"],
                 ["j","k","l"],
                 ["m","n","o"],
                 ["p","q","r"]])
    
    c_labels=["Soft Cut \n with Fast Activation", "Soft Cut \n without Fast Activation", "Hard Cut"]
    
    
    for row in range(len(rows1)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            
            print(0,row,column)
            
            if row==0:
                temp.set_xlabel(c_labels[column])
                temp.xaxis.set_label_position("top")
            
            
            pc.parameter_cut(rows1[row],columns[column][0],columns[column][1],True)
            if column==2:
                temp.set_ylabel(str(rows1[row]))
                temp.yaxis.set_label_position("right")
            temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
            
    
            if row==5 and column==2:
                custom_lines = [Line2D([0], [0], color="C1", lw=3),
                            Line2D([0], [0], color="C2", lw=3)]
                temp.legend(custom_lines, ['Crab Simulation', 'Original Data'],loc="lower right",bbox_to_anchor=(1,-0.5))
    
    plt.savefig(rd.p2+'Final_Cuts1.pdf',bbox_inches='tight')
        
    plt.clf()
    
    fig=plt.figure(figsize=(11,11))
    grid=plt.GridSpec(4, 3,hspace=0.15,wspace=0.25)
    
    pa=fig.add_subplot(grid[:,:])
    pa.spines['top'].set_color('none')
    pa.spines['bottom'].set_color('none')
    pa.spines['left'].set_color('none')
    pa.spines['right'].set_color('none')
    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
    pa.set_ylabel("Frequency [Counts/Bin]",labelpad=45)
    pa.set_xlabel("ARM [rad]",labelpad=30)
    
    for row in range(len(rows2)):
        for column in range(len(columns)):
            temp=fig.add_subplot(grid[row,column])
            
            print(1,row,column)
            
            if row==0:
                temp.set_xlabel(c_labels[column])
                temp.xaxis.set_label_position("top")
            
            
            pc.parameter_cut(rows2[row],columns[column][0],columns[column][1],True)
            if column==2:
                temp.set_ylabel(str(rows2[row]))
                temp.yaxis.set_label_position("right")
            temp.text(0.01,0.99,"("+letters[row,column]+")",transform=temp.transAxes,ha='left', va='top')
            
    
            if row==3 and column==2:
                custom_lines = [Line2D([0], [0], color="C1", lw=3),
                            Line2D([0], [0], color="C2", lw=3)]
                temp.legend(custom_lines, ['Crab Simulation', 'Original Data'],loc="lower right",bbox_to_anchor=(1,-0.5))
    

    plt.savefig(rd.p2+'Final_Cuts2.pdf',bbox_inches='tight')








def test():
    array=np.array([[1,1,2,3,5],
                   [-1,2,3,6,3],
                   [4,4,3,2,4],
                   [6,43,7,4,4]])
    print(np.min(array))
    n,bins= np.histogramdd(array,bins=(2,1,2,1,1),range=((0,6),(1,10),None,None,None))
    print(n[1,0,0,0,0])
    return n,bins

























