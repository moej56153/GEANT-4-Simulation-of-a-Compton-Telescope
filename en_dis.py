import numpy as np
import matplotlib.pyplot as plt
import reading_data as rd
import ang_tof as at
import pos_dis as pd
import att_func as af
from scipy.integrate import quad



def SAD_E(forward,num_bins=100,data_array=None):
    d=rd.read_h5py_file() if type(data_array)==type(None) else data_array
    d=rd.cut_selection(d, 9, 0, forward)
    E_e=rd.extract_parameter(d, 3)
    E_g=rd.extract_parameter(d, 1)
    c=2.998e8
    m=511/c**2
    a=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    n,bins,patches=plt.hist(a,num_bins)
    if forward:
        D=pd.NF
        x_min=at.fsad_min
        x_max=at.fsad_max
    else:
        D=pd.NB
        x_min=at.bsad_min
        x_max=at.bsad_max
    D=D*np.amax(n)/np.amax(D)
    rd.plot_points(D,x_min,x_max,2.0)
    # plt.xlabel("Scattering Angle [rad]")
    # plt.ylabel("Frequency [Counts/Bins]")
    
    # s=0
    # for i in range(num_bins):
    #     ang=(bins[i+1]+bins[i])/2
    #     dif=n[i]-rd.continue_line(ang,D,0,0.7) if ang>0.70 else 0
    #     print(ang,dif)
    #     s+=dif
    # return s,len(d)

    



def SAD_E_vs_Pos(forward,size=100,data_array=None):
    d=rd.read_h5py_file() if type(data_array)==None else data_array
    d=rd.cut_selection(d, 9, 0, forward)
    E_e=rd.extract_parameter(d, 3)
    E_g=rd.extract_parameter(d, 1)
    c=2.998e8
    m=511/c**2
    a_E=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    a_P=rd.calculate_angles(d)
    c=0
    while c< len(a_E):
        if np.isnan(a_E[c]):
            a_E=np.delete(a_E,c)
            a_P=np.delete(a_P,c)
            c+=-1
        c+=1
    plt.hist2d(a_E,a_P,bins=(size,size))
    if forward:
        x=np.linspace(at.fsad_min,at.fsad_max,size)
        plt.plot(x,x,color="C1")
    else:
        x=np.linspace(at.bsad_min,at.bsad_max,size)
        plt.plot(x,x,color="C1")
    # plt.xlabel("Scattering Angle via Energies [rad]")
    # plt.ylabel("Scattering Angle via Positions [rad]")


def SAD_CROSS_GOOD(forward,num_bins=50):
    d=rd.read_h5py_file()
    d=rd.cut_selection(d, 9, 0, forward)
    # E_e=rd.extract_parameter(d, 3)
    # E_g=rd.extract_parameter(d, 1)
    c=2.998e8
    m=511/c**2
    # a_E=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    # a_P=rd.calculate_angles(d)
    # a=np.array([])
    # for i in range(len(a_E)):
    #     if abs(a_E[i]-a_P[i])<0.01:
    #         a=np.append(a,a_P[i])
    margin=0.01
    d=d[abs( np.arccos( 1 - (d[:,3]/d[:,1]*m*c**2/(d[:,3]+d[:,1])) ) 
            -np.arccos( -1*(-d[:,9]+d[:,15]) / d[:,24]) )
            <=margin]
    a=rd.calculate_angles(d)
    n,bins,patches=plt.hist(a,num_bins)
    if forward:
        D=pd.NF
        x_min=at.fsad_min
        x_max=at.fsad_max
    else:
        D=pd.NB
        x_min=at.bsad_min
        x_max=at.bsad_max
    D=D*np.amax(n)/np.amax(D)
    rd.plot_points(D,x_min,x_max)
    plt.xlabel("Scattering Angle [rad]")
    plt.ylabel("Frequency [Counts/Bins]")
            

def plot_E_dist(forward,num_bins=300,):
    x_min=0000
    x_max=5000
    y_min=0
    #y_max=140000
    scale1=1.8
    scale2=1

    
    d=rd.read_h5py_file()
    d=rd.cut_selection(d, 9, 0, forward)
    E=rd.extract_parameter(d, 1) + rd.extract_parameter(d, 3)
    n,bins,patches=plt.hist(E,num_bins)
    x=np.linspace(x_min,x_max,100)
    y=np.array([at.energy_distribution(i) for i in x])
    y=y*np.amax(n)/np.amax(y)*scale1
    plt.plot(x,y,label="A")
    
    if forward:
        t_min,t_max,sad=at.fsad_min,at.fsad_max,at.fsad
    else:
        t_min,t_max,sad=at.bsad_min,at.bsad_max,at.bsad
    y2=np.array([quad(integrand1,t_min,t_max,(i,forward,t_min,t_max,sad),0,10e-5,10e-5)[0] for i in x])
    y2=y2*np.amax(n)/np.amax(y2)*scale2
    plt.plot(x,y2,label="B")
    
    

    plt.xlabel("Total Detected Energy [keV]")
    plt.ylabel("Frequency [Counts/Bin]")
    plt.legend()
    
    plt.xlim(x_min,x_max)
    plt.ylim(0,50000)
    
def plot_E_all():
    scale1=1.7
    x_min=0
    x_max=6000
    
    
    d=rd.read_h5py_file()
    d=rd.cut_selection(d, 9, 0, True)
    E=rd.extract_parameter(d, 1)# + rd.extract_parameter(d, 3)
    n,bins,patches=plt.hist(E,500)
    x=np.linspace(x_min,x_max,200)
    y=np.array([at.energy_distribution(i) for i in x])
    y=y*np.amax(n)/np.amax(y)*scale1
    #plt.plot(x,y,label="A")
    
    plt.xlim(x_min,x_max)
    plt.xlabel("Detected Energy [keV]")
    plt.ylabel("Frequency [Counts/Bin]")


def integrand1(theta,E,forward,x_min,x_max,sad):
    p1=at.klein_nishina(E, theta)*10e30
    p2=rd.continue_line(theta,sad,x_min,x_max)
    p3=at.energy_limit_factor(E,theta,forward)
    p4=at.energy_distribution(E)
    return p1*p2*p3*p4

def integrand2(theta,E,forward,x_min,x_max,sad): 
    p1=at.klein_nishina(E, theta)*10e30
    p2=rd.continue_line(theta,sad,x_min,x_max)
    p3=at.energy_limit_factor(E,theta,forward)
    p4=at.energy_distribution(E)
    p5=rd.continue_log(E/1000,af.E,af.prob_NE213)
    return p1*p2*p3*p4*(1-p5)












