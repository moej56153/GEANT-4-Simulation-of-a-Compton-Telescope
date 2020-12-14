import numpy as np
import matplotlib.pyplot as plt
import h5py
import reading_data as rd
from scipy.optimize import minimize




def create_AoI_plot(d,num_bins=200,rang=0.5,ploto=True):
    E_e=rd.extract_parameter(d, 3)
    E_g=rd.extract_parameter(d, 1)
    c=2.998e8
    m=511/c**2
    a=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    tra_a=np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])*np.sign(d[:,11]-d[:,5])
    or_a=np.zeros(len(tra_a))
    for i in range(len(or_a)):
        if abs(tra_a[i]-a[i])<abs(tra_a[i]+a[i]):
            or_a[i]=tra_a[i]-a[i]
        else:
            or_a[i]=tra_a[i]+a[i]
    or_a=or_a[or_a[:]>-rang]
    or_a=or_a[or_a[:]<rang]
    if not ploto:
        n,bins=np.histogram(or_a,num_bins)

    else:
        n,bins,patches=plt.hist(or_a,num_bins)
    bins=bins[:-1]+(bins[1]-bins[0])/2
    return n,bins

def create_ARM_plot(d,num_bins=200,rang=0.5,ploto=True):
    E_e=d[:,3]
    E_g=d[:,1]
    c=2.998e8
    m=511/c**2
    phibar=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    phigeo=np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])
    ARM=phibar-phigeo
    ARM=ARM[ np.logical_and( ARM[:]>-rang, ARM[:]<rang ) ]
    if not ploto:
        n,bins=np.histogram(ARM,num_bins)

    else:
        n,bins,patches=plt.hist(ARM,num_bins)
    bins=bins[:-1]+(bins[1]-bins[0])/2
    return n,bins
    
    
    
def abs_min_return(a,b):
    if abs(a)<=abs(b):
        return a
    else:
        return b

path="C:\\Users\\moell\\Desktop\\COMPTEL_Simulation_Project\\Werkstudent\\Data_from_Simulations\\Thesis_Simulations\\Real_COMPTEL_Data/"

file_name="vp426_43-30_all.ascii.txt"
h5_name="h5_vp426_09-43"
length=1546605
h5_conv_name="h5_vp426_09-43_c"

# file_name="vp426_43-30_all.ascii.txt"
# h5_name="h5_vp426_43-30"
# length=426659
# h5_conv_name="h5_vp426_43-30_c"

data_le="h5_vp426_09-43_c"
data_he="h5_vp426_43-30_c"

p_act="Activation/"
h5_act="Activation1"
t_act=15*3.6e4
h5_act_f="Activation_fast"
t_act_f=5*15
p_alb="Albedo/"
h5_alb="Albedo1"
t_alb=1.2e4
p_cos="Cosmic/"
h5_cos="Cosmic1"
t_cos=4e5
p_cra0="Crab_0/"
h5_cra0="Crab_0"
t_cra0=3e6
p_cra15="Crab_15/"
h5_cra15="Crab_15"

p_pre='C:\\Users\\moell\\Desktop\\COMPTEL_Simulation_Project\\Werkstudent\\Data_from_Simulations\\Thesis_Simulations/'

def load_od():
    od=np.vstack((rd.read_h5py_file(path,data_le),rd.read_h5py_file(path,data_he)))
    od_psd=rd.slice_selection(od, 21, 60, 90)
    od_all=rd.slice_selection(od_psd, 3, 70, 20000)
    od_all=rd.slice_selection(od_all, 1, 650, 30000)
    od_all=rd.slice_selection(od_all, 17, 4e-9, 7e-9)
    c=2.998e8
    m=511/c**2
    od_all=od_all[ abs(20*np.pi/180-abs(np.arccos( 1 - (od_all[:,3]/od_all[:,1]*m*c**2/(od_all[:,3]+od_all[:,1])) )) <= 16*np.pi/180 )]
    od_all=od_all[od_all[:,22]>0]
    od_all=od_all[od_all[:,23]>0]
    return od, od_psd, od_all



raw_sim_data=[rd.cut_selection(rd.read_h5py_file(p_pre+p_cra0,h5_cra0),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_pre+p_cos,h5_cos),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_pre+p_alb,h5_alb),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_pre+p_act,h5_act),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_pre+p_act,h5_act_f),9,0,True)]

sim_times=np.array([t_cra0,
                    t_cos,
                    t_alb,
                    t_act,
                    t_act_f])

parameters={0:"Line",
            1:"Time",
            2:"E1",
            3:"E2",
            4:"E Total",
            5:"Phibar",
            6:"ToF",
            7:"PSD",
            8:"Zeta",
            9:"Rigidty",
            10:"X:D1",
            11:"X:D2",
            12:"Y:D1",
            13:"Y:D2",
            14:"ARM"}


def calculate_data_rates():
    data=np.array([rd.cut_selection(rd.read_h5py_file(p_cra0,h5_cra0),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_cos,h5_cos),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_alb,h5_alb),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_act,h5_act),9,0,True),
                   rd.cut_selection(rd.read_h5py_file(p_act,h5_act_f),9,0,True)])
    
    datat=np.array([rd.read_h5py_file(p_cra0,h5_cra0),
                   rd.read_h5py_file(p_cos,h5_cos),
                   rd.read_h5py_file(p_alb,h5_alb),
                   rd.read_h5py_file(p_act,h5_act),
                   rd.read_h5py_file(p_act,h5_act_f)])

    for i in range(len(data)):
        E=data[i].shape[0]
        ET=datat[i].shape[0]
        print(E,E/ET,E/sim_times[i],sim_times[i])
        
    


def import_real_data(p=path,
                     f=file_name,
                     h5=h5_name,
                     l=length):
    np_data=np.zeros((l,15))
    print(np_data.shape)
    with open(p+f, "r") as data:
        for i in range(l):
            line=data.readline()
            line=" "+line.strip()+" "
            insert_pos={1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14}
            np_data[i,:]=rd.line_extracter(np_data[i,:],line,insert_pos)
            np_data[i,0]=i
            if i%100000==0:
                print(i)
    with h5py.File(p+h5,"w") as h5_f:
        h5_f.create_dataset("Data", data=np_data)
    

def read_real_data(h5=h5_name,p=path):
    with h5py.File(p+h5,"r") as h5_f:
        data_array=np.array(h5_f.get("Data"))
    return data_array


def data_converter(h5=h5_name,l=length,h5_c=h5_conv_name,p=path):
    d=read_real_data()
    dc=np.zeros((l,25))
    for i in range(l):
        dc[i,0]=d[i,0]
        forward=True if d[i,6]>=100 else False
        dc[i,1]=d[i,3] if forward else d[i,2]
        dc[i,3]=d[i,2] if forward else d[i,3]
        dc[i,5]=d[i,10]/10 if forward else d[i,12]/10
        dc[i,7]=d[i,11]/10 if forward else d[i,13]/10
        dc[i,9]=102.35 if forward else -55.65
        dc[i,11]=d[i,12]/10 if forward else d[i,10]/10
        dc[i,13]=d[i,13]/10 if forward else d[i,11]/10
        dc[i,15]=-55.65 if forward else 102.35
        dc[i,17]=abs(d[i,6]-100)*0.25e-9
        dc[i,19]=d[i,1]
        dc[i,21]=d[i,7]
        dc[i,22]=d[i,8]
        dc[i,23]=d[i,9]
        dc[i,24]=np.sqrt( (dc[i,11]-dc[i,5])**2 + (dc[i,13]-dc[i,7])**2 + (dc[i,15]-dc[i,9])**2 )
        if i%100000==0:
                print(i)
    with h5py.File(p+h5_c,"w") as h5_f:
        h5_f.create_dataset("Data", data=dc)


    
def plot_ToF(d,return_values=False,num_bins=100):
    t=rd.extract_parameter(d, 17)*1e9
    if return_values:
        n,bins=np.histogram(t,num_bins)
        x_dist=bins[:-1]+(bins[1]-bins[0])/2
        return x_dist,n
    n,bins,patches=plt.hist(t,num_bins)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    return x_dist,n

def plot_flight_angle(d,return_values=False,num_bins=100):
    a=rd.calculate_angles(d)
    if return_values:
        n,bins=np.histogram(a,num_bins)
        x_dist=bins[:-1]+(bins[1]-bins[0])/2
        return x_dist,n
    n,bins,patches=plt.hist(a,num_bins)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    return x_dist,n


def plot_AoI(d,return_values=False,num_bins=100):
    E_e=rd.extract_parameter(d, 3)
    E_g=rd.extract_parameter(d, 1)
    c=2.998e8
    m=511/c**2
    a=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    tra_a=np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])*np.sign(d[:,11]-d[:,5])
    or_a=np.zeros(len(tra_a))
    for i in range(len(or_a)):
        if abs(tra_a[i]-a[i])<abs(tra_a[i]+a[i]):
            or_a[i]=tra_a[i]-a[i]
        else:
            or_a[i]=tra_a[i]+a[i]
    or_a=or_a[or_a[:]>-0.5]
    or_a=or_a[or_a[:]<0.5]
    if return_values:
        n,bins=np.histogram(or_a,num_bins)
        x_dist=bins[:-1]+(bins[1]-bins[0])/2
        return x_dist,n
    n,bins,patches=plt.hist(or_a,num_bins)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    return x_dist,n

def plot_ARM(d,return_values=False,num_bins=100,range1=-0.5,range2=0.75):
    E_e=d[:,3]
    E_g=d[:,1]
    c=2.998e8
    m=511/c**2
    phibar=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
    phigeo=np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])
    ARM=phibar-phigeo
    ARM=ARM[ np.logical_and( ARM[:]>range1, ARM[:]<range2 ) ]
    if return_values:
        n,bins=np.histogram(ARM,num_bins)
        x_dist=bins[:-1]+(bins[1]-bins[0])/2
        return x_dist,n
    n,bins,patches=plt.hist(ARM,num_bins)
    x_dist=bins[:-1]+(bins[1]-bins[0])/2
    return x_dist,n


def plot_E1(d,return_values=False,num_bins=100):
    E=rd.extract_parameter(d, 3)/1000
    logbins = np.geomspace(0.04, 30, num_bins)
    if return_values:
        n,bins=np.histogram(E,bins=logbins)
        x_dist=np.zeros(len(logbins)-1)
        for i in range(len(x_dist)):
            x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
        return x_dist,n
    n,bins,patches=plt.hist(E,bins=logbins)
    plt.xscale("log")
    x_dist=np.zeros(len(logbins)-1)
    for i in range(len(x_dist)):
        x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
    return x_dist,n


def plot_E2(d,return_values=False,num_bins=100):
    E=rd.extract_parameter(d, 1)/1000
    logbins = np.geomspace(0.4, 30, num_bins)
    if return_values:
        n,bins=np.histogram(E,bins=logbins)
        x_dist=np.zeros(len(logbins)-1)
        for i in range(len(x_dist)):
            x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
        return x_dist,n
    n,bins,patches=plt.hist(E,bins=logbins)
    plt.xscale("log")
    x_dist=np.zeros(len(logbins)-1)
    for i in range(len(x_dist)):
        x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
    return x_dist,n

def plot_E2_vert(d,return_values=False,num_bins=100):
    E=rd.extract_parameter(d, 1)/1000
    logbins = np.geomspace(0.4, 30, num_bins)
    if return_values:
        n,bins=np.histogram(E,bins=logbins)
        x_dist=np.zeros(len(logbins)-1)
        for i in range(len(x_dist)):
            x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
        return x_dist,n
    n,bins,patches=plt.hist(E,bins=logbins,orientation="horizontal")
    plt.yscale("log")
    x_dist=np.zeros(len(logbins)-1)
    for i in range(len(x_dist)):
        x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
    return x_dist,n


def plot_E1_E2(d,plot_space,num_bins=100,label_pos_left=True,E1_min=0.04,E2_min=0.4,E_max=20):
    E1=rd.extract_parameter(d, 3)/1000
    E2=rd.extract_parameter(d, 1)/1000

        
    logbins1 = np.geomspace(E1_min, E_max, num_bins)
    logbins2 = np.geomspace(E2_min, E_max, num_bins)
    
    counts, _, _ = np.histogram2d(E1, E2, bins=(logbins1, logbins2))

    pcmap=plot_space.pcolormesh(logbins1, logbins2, counts.T)
    plot_space.plot()
    plot_space.set_xscale('log')
    plot_space.set_yscale('log')
    


    mult=10
    angs=(1,10,20,30,40,50)
    ang_lines=np.zeros((len(angs),num_bins*mult))
    
    logbins1a = np.geomspace(E1_min, E_max, num_bins*mult)
    logbins2a = np.geomspace(E2_min, E_max, num_bins*mult)
    
    for ang in range(len(angs)):
        for b in range(num_bins*mult):
            e_sca=logbins2a[b]
            wl_sca=6.625e-34*3e8/(1.602e-19*float(e_sca)*1e6)
            wl_ini=wl_sca-2.426e-12*(1-np.cos(angs[ang]*np.pi/180))
            e_ini=6.625e-34*3e8/(1.602e-19*wl_ini)*1e-6
            ang_lines[ang,b]=e_ini-e_sca if e_ini-e_sca>0 else float("NaN")

    if label_pos_left:
        for ang in range(len(angs)):
            plot_space.plot(ang_lines[ang],logbins2a,color="C6",lw=1)
            for b in range(num_bins*mult):
                if ang_lines[ang,b]>E1.min():
                    plt.text(ang_lines[ang,b],logbins2a[b],str(angs[ang])+"$^\circ$",color="bisque")
                    break
    
    else:
        for ang in range(len(angs)):
            plot_space.plot(ang_lines[ang],logbins2a,color="C6",lw=1)
            for b in range(num_bins*mult):
                if ang_lines[ang,b]>14 or logbins2a[b]>E_max:
                    plt.text(ang_lines[ang,b-1],logbins2a[b-1]/1.15,str(angs[ang])+"$^\circ$",color="bisque")
                    break
    equi_ener=[0.75,1.5,3,6,12]   
    for ene in equi_ener:
        plot_space.plot(ene-logbins2a,logbins2a,color="C1",lw=1.0)
        plt.text(ene-E2_min,E2_min*1.05,str(ene)+"MeV",color="lightyellow")
    
    
    plt.xlim(E1.min(),np.amax(E1))
    plt.ylim(E2.min(),np.amax(E2))
    
    return pcmap


def plot_ET(d,return_values=False,num_bins=100):
    E=rd.extract_parameter(d, 1)/1000+rd.extract_parameter(d, 3)/1000
    logbins = np.geomspace(0.4, 40, num_bins)
    if return_values:
        n,bins=np.histogram(E,bins=logbins)
        x_dist=np.zeros(len(logbins)-1)
        for i in range(len(x_dist)):
            x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
        return x_dist,n
    n,bins,patches=plt.hist(E,bins=logbins)
    plt.xscale("log")
    x_dist=np.zeros(len(logbins)-1)
    for i in range(len(x_dist)):
        x_dist[i]=np.exp( (np.log(bins[i+1])+np.log(bins[i]))/2 )
    return x_dist,n


def plot_ToF_sum(remove_fast=True,return_values=False,num_bins=50,lims=(0,8)):
    sim_d,sim_t=(raw_sim_data,sim_times) if not remove_fast else (raw_sim_data[:-1],sim_times[:-1])
    x=np.linspace(lims[0],lims[1],num_bins)
    x_dist=x[:-1]+(x[1]-x[0])/2
    n=[np.zeros(len(x)-1) for i in range(len(sim_d))]
    for i in range(len(n)):
        n[i]=( np.histogram( rd.extract_parameter(sim_d[i], 17)*1e9,bins=x)[0]/sim_times[i]*np.min(sim_t) )
    ns=np.zeros(len(n[0]))
    for i in n:
        ns+=i
    if return_values:
        return x,n
    n2,bins,patches=plt.hist(x_dist,bins=x,weights=ns)
    return x,n2

def plot_flight_angle_sum(remove_fast=True,return_values=False,num_bins=50,lims=(0,0.7)):
    sim_d,sim_t=(raw_sim_data,sim_times) if not remove_fast else (raw_sim_data[:-1],sim_times[:-1])
    x=np.linspace(lims[0],lims[1],num_bins)
    x_dist=x[:-1]+(x[1]-x[0])/2
    n=[np.zeros(len(x)-1) for i in range(len(sim_d))]
    for i in range(len(n)):
        n[i]=( np.histogram( rd.calculate_angles(sim_d[i]),bins=x)[0]/sim_times[i]*np.min(sim_t) )
    ns=np.zeros(len(n[0]))
    for i in n:
        ns+=i
    if return_values:
        return x,n
    n2,bins,patches=plt.hist(x_dist,bins=x,weights=ns)
    return x,n2

def plot_AoI_sum(remove_fast=True,return_values=False,num_bins=50,lims=(-0.5,0.5)):
    sim_d,sim_t=(raw_sim_data,sim_times) if not remove_fast else (raw_sim_data[:-1],sim_times[:-1])
    x=np.linspace(lims[0],lims[1],num_bins)
    x_dist=x[:-1]+(x[1]-x[0])/2
    n=[np.zeros(len(x)-1) for i in range(len(sim_d))]
    for i in range(len(n)):
        d=sim_d[i]
        E_e=rd.extract_parameter(d, 3)
        E_g=rd.extract_parameter(d, 1)
        c=2.998e8
        m=511/c**2
        a=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
        tra_a=np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])*np.sign(d[:,11]-d[:,5])
        or_a=np.zeros(len(tra_a))
        for j in range(len(or_a)):
            if abs(tra_a[j]-a[j])<abs(tra_a[j]+a[j]):
                or_a[j]=tra_a[j]-a[j]
            else:
                or_a[j]=tra_a[j]+a[j]

        n[i]=( np.histogram( or_a,bins=x)/sim_times[i]*np.min(sim_t) )[0]
    ns=np.zeros(len(n[0]))
    for i in n:
        ns+=i
    if return_values:
        return x,n
    n2,bins,patches=plt.hist(x_dist,bins=x,weights=ns)
    return x,n2

def plot_ARM_sum(remove_fast=True,return_values=False,num_bins=50,lims=(-0.5,0.75)):
    sim_d,sim_t=(raw_sim_data,sim_times) if not remove_fast else (raw_sim_data[:-1],sim_times[:-1])
    x=np.linspace(lims[0],lims[1],num_bins)
    x_dist=x[:-1]+(x[1]-x[0])/2
    n=[np.zeros(len(x)-1) for i in range(len(sim_d))]
    for i in range(len(n)):
        d=sim_d[i]
        E_e=d[:,3]
        E_g=d[:,1]
        c=2.998e8
        m=511/c**2
        phibar=np.arccos( 1 - (E_e/E_g*m*c**2/(E_e+E_g)) )
        phigeo=np.arccos( -( -d[:,9]+d[:,15] ) / d[:,24])
        ARM=phibar-phigeo

        n[i]=( np.histogram( ARM,bins=x)[0]/sim_times[i]*np.min(sim_t) )
    ns=np.zeros(len(n[0]))
    for i in n:
        ns+=i
    if return_values:
        return x,n
    n2,bins,patches=plt.hist(x_dist,bins=x,weights=ns)
    return x,n2

def plot_E1_sum(remove_fast=True,return_values=False,num_bins=50,lims=(0.04,30)):
    sim_d,sim_t=(raw_sim_data,sim_times) if not remove_fast else (raw_sim_data[:-1],sim_times[:-1])
    x=np.geomspace(lims[0], lims[1], num_bins)
    n=[np.zeros(len(x)-1) for i in range(len(sim_d))]
    for i in range(len(n)):
        d=sim_d[i]
        d=rd.extract_parameter(d, 3)/1000
        n[i]=( np.histogram( d,bins=x)[0]/sim_times[i]*np.min(sim_t) )
    ns=np.zeros(len(n[0]))
    for i in n:
        ns+=i
    if return_values:
        return x,n
    n2,bins,patches=plt.hist(x[:-1],bins=x,weights=ns)
    plt.xscale("log")
    return x,n2

def plot_E2_sum(remove_fast=True,return_values=False,num_bins=50,lims=(0.4,30)):
    sim_d,sim_t=(raw_sim_data,sim_times) if not remove_fast else (raw_sim_data[:-1],sim_times[:-1])
    x=np.geomspace(lims[0], lims[1], num_bins)
    n=[np.zeros(len(x)-1) for i in range(len(sim_d))]
    for i in range(len(n)):
        d=sim_d[i]
        d=rd.extract_parameter(d, 1)/1000
        n[i]=( np.histogram( d,bins=x)[0]/sim_times[i]*np.min(sim_t) )
    ns=np.zeros(len(n[0]))
    for i in n:
        ns+=i
    if return_values:
        return x,n
    n2,bins,patches=plt.hist(x[:-1],bins=x,weights=ns)
    plt.xscale("log")
    return x,n2

def plot_ET_sum(remove_fast=True,return_values=False,num_bins=50,lims=(0.4,40)):
    sim_d,sim_t=(raw_sim_data,sim_times) if not remove_fast else (raw_sim_data[:-1],sim_times[:-1])
    x=np.geomspace(lims[0], lims[1], num_bins)
    n=[np.zeros(len(x)-1) for i in range(len(sim_d))]
    for i in range(len(n)):
        d=sim_d[i]
        d=rd.extract_parameter(d, 1)/1000+rd.extract_parameter(d, 3)/1000
        n[i]=( np.histogram( d,bins=x)[0]/sim_times[i]*np.min(sim_t) )
    ns=np.zeros(len(n[0]))
    for i in n:
        ns+=i
    if return_values:
        return x,n
    n2,bins,patches=plt.hist(x[:-1],bins=x,weights=ns)
    plt.xscale("log")
    return x,n2





def cut_converter(array):
    output=np.zeros((len(array),5))
    for i in range(len(array)):
        output[i,0]=array[i,17]
        output[i,1]=np.log(array[i,3])
        output[i,2]=np.log(array[i,1])
        output[i,3]=np.log(array[i,3]+array[i,1])
        output[i,4]=np.arccos( (array[i,9]-array[i,15]) / array[i,24] )
    return output








def parameter_cut(num_bins=(1,1,1,1,1),soft=True,w_fast=False,fit=True):
    ranges=((4e-9,7e-9),
            (np.log(70),np.log(20000)),
            (np.log(650),np.log(30000)),
            (np.log(900),np.log(30000)),
            (0*np.pi/180,40*np.pi/180))
    
    od=load_od()[2]
    odc=cut_converter(od)
    sim_d,sim_t=(raw_sim_data,sim_times) if w_fast else (raw_sim_data[:-1],sim_times[:-1])
    t_min=np.min(sim_t)
    S,bins=np.histogramdd( cut_converter(sim_d[0]) , bins=num_bins , range=ranges)
    S=S/sim_t[0]*t_min
    if soft:
        B=np.zeros(S.shape)
        for i in range(1,len(sim_d)):
            B+=np.histogramdd( cut_converter(sim_d[i]) , bins=num_bins , range=ranges)[0]/sim_t[i]*t_min
        T=S/(S+B)
        T=np.nan_to_num(T)
        T=T/np.amax(T)
        output=np.empty(od.shape)
        output[:,:]=np.NaN
        
        for row in range(len(od)):
            if (odc[row,0]<ranges[0][0] or odc[row,0]>=ranges[0][1]
                or odc[row,1]<ranges[1][0] or odc[row,1]>=ranges[1][1]
                or odc[row,2]<ranges[2][0] or odc[row,2]>=ranges[2][1]
                or odc[row,3]<ranges[3][0] or odc[row,3]>=ranges[3][1]
                or odc[row,4]<ranges[4][0] or odc[row,4]>=ranges[4][1]):
                #print(odc[row,:])
                continue
            for i in range(len(bins[0])-1):
                if odc[row,0]>=bins[0][i] and odc[row,0]<bins[0][i+1]:
                    p1=i
            for i in range(len(bins[1])-1):
                if odc[row,1]>=bins[1][i] and odc[row,1]<bins[1][i+1]:
                    p2=i
            for i in range(len(bins[2])-1):
                if odc[row,2]>=bins[2][i] and odc[row,2]<bins[2][i+1]:
                    p3=i
            for i in range(len(bins[3])-1):
                if odc[row,3]>=bins[3][i] and odc[row,3]<bins[3][i+1]:
                    p4=i
            for i in range(len(bins[4])-1):
                if odc[row,4]>=bins[4][i] and odc[row,4]<bins[4][i+1]:
                    p5=i
            #print(row)
            if np.random.rand()<=T[p1,p2,p3,p4,p5]:
                output[row,:]=od[row,:]
    
    
    else:
        odm=np.histogramdd( odc , bins=num_bins , range=ranges)[0]
        S=S/np.amax(S)
        odm=odm/np.amax(odm)
        
        if fit:
            
            a=minimize(rd.fit_func,1,(odm,S)).x
            print(a)
            T=S*a
            
            output=np.empty(od.shape)
            output[:,:]=np.NaN
        
            for row in range(len(od)):
                if (odc[row,0]<ranges[0][0] or odc[row,0]>=ranges[0][1]
                    or odc[row,1]<ranges[1][0] or odc[row,1]>=ranges[1][1]
                    or odc[row,2]<ranges[2][0] or odc[row,2]>=ranges[2][1]
                    or odc[row,3]<ranges[3][0] or odc[row,3]>=ranges[3][1]
                    or odc[row,4]<ranges[4][0] or odc[row,4]>=ranges[4][1]):
                    #print(odc[row,:])
                    continue
                for i in range(len(bins[0])-1):
                    if odc[row,0]>=bins[0][i] and odc[row,0]<bins[0][i+1]:
                        p1=i
                for i in range(len(bins[1])-1):
                    if odc[row,1]>=bins[1][i] and odc[row,1]<bins[1][i+1]:
                        p2=i
                for i in range(len(bins[2])-1):
                    if odc[row,2]>=bins[2][i] and odc[row,2]<bins[2][i+1]:
                        p3=i
                for i in range(len(bins[3])-1):
                    if odc[row,3]>=bins[3][i] and odc[row,3]<bins[3][i+1]:
                        p4=i
                for i in range(len(bins[4])-1):
                    if odc[row,4]>=bins[4][i] and odc[row,4]<bins[4][i+1]:
                        p5=i
                #print(row)
                if  T[p1,p2,p3,p4,p5] > odm[p1,p2,p3,p4,p5] or np.random.rand() <= T[p1,p2,p3,p4,p5] / odm[p1,p2,p3,p4,p5] :
                    output[row,:]=od[row,:]
            
        else:
            a=np.min(odm/S)
            print(a)
            T=S*a
            
            output=np.empty(od.shape)
            output[:,:]=np.NaN
        
            for row in range(len(od)):
                if (odc[row,0]<ranges[0][0] or odc[row,0]>=ranges[0][1]
                    or odc[row,1]<ranges[1][0] or odc[row,1]>=ranges[1][1]
                    or odc[row,2]<ranges[2][0] or odc[row,2]>=ranges[2][1]
                    or odc[row,3]<ranges[3][0] or odc[row,3]>=ranges[3][1]
                    or odc[row,4]<ranges[4][0] or odc[row,4]>=ranges[4][1]):
                    #print(odc[row,:])
                    continue
                for i in range(len(bins[0])-1):
                    if odc[row,0]>=bins[0][i] and odc[row,0]<bins[0][i+1]:
                        p1=i
                for i in range(len(bins[1])-1):
                    if odc[row,1]>=bins[1][i] and odc[row,1]<bins[1][i+1]:
                        p2=i
                for i in range(len(bins[2])-1):
                    if odc[row,2]>=bins[2][i] and odc[row,2]<bins[2][i+1]:
                        p3=i
                for i in range(len(bins[3])-1):
                    if odc[row,3]>=bins[3][i] and odc[row,3]<bins[3][i+1]:
                        p4=i
                for i in range(len(bins[4])-1):
                    if odc[row,4]>=bins[4][i] and odc[row,4]<bins[4][i+1]:
                        p5=i
                #print(row)
                if  T[p1,p2,p3,p4,p5] > odm[p1,p2,p3,p4,p5] or np.random.rand() <= T[p1,p2,p3,p4,p5] / odm[p1,p2,p3,p4,p5] :
                    output[row,:]=od[row,:]
    
    
    d=rd.read_h5py_file()
    d=rd.cut_selection(d,9,0,True)
    
    x1,n1=plot_ARM(output)
    
    x2,n2=plot_ARM(d,True)
    n2=(n1[48]+n1[49]+n1[50]+n1[51])/(n2[48]+n2[49]+n2[50]+n2[51])*n2
    plt.plot(x2,n2,label="Crab Simulation",lw=2.0)
    
    x3,n3=plot_ARM(od,True)
    n3=(n1[48]+n1[49]+n1[50]+n1[51])/(n3[48]+n3[49]+n3[50]+n3[51])*n3
    plt.plot(x3,n3,label="Before Cuts",lw=2.0)


































