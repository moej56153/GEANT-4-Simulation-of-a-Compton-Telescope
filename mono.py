import numpy as np
import matplotlib.pyplot as plt
import reading_data as rd
import ang_tof as at
import pos_dis as pd
import att_func as af
from scipy.optimize import curve_fit, minimize
from scipy.integrate import quad

mono_e=1500

def mono_sad_dist(forward,mo_e=mono_e,length=71):
    interval=round(0.7/(length-1),7)
    if forward:
        angles=[interval*i for i in range(length)]
        x_min,x_max,sad=at.fsad_min,at.fsad_max,at.fsad
    else:
        angles=[interval*i+(np.pi-0.7) for i in range(length)]
        x_min,x_max,sad=at.bsad_min,at.bsad_max,at.bsad
    N=np.zeros(length)
    for i in range(length):
        p1=rd.continue_line(angles[i],sad,x_min,x_max)
        p2=at.klein_nishina(mo_e,angles[i])*1e35
        p3=at.energy_limit_factor(mo_e,angles[i],forward)
        #print(i,angles[i])
        N[i]=p1*p2*p3

    return N


def plot_ang_dist_mono(forward,data_array=None,mo_e=mono_e,num_bins=50,size=71):
    d=rd.read_h5py_file()
    d=rd.cut_selection(d,9,0,forward)
    d=data_array if not type(data_array)==type(None) else d
    a=rd.calculate_angles(d)
    n,bins,patches=plt.hist(a,num_bins)
    x_data=bins[:-1]+(bins[1]-bins[0])/2
    y_data=n
    if forward:
        D=mono_sad_dist(forward,mo_e,size)
        x_min=at.fsad_min
        x_max=at.fsad_max
    else:
        D=mono_sad_dist(forward,mo_e,size)
        x_min=at.bsad_min
        x_max=at.bsad_max
    y2=np.zeros(len(x_data))
    for i in range(len(x_data)):
        y2[i]=rd.continue_line(x_data[i],D,x_min,x_max)
    a=minimize(rd.fit_func,1,(y_data,y2))
    rd.plot_points(D*a.x,x_min,x_max,2.0)
    # plt.xlabel("Scattering Angle [rad]")
    # plt.ylabel("Frequency [Counts/Bins]")
    
    
    
#Remove Outliers

def remove_vel_out(forward,margin=2*10**9,p=rd.p,h5=rd.h5_f_n,data=None):
    d=rd.read_h5py_file(p,h5)
    d=rd.cut_selection(d,9,0,forward)
    d=data if not type(data)==type(None) else d
    d=d[abs(d[:,24]/d[:,17]-at.c)<=margin]
    return d
    
def remove_e_out(forward,margin=4000,p=rd.p,h5=rd.h5_f_n,mo_e=mono_e,data=None):
    d=rd.read_h5py_file(p,h5)
    d=rd.cut_selection(d,9,0,forward)
    d=data if not type(data)==type(None) else d
    d=d[abs(d[:,1]+d[:,3]-mo_e)<=margin]
    return d

def remove_sa_out(forward,margin=0.02,p=rd.p,h5=rd.h5_f_n,data=None):
    d=rd.read_h5py_file(p,h5)
    d=rd.cut_selection(d, 9, 0, forward)
    d=data if not type(data)==type(None) else d
    c=2.998e8
    m=511/c**2
    d=d[abs( np.arccos( 1 - (d[:,3]/d[:,1]*m*c**2/(d[:,3]+d[:,1])) ) 
            -np.arccos( -1*(-d[:,9]+d[:,15]) / d[:,24]) )
            <=margin]
    return d



#ToF
c=2.998e10
# FSAD=mono_sad_dist(True)
# BSAD=mono_sad_dist(False)
FSAD_min=0
FSAD_max=0.7
BSAD_min=np.pi-0.7
BSAD_max=np.pi


def plot_ToF_mono(forward,data_array=None,num_bins=50,mo_e=mono_e,vel_cut=False,xlims=None):
    global FSAD,BSAD
    FSAD=mono_sad_dist(True,mo_e)
    BSAD=mono_sad_dist(False,mo_e)
    d=rd.read_h5py_file()
    d=rd.cut_selection(d,9,0,forward)
    d=data_array if not type(data_array)==type(None) else d
    n,bins,patches=plt.hist(d[:,17],num_bins)

    x_data=bins[:-1]+(bins[1]-bins[0])/2
    x_data_g=np.linspace(xlims[0],xlims[1],200)
    
    av_d=calculate_total_distance(mo_e/1000,forward)
    if mo_e/1000>20:
        dis1=calculate_total_distance(18.5,forward)
        dis2=calculate_total_distance(19.5,forward)
        av_d=dis2+(dis2-dis1)/(np.log(19.5)-np.log(18.5))*(np.log(mo_e/1000)-np.log(19.5))
    
    c=2.998e10
    sig_t=3.53553e-10
    sig_D1=rd.D1_depth/2/np.sqrt(3)
    sig_D2=rd.D2_depth/2/np.sqrt(3)
    
    sig=np.sqrt(sig_t**2+(sig_D1/c)**2+(sig_D2/c)**2)
    if forward:
        y_data=forward_ToF_mono(x_data,sig,av_d,1)
        y_data_g=forward_ToF_mono(x_data_g,sig,av_d,1)
    else:
        y_data=backward_ToF_mono(x_data,sig,av_d,1)
        y_data_g=backward_ToF_mono(x_data_g,sig,av_d,1)
    if vel_cut:
        if forward:
            y_data=np.array([forward_ToF_theo_mono(i,av_d) for i in x_data])
            y_data_g=np.array([forward_ToF_theo_mono(i,av_d) for i in x_data_g])
        else:
            y_data=np.array([backward_ToF_theo_mono(i,av_d) for i in x_data])
            y_data_g=np.array([backward_ToF_theo_mono(i,av_d) for i in x_data_g])
        sig=0
    y_data=y_data*np.amax(n)/np.amax(y_data)
    y_data_g=y_data_g*np.amax(n)/np.amax(y_data_g)
    
    a=minimize(rd.fit_func,1,(n,y_data)).x
    
    plt.plot(x_data_g,y_data_g*a,lw=2.0)
    

    return av_d,sig
    
def forward_ToF_mono(t,sigma,distance,amplitude):
    if type(t)==type(np.array([])):
        output=np.array([])
        for i in t:
            output=np.append(output,amplitude*quad(integrand_product_forward_mono,-0.25e-8,0.25e-8,(i,sigma,distance),0,1e-2,1e-2)[0])
        return output
    else:
        return amplitude*quad(integrand_product_forward_mono,-0.25e-8,0.25e-8,(t,sigma,distance),0,1e-2,1e-2)[0]
    
def forward_ToF_theo_mono(t,distance):
    if t>=distance/c:
        return 2*np.pi*distance**3/(t**2*c)*ang_weight_forward_mono(t,distance)
    else:
        return 0

def ang_weight_forward_mono(t,distance):
    global FSAD
    theta=np.arccos(distance/t/c)
    return rd.continue_line(theta, FSAD, FSAD_min, FSAD_max)
    


def integrand_product_forward_mono(tao,t,sigma,distance):
    return forward_ToF_theo_mono(t-tao,distance)*normal_dis_mono(tao,sigma)
    
def normal_dis_mono(t,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.e**(-0.5*(t/sigma)**2)

def backward_ToF_mono(t,sigma,distance,amplitude):
    if type(t)==type(np.array([])):
        output=np.array([])
        for i in t:
            output=np.append(output,amplitude*quad(integrand_product_backward_mono,-0.25e-8,0.25e-8,(i,sigma,distance),0,1e-2,1e-2)[0])
        return output
    else:
        return amplitude*quad(integrand_product_backward_mono,-0.25e-8,0.25e-8,(t,sigma,distance),0,1e-2,1e-2)[0]
    
def backward_ToF_theo_mono(t,distance):
    if t>=distance/c:
        return 2*np.pi*distance**3/(t**2*c)*ang_weight_backward_mono(t,distance)
    else:
        return 0

def ang_weight_backward_mono(t,distance):
    global BSAD
    theta=np.pi-np.arccos(distance/t/c)
    return rd.continue_line(theta, BSAD, BSAD_min, BSAD_max)
    

def integrand_product_backward_mono(tao,t,sigma,distance):
    return backward_ToF_theo_mono(t-tao,distance)*normal_dis_mono(tao,sigma)


def energy_layer_dist(forward,initial,mo_e,e_dist):
    initial=True if initial==3 else False
    sad=mono_sad_dist(forward,mo_e)
    int_dist=np.zeros(len(e_dist))
    if forward:
        sa_min,sa_max=at.fsad_min,at.fsad_max
    else:
        sa_min,sa_max=at.bsad_min,at.bsad_max
    for i in range(len(int_dist)):
        if initial:
            e=mo_e-e_dist[i]
        else:
            e=e_dist[i]
        wl1=6.625e-34*3e8/(1.602e-19*mo_e)*1e-3
        wl2=6.625e-34*3e8/(1.602e-19*(e))*1e-3
        sa=np.arccos( ( (wl1-wl2)/2.426e-12 ) + 1 )
        if np.isnan(sa):
            #print(forward,initial,mo_e,e,sa,type(sa),wl1,wl2)
            continue
        dens_cor=(e)**2*np.sin(sa)
        if dens_cor==0:
            #print(forward,initial,mo_e,e)
            continue
        int_dist[i]= rd.continue_line(sa,sad,sa_min,sa_max) / dens_cor

    return int_dist
            
    
def distance_integrand_1(x,mu,p):
    return x*mu*p*np.e**(-mu*p*x)

def distance_integrand_2(x,mu,p):
    return mu*p*np.e**(-mu*p*x)


def calculate_travel_distance(Ei,forward,D1):
    
    if D1:
        mu,mu_E,p,t=af.NE213,af.E,af.p_NE213,rd.D1_depth
    else:
        mu,mu_E,p,t=af.NaI,af.E,af.p_NaI,rd.D2_depth
    wli=6.625e-34*3e8/(1.602e-19*Ei)*1e-3
    if forward:
        wlmi=wli+2.426e-12*(1-np.cos(0.7))
        wlma=wli+2.426e-12*(1-np.cos(0.0))
    else:
        wlmi=wli+2.426e-12*(1-np.cos(np.pi))
        wlma=wli+2.426e-12*(1-np.cos(np.pi-0.7))
    Emi=6.625e-34*3e8/(1.602e-19*wlmi)*1e-3
    Ema=6.625e-34*3e8/(1.602e-19*wlma)*1e-3
    E_dist=np.geomspace(Emi,Ema,100)
    fe=energy_layer_dist(forward,False,Ei,E_dist)
    s_1=0
    s_2=0
    for E in range(len(E_dist)):
        wla=6.625e-34*3e8/(1.602e-19*E_dist[E])*1e-3
        
        sa=np.arccos( ( (wli-wla)/2.426e-12 ) + 1 )
        if wli>wla:
            sa=0
        if (( (wli-wla)/2.426e-12 ) + 1)<-1:
            sa=np.pi
        th=abs(t/np.cos(sa))

        pos_dis=np.linspace(0,th,50)
        s_pos_1=0
        s_pos_2=0
        for x in pos_dis:
            s_pos_1+=abs(np.cos(sa))*distance_integrand_1(x,rd.continue_linear_set(E_dist[E]/1000,mu_E,mu),p)
            s_pos_2+=distance_integrand_2(x,rd.continue_linear_set(E_dist[E]/1000,mu_E,mu),p)

        if E_dist[E]/1000<mu_E[0] or E_dist[E]/1000>mu_E[-1]:
            s_1+=0
            s_2+=0
            print(Ei,E_dist[E])
        else:
            s_1+=s_pos_1/s_pos_2*fe[E]*E_dist[E]
            s_2+=fe[E]*E_dist[E]

    if s_1==0 and s_2==0:
        return np.arccos(2)
    elif s_2==0:
        print("here")
    return s_1/s_2
        

#########E=np.geomspace(0.5,20,100)
FD2=np.array([       float("NaN"),       float("NaN"),        float("NaN"), 2.10446283, 2.10890279,
       2.13479908, 2.17520697, 2.21426784, 2.25166512, 2.29323136,
       2.32650687, 2.35949041, 2.39592812, 2.42724276, 2.45733857,
       2.48647949, 2.51429539, 2.54108723, 2.56707176, 2.59236796,
       2.61336972, 2.63741303, 2.66064464, 2.683191  , 2.70514246,
       2.72333738, 2.74429991, 2.76432995, 2.78071272, 2.79961073,
       2.81513839, 2.83281617, 2.84918893, 2.86250156, 2.87755422,
       2.88998852, 2.90456998, 2.91605696, 2.92722064, 2.93907686,
       2.94901098, 2.9612536 , 2.9700089 , 2.97835479, 2.98698809,
       2.99480709, 3.00236081, 3.01206277, 3.01906117, 3.02582425,
       3.03209166, 3.0388004 , 3.04433407, 3.04955633, 3.05449556,
       3.05983633, 3.06421216, 3.06830791, 3.07202177, 3.07540925,
       3.08037545, 3.08327335, 3.08592892, 3.08831055, 3.09040496,
       3.09225656, 3.09411766, 3.09560142, 3.09682553, 3.09776463,
       3.09845742, 3.09893385, 3.09922018, 3.09933707, 3.09930472,
       3.09912794, 3.09873425, 3.09812565, 3.09733025, 3.09636525,
       3.09525038, 3.09399304, 3.09254158, 3.09089809, 3.08908472,
       3.08711529, 3.08500405, 3.08276284, 3.08040682, 3.07794383,
       3.07537901, 3.07272391, 3.06997715, 3.06713189, 3.06418469,
       3.06114815, 3.0580265 , 3.05482351, 3.05154704, 3.04820268])


BD1=np.array([       float("NaN"),        float("NaN"),        float("NaN"),        float("NaN"),        float("NaN"),
              float("NaN"),        float("NaN"),        float("NaN"),        float("NaN"), 3.47358439,
       3.47575427, 3.47805948, 3.47877132, 3.48099457, 3.48318264,
       3.48686048, 3.48898257, 3.48957863, 3.49161744, 3.49508062,
       3.49550509, 3.49735706, 3.5006233 , 3.50088564, 3.50402975,
       3.50565094, 3.50721987, 3.50728608, 3.51020559, 3.51018218,
       3.51299884, 3.51432694, 3.51561077, 3.51542309, 3.51662614,
       3.51778928, 3.52033259, 3.51999933, 3.52246243, 3.52206277,
       3.52304247, 3.52398861, 3.52490275, 3.52578554, 3.52803594,
       3.52746138, 3.52825633, 3.52902376, 3.531154  , 3.5304803 ,
       3.53117072, 3.53183731, 3.53248076, 3.53310168, 3.53370078,
       3.53427908, 3.53621192, 3.53674889, 3.53726687, 3.53639618,
       3.53824873, 3.53734635, 3.53916273, 3.53823035, 3.54001297,
       3.53905291, 3.53944249, 3.53981824, 3.54018057, 3.54188851,
       3.54222444, 3.54119175, 3.54286081, 3.54316212, 3.54209869,
       3.54237962, 3.54265047, 3.54291157, 3.54451431, 3.54340597,
       3.54363997, 3.54386559, 3.54408313, 3.54564043, 3.54449491,
       3.54468979, 3.54487759, 3.54640405, 3.54657811, 3.54674574,
       3.5469074 , 3.54706313, 3.54587013, 3.54735798, 3.54615522,
       3.54763192, 3.54642006, 3.5465453 , 3.54800682, 3.54812281])



def calculate_travel_distance_inf(D1,E):
    if D1:
        mu,mu_E,p,t=af.NE213,af.E,af.p_NE213,rd.D1_depth
    else:
        mu,mu_E,p,t=af.NaI,af.E,af.p_NaI,rd.D2_depth
    if E/1000<mu_E[0] or E/1000>mu_E[-1]:
        return np.arccos(2)
    return 1/( rd.continue_linear_set(E/1000,mu_E,mu)*p )


def calculate_total_distance(E,forward):#in MeV
    E_dist1=af.E
    FD1=af.dis_NE213
    BD2=af.dis_NaI
    E_dist2=np.geomspace(0.5,20,100)
    for i in range(len(FD2)):
        if np.isnan(FD2[i]):
            posa=i+1
        if np.isnan(BD1[i]):
            posb=i+1
    pos=posa if forward else posb
    d1=float("NaN")
    d2=float("NaN")
    for i in range(pos,len(E_dist2)-1):
        if E>=E_dist2[i] and E<=E_dist2[i+1]:
            if forward:
                d2=FD2[i]+(FD2[i+1]-FD2[i])/(np.log(E_dist2[i+1])-np.log(E_dist2[i]))*(np.log(E)-np.log(E_dist2[i]))
            else:
                d1=BD1[i]+(BD1[i+1]-BD1[i])/(np.log(E_dist2[i+1])-np.log(E_dist2[i]))*(np.log(E)-np.log(E_dist2[i]))
    for i in range(len(E_dist1)-1):
        if E>=E_dist1[i] and E<=E_dist1[i+1]:
            if forward:
                d1=FD1[i]+(FD1[i+1]-FD1[i])/(np.log(E_dist1[i+1])-np.log(E_dist1[i]))*(np.log(E)-np.log(E_dist1[i]))
            else:
                d2=BD2[i]+(BD2[i+1]-BD2[i])/(np.log(E_dist1[i+1])-np.log(E_dist1[i]))*(np.log(E)-np.log(E_dist1[i]))
    if forward:
        totd=rd.D1_D2_Distance+(rd.D1_depth/2-d1)-(rd.D2_depth/2-d2)
        #totd=rd.D1_D2_Distance+(-d1)-(-d2)
    else:
        totd=rd.D1_D2_Distance-(rd.D1_depth/2-d1)-(rd.D2_depth/2-d2)
        #totd=rd.D1_D2_Distance-(rd.D1_depth-d1)-(-d2)
    return totd
    
        
    






def events_per_cell_mono(forward=True,D1=True,data_array=None,mo_e=None,text=True):
    if forward:
        if D1:
            N=calculate_expected_number_mono(forward,D1,mo_e)
            par1=5
            par2=7
            points=rd.D1_points
        else:
            N=calculate_expected_number_mono(forward,D1,mo_e)
            par1=11
            par2=13
            points=rd.D2_points
    else:
        if D1:
            N=calculate_expected_number_mono(forward,D1,mo_e)
            par1=11
            par2=13
            points=rd.D1_points
        else:
            N=calculate_expected_number_mono(forward,D1,mo_e)
            par1=5
            par2=7
            points=rd.D2_points
    rd.plot_histogram_2D(data_array, par1, par2)
    s=0
    for i in N:
        s+=i
    N=N*len(data_array)/s
    if text:
        for i in range(len(points)):
            plt.text(points[i,0],points[i,1],"M="+str(len(rd.select_cell(data_array, i,forward,D1)))+"\n E="+str(int(N[i])),ha="center",
                     va="center",bbox=dict(facecolor='white', edgecolor='none'))

def cxn_mono(sad,forward,D1,mo_e,length=211):
    interval=round(0.7/(length-1),7)
    if forward:
        angles=[interval*i for i in range(length)]
        x_min,x_max=at.fsad_min,at.fsad_max
        if not D1:
            sad=sad[::-1]
    else:
        angles=[interval*i+(np.pi-0.7) for i in range(length)]
        x_min,x_max=at.bsad_min,at.bsad_max
        if D1:
            sad=sad[::-1]
    N=np.zeros(length)
    for i in range(length):
        #print(i,angles[i])
        N[i]=integrand_mono(mo_e,angles[i],sad,forward,x_min,x_max)
    return N

def integrand_mono(E,theta_S,sad,forward,x_min,x_max):

    p1=rd.continue_line(theta_S,sad,x_min,x_max)
    p2=at.klein_nishina(E,theta_S)
    p3=at.energy_limit_factor(E,theta_S,forward)*1e35
    #print(p1*p2*p3*p4,p1,p2,p3,p4,E,theta_S)
    return p1*p2*p3 

def calculate_expected_number_mono(forward,D1,mo_e):
    if D1:
        D=np.array([cxn_mono(pd.D1_0,forward,D1,mo_e),cxn_mono(pd.D1_1,forward,D1,mo_e),cxn_mono(pd.D1_2,forward,D1,mo_e),
                    cxn_mono(pd.D1_3,forward,D1,mo_e),cxn_mono(pd.D1_4,forward,D1,mo_e),
                    cxn_mono(pd.D1_5,forward,D1,mo_e),cxn_mono(pd.D1_6,forward,D1,mo_e)])
    else:
        D=np.array([cxn_mono(pd.D2_0,forward,D1,mo_e),cxn_mono(pd.D2_1,forward,D1,mo_e),cxn_mono(pd.D2_2,forward,D1,mo_e),
                    cxn_mono(pd.D2_3,forward,D1,mo_e),cxn_mono(pd.D2_4,forward,D1,mo_e),cxn_mono(pd.D2_5,forward,D1,mo_e),
                    cxn_mono(pd.D2_6,forward,D1,mo_e),cxn_mono(pd.D2_7,forward,D1,mo_e),cxn_mono(pd.D2_8,forward,D1,mo_e),
                    cxn_mono(pd.D2_9,forward,D1,mo_e),cxn_mono(pd.D2_10,forward,D1,mo_e),
                    cxn_mono(pd.D2_11,forward,D1,mo_e),cxn_mono(pd.D2_12,forward,D1,mo_e),cxn_mono(pd.D2_13,forward,D1,mo_e)])
    N=np.zeros(len(D))
    for cell in range(len(D)):
        s=0
        for i in range(len(D[cell])):
            s+=D[cell,i]
        N[cell]=s
    return N











    
    
    
    
    

