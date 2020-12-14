import numpy as np
import matplotlib.pyplot as plt
import reading_data as rd
from scipy.integrate import quad,dblquad
from scipy.optimize import curve_fit, minimize


#Funcitons and Plotting
def plot_function(function_name,x_min,x_max, points=71):
    x=np.arange(x_min,x_max,(x_max-x_min)/points)
    y=np.array([function_name(i) for i in x])
    plt.plot(x,y)

def plot_points(data_points,x_min,x_max,col=None,linwi=None):
    increment=(x_max-x_min)/(data_points.size-1)
    x=np.arange(x_min,x_max+increment,increment) 
    if x.size>data_points.size:
        x=x[:-1]
    if type(col)==type(None):
        if type(linwi)==type(None):
            plt.plot(x,data_points)
        else:
            plt.plot(x,data_points,lw=linwi)
    else:
        if type(linwi)==type(None):
            plt.plot(x,data_points,color=col)
        else:
            plt.plot(x,data_points,color=col,lw=linwi)



def plot_histogram_and_distribution(data_array,distribution_points,x_min,x_max,num_bins=100):
    n,bins,patches=plt.hist(data_array,num_bins)
    x_data=bins[:-1]+(bins[1]-bins[0])/2
    y_data=n
    y2_data=np.zeros(len(x_data))
    for i in range(len(x_data)):
        y2_data[i]=rd.continue_line(x_data[i],distribution_points,x_min,x_max)
    a=minimize(rd.fit_func,1,(y_data,y2_data)).x

    scaled_distribution=distribution_points*a
    plot_points(scaled_distribution, x_min, x_max,"C1",2.0)
    #plt.xlabel("Scattering Angle [rad]")
    #plt.ylabel("Frequency [Counts/Bin]")
    return n,bins

def plot_histogram_and_optimized_distribution(num_bins=100,forward=True):
    d=rd.read_h5py_file()
    d=rd.cut_selection(d,9,0,forward)
    data_array=rd.calculate_angles(d)
    if forward:
        distribution_points,x_min,x_max=fsad,fsad_min,fsad_max
    else:
        distribution_points,x_min,x_max=bsad,bsad_min,bsad_max
    n,bins=plot_histogram_and_distribution(data_array, distribution_points, x_min, x_max,num_bins)
    x_data=bins[:-1]+(bins[1]-bins[0])/2
    y_data=n
    
    if forward:
        popt,pcov=curve_fit(fsad_correction, x_data,y_data)
        #plt.plot(x_data,fsad_correction(x_data,*popt),color="C8",lw=3.0)
    else:
        popt,pcov=curve_fit(bsad_correction, x_data,y_data)
        #plt.plot(x_data,bsad_correction(x_data,*popt),color="C8",lw=3.0)
    #plt.xlabel("Scattering Angle [rad]")
    #plt.ylabel("Frequency [Counts/Bin]")
    return popt,pcov

def fsad_correction(x,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14):
    x_min=fsad_min
    x_max=fsad_max
    correction= continue_line(x, np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14]), x_min, x_max)
    return continue_line(x,fsad,x_min,x_max) * correction

def bsad_correction(x,a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14):
    x_min=bsad_min
    x_max=bsad_max
    correction= continue_line(x, np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14]), x_min, x_max)
    return continue_line(x,bsad,x_min,x_max) * correction
    
def save_line(x_min,x_max,function_name,data_points=100):
    data=np.array([])
    for i in range(data_points):
        #(i,(x_max-x_min)/(data_points-1)*i+x_min)
        data=np.append(data,[function_name((x_max-x_min)/(data_points-1)*i+x_min)])
    return data

def continue_line(value, data,x_min,x_max):
    data_points=data.size
    increment=round((x_max-x_min)/(data_points-1),sig_dec)
    if type(value)==type(np.array([])):
        output=np.array([])
        for i in value:
            if i>x_max or i<x_min or np.isnan(i):
                output=np.append(output,0)
            else:
                index=int((i-x_min)//increment)
                progress=(i-x_min)%increment
                output=np.append(output,data[index]+(data[index+1]-data[index])*progress/increment)
        return output
    else:
        if value>x_max or value<x_min or np.isnan(value):
            return 0
        else:
            index=int((value-x_min)//increment)
            progress=(value-x_min)%increment
            return data[index]+(data[index+1]-data[index])*progress/increment


#Detector Details  
D1_points=np.array([[0,0,102.35],
                    [26,39.1,102.35],
                    [-26,39.1,102.35],
                    [26,-39.1,102.35],
                    [-26,-39.1,102.35],
                    [42.3,0,102.35],
                    [-42.3,0,102.35]
                    ])
D1_radius=14
D1_depth=8.5
D2_points=np.array([[0,41.254,-55.65],
                    [0,-41.254,-55.65],
                    [30.2,41.254,-55.65],
                    [-30.2,41.254,-55.65],
                    [30.2,-41.254,-55.65],
                    [-30.2,-41.254,-55.65],
                    [45.3,15.1,-55.65],
                    [-45.3,15.1,-55.65],
                    [45.3,-15.1,-55.65],
                    [-45.3,-15.1,-55.65],
                    [15.1,15.1,-55.65],
                    [-15.1,15.1,-55.65],
                    [15.1,-15.1,-55.65],
                    [-15.1,-15.1,-55.65]
                    ])
D2_radius=14.1
D2_depth=7.5
#D1_D2_Distance=158
c=2.998e10


#Scattering Angle Distribution
def point_circle_distribution(displacement, displacement_angle, theta_s, center1=D1_points[0], center2=D2_points[13], radius2=D2_radius):
    position=center1+np.array([displacement*np.sin(displacement_angle),displacement*np.cos(displacement_angle),0])    
    d=np.sqrt((position-center2).dot(position-center2))    
    alpha=np.arctan( np.sqrt(  (position[0]-center2[0])**2 + (position[1]-center2[1])**2 )  /abs(position[2]-center2[2]))
    if position[2]<center2[2]:
        alpha=np.pi-alpha
    theta=np.arctan(radius2/d)
    z_h=d*np.cos(theta_s)
    z_1=d*np.cos(alpha+theta)
    z_2=d*np.cos(alpha-theta)
    if alpha-theta<=0 and theta_s<=theta-alpha:
        return np.sin(theta_s)*2*np.pi*displacement
    elif alpha+theta>=np.pi and np.pi-theta_s<=alpha+theta-np.pi:
        return (np.sin(theta_s))*2*np.pi*displacement
    elif (z_h<=z_1 and z_h<=z_2) or (z_h>=z_1 and z_h>=z_2):
        return 0
    else:
        y_1=d*np.sin(alpha+theta)
        y_2=d*np.sin(alpha-theta)
        c=np.array([0.0,(y_1+y_2)/2,(z_1+z_2)/2])
        d_y=(y_1-y_2)/2
        d_z=(z_1-z_2)/2
        r=(d_y**2+d_z**2)**0.5
        phi=np.arccos((z_h-(z_1+z_2)/2)/(r*np.sin(alpha)))
        p_x=c[0]+r*np.sin(phi)
        p_y=c[1]-r*np.cos(phi)*np.cos(alpha)
        beta=np.arccos((p_y)/(p_x**2+p_y**2)**0.5)
        if theta_s<np.pi/2:
            return 2*np.sin(theta_s)*beta *displacement
        else:
            return 2*(np.sin(theta_s))*beta*displacement
    
def circle_circle_distribution(theta_s,center1=D1_points[1],radius1=D1_radius,center2=D2_points[5], radius2=D2_radius):
    return dblquad(point_circle_distribution,0,2*np.pi,0,D1_radius,(theta_s,center1,center2,radius2),0.5e-1,0.5e-1)

def circle_layer_distribution(theta_s,center1=D2_points[13], radius1=D2_radius, points2=D1_points,radius2=D1_radius):
    s=0
    e=0
    for point2 in points2:
        t=circle_circle_distribution(theta_s,center1,radius1,point2,radius2)
        s+=t[0]
        e+=t[1]
    return s#,e

def layer_layer_distribution(theta_s,D1_to_D2=True):
    s=0
    #e=0
    if D1_to_D2:
        for point1 in D1_points:
            t=circle_layer_distribution(theta_s,point1,D1_radius,D2_points,D2_radius)
            s+=t#[0]
            #e+=t[1]
    else:
        for point1 in D2_points:
            t=circle_layer_distribution(theta_s,point1,D2_radius,D1_points,D1_radius)
            s+=t#[0]
            #e+=t[1]
    return s#,e
        
def ang_dist_conv_norm(theta_s):
    return quad(integrand_ang,-ang_unc*5,ang_unc*5,(theta_s,),0,1e-2,1e-2)[0]

def integrand_ang(gamma,theta_s):
    return continue_line(theta_s-gamma, fsad, fsad_min, fsad_max)*normal_dis(gamma,ang_unc)

ang_unc=4/150

def plot_ang_all_f(data_array,num_bins=100):
    n,bins,patches=plt.hist(data_array,num_bins,facecolor="blue")
    scale1=np.amax(n)
    scale2=np.amax(fsad)
    plot_points(fsad*scale1/scale2, fsad_min, fsad_max)
    ang_unc=save_line(fsad_min, fsad_max, ang_dist_conv_norm)
    scale3=np.amax(ang_unc)
    plot_points(ang_unc*scale1/scale3, fsad_min, fsad_max)

#Time of Flight Distribution


def optimize_ToF_parameters(forward=True, num_bins=100,data=None):
    d=rd.read_h5py_file()
    d=data if not type(data)==type(None) else d
    d=rd.slice_selection(d,17,3.5e-9,9e-9)
    if forward:
        
        df=rd.cut_selection(d, 9, 0)
        tf=rd.extract_parameter(df, 17)
        n,bins,patches=plt.hist(tf,num_bins)
        x=bins
        x_data=bins[:-1]+(bins[1]-bins[0])/2
        y_data=n
        popt,pcov=curve_fit(forward_ToF, x_data,y_data,(0.4e-9,150,1e-13))
        y=np.array([forward_ToF(i,popt[0],popt[1],popt[2]) for i in x])
        plt.plot(x,y,lw=2.0)
         
        
    else:
        db=rd.cut_selection(d, 9, 0, False)
        tb=rd.extract_parameter(db, 17)
        n,bins,patches=plt.hist(tb,num_bins)
        x=bins
        x_data=bins[:-1]+(bins[1]-bins[0])/2
        y_data=n
        popt,pcov=curve_fit(backward_ToF, x_data,y_data,(0.4e-9,150,1e-13))
        y=np.array([backward_ToF(i,popt[0],popt[1],popt[2]) for i in x])
        plt.plot(x,y,lw=2.0)
        
    # plt.xlabel("Time of Flight [s]")
    # plt.ylabel("Frequency [Counts/Bin]")
    return popt,pcov
    

def forward_ToF_theo(t,distance):
    if t>=distance/c:
        return 2*np.pi*distance**3/(t**2*c)*ang_weight_forward(t,distance)
    else:
        return 0

def ang_weight_forward(t,distance):
    theta=np.arccos(distance/t/c)
    return continue_line(theta, fsad, fsad_min, fsad_max)*continue_line(theta, pf, fsad_min, fsad_max)


def forward_ToF(t,sigma,distance,amplitude):
    if type(t)==type(np.array([])):
        output=np.array([])
        for i in t:
            output=np.append(output,amplitude*quad(integrand_product,-0.25e-8,0.25e-8,(i,sigma,distance),0,1e-2,1e-2)[0])
        return output
    else:
        return amplitude*quad(integrand_product,-0.25e-8,0.25e-8,(t,sigma,distance),0,1e-2,1e-2)[0]

def normal_dis(t,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.e**(-0.5*(t/sigma)**2)

def integrand_product(tao,t,sigma,distance):
    return forward_ToF_theo(t-tao,distance)*normal_dis(tao,sigma)

#Backward ToF
def backward_ToF_theo(t,distance):
    if t>=distance/c:
        return 2*np.pi*distance**3/(t**2*c)*ang_weight_backward(t,distance)
    else:
        return 0

def ang_weight_backward(t,distance):
    theta=np.pi-np.arccos(distance/t/c)
    return continue_line(theta, bsad, bsad_min, bsad_max)*continue_line(theta, pb, bsad_min, bsad_max)

def integrand_product_b(tao,t,sigma,distance):
    return backward_ToF_theo(t-tao,distance)*normal_dis(tao,sigma)

def backward_ToF(t,sigma,distance,amplitude):
    if type(t)==type(np.array([])):
        output=np.array([])
        for i in t:
            output=np.append(output,amplitude*quad(integrand_product_b,-0.25e-8,0.25e-8,(i,sigma,distance),0,1e-2,1e-2)[0])
        return output
    else:
        return amplitude*quad(integrand_product_b,-0.25e-8,0.25e-8,(t,sigma,distance),0,1e-2,1e-2)[0]

#Diff. cross sec.
E_min=700           #keV
E_max=50000
E_power=2.17
def energy_distribution(E):
    if E>=E_min and E<=E_max:
        return E**-E_power
    else:
        return 0

def klein_nishina(E,theta_S):
    alpha=1/137
    c=2.998e8
    m=511/c**2

    h_bar=1.055e-34*(6.242e15)
    r=h_bar/(m*c)
    P=1/ ( 1 + (E/(m*c**2)) * (1-np.cos(theta_S)) )
    return 1/2*alpha**2*r**2*P**2* ( P + 1/P - np.sin(theta_S)**2 )

def integrand_energy_f(E,theta_S):
    E_e_l,E_g_l=50,500

    
    c=2.998e8
    m=511/c**2

    E_g=E/(1+E/(m*c**2)*(1-np.cos(theta_S)))
    E_e=E*(1-1/(1+E/(m*c**2)*(1-np.cos(theta_S))))
    if E_e>E_e_l and E_g>E_g_l:
        return energy_distribution(E)*klein_nishina(E,theta_S)
    else:
        return 0

def diff_cross_sec_f(theta_S):
    #Change error between e-30 and e-40
    return quad(integrand_energy_f,E_min,E_max,(theta_S,),0,1e-40,1e-40)[0]

def integrand_energy_b(E,theta_S):
    E_e_l,E_g_l=500,50

    
    c=2.998e8
    m=511/c**2

    E_g=E/(1+E/(m*c**2)*(1-np.cos(theta_S)))
    E_e=E*(1-1/(1+E/(m*c**2)*(1-np.cos(theta_S))))
    if E_e>E_e_l and E_g>E_g_l:
        return energy_distribution(E)*klein_nishina(E,theta_S)
    else:
        return 0

def diff_cross_sec_b(theta_S):
    #Change error between e-30 and e-40
    return quad(integrand_energy_b,E_min,E_max,(theta_S,),0,1e-30,1e-30)[0]

def plot_diff_cross_sec(forward=True):
    pfn=pf[:-1]
    pbn=pb[1:]
    xf=np.linspace(fsad_min,fsad_max,15)[:-1]
    xb=np.linspace(bsad_min,bsad_max,15)[1:]
    
    if forward:
        y2_data=np.zeros(len(xf))
        for i in range(len(xf)):
            y2_data[i]=diff_cross_sec_f(xf[i])*1e33
        f_scale=minimize(rd.fit_func,1,(y2_data[4:],pfn[4:])).x/1e33
        pfe=np.array([])
        for i in range(len(pfcov)):
            pfe=np.append(pfe, np.sqrt(pfcov[i,i]))
        pfe=pfe[:-1]
        plot_function(diff_cross_sec_f,fsad_min,fsad_max,100)
        rd.plot_points_error(pfn*f_scale, pfe*f_scale, xf[0], xf[-1])
    else:
        y2_data=np.zeros(len(xb))
        for i in range(len(xb)):
            y2_data[i]=diff_cross_sec_b(xb[i])*1e34
        b_scale=minimize(rd.fit_func,1,(y2_data,pbn)).x/1e34
        pbe=np.array([])
        for i in range(len(pfcov)):
            pbe=np.append(pbe, np.sqrt(pbcov[i,i]))
        pbe=pbe[1:]
        plot_function(diff_cross_sec_b,bsad_min,bsad_max,100)
       
        rd.plot_points_error(pbn*b_scale, pbe*b_scale, xb[0], xb[-1])
    #plt.xlabel("Scattering Angle [rad]")
    #plt.ylabel("Differential Cross Section Proportionality [arb. unit]")


def energy_limit_factor(E,theta_S,forward=True):
    c=2.998e8
    m=511/c**2

    E_g=E/(1+E/(m*c**2)*(1-np.cos(theta_S)))
    E_e=E*(1-1/(1+E/(m*c**2)*(1-np.cos(theta_S))))
    if forward:
        if E_e>50 and E_g>500:
            return 1
        else:
            return 0
    else:
        if E_e>500 and E_g>50:
            return 1
        else:
            return 0


sig_dec=3





fsad_min=0
fsad_max=0.7


fsad=np.array([   0.        ,  201.43163531,  402.06043257,  599.51332396,
        786.65852197,  967.08162773, 1141.58216596, 1309.11187449,
       1469.5169325 , 1623.1186603 , 1764.08346261, 1889.57577979,
       2001.88900989, 2101.30278381, 2188.73611879, 2267.63555088,
       2343.27402808, 2407.73912728, 2455.36722708, 2497.96182966,
       2539.51471317, 2587.0820879 , 2640.40227477, 2696.82831531,
       2761.47051897, 2833.23756957, 2911.44949752, 2992.77581989,
       3069.47275146, 3138.80818543, 3193.71984223, 3235.53754503,
       3254.69213212, 3274.67470869, 3272.48929788, 3242.30051898,
       3180.17303765, 3093.41263257, 2991.37366459, 2885.20562543,
       2780.26784525, 2678.99618831, 2588.55564659, 2516.70881435,
       2459.78359522, 2416.54222519, 2386.46987666, 2354.38366305,
       2319.50665047, 2270.73906095, 2210.68743134, 2129.62814026,
       2023.02070449, 1883.49694516, 1715.79307033, 1529.51318541,
       1333.62474285, 1131.83863015,  928.84496897,  736.20050045,
        562.28654206,  408.87607081,  284.55487093,  189.83177085,
        120.11852167,   71.01521825,   41.44311728,   19.2423795 ,
          4.86717084,    0.        ,    0.        ])

pf=np.array([ 0.48275373,  0.72479413,  1.00434931,  1.20672616,  1.344403  ,
         1.43965187,  1.50811774,  1.47490908,  1.25073531,  1.10611433,
         1.07005598,  0.87735491,  0.59321103,  0.34456518, -0.4118638 ])
pfcov=np.array([[ 1.04959580e-02, -8.23738572e-04,  1.41380013e-04,
         -3.10699231e-05,  7.58052897e-06, -1.80740506e-06,
          4.44357115e-07, -1.24130187e-07,  3.92284870e-08,
         -1.18142971e-08,  3.64477677e-09, -1.58002135e-09,
          1.13406479e-09, -1.49119530e-09,  9.25836076e-09],
        [-8.23738572e-04,  5.69635568e-04, -9.77677714e-05,
          2.14856192e-05, -5.24212302e-06,  1.24986524e-06,
         -3.07283921e-07,  8.58390902e-08, -2.71274677e-08,
          8.16987823e-09, -2.52045316e-09,  1.09262379e-09,
         -7.84233815e-10,  1.03119838e-09, -6.40238515e-09],
        [ 1.41380013e-04, -9.77677714e-05,  1.79740693e-04,
         -3.95001341e-05,  9.63735606e-06, -2.29780879e-06,
          5.64924658e-07, -1.57810466e-07,  4.98723635e-08,
         -1.50198736e-08,  4.63371509e-09, -2.00872899e-09,
          1.44177092e-09, -1.89580175e-09,  1.17704344e-08],
        [-3.10699231e-05,  2.14856192e-05, -3.95001341e-05,
          1.06068481e-04, -2.58788924e-05,  6.17023444e-06,
         -1.51697461e-06,  4.23763534e-07, -1.33920706e-07,
          4.03323994e-08, -1.24427709e-08,  5.39397746e-09,
         -3.87154258e-09,  5.09073744e-09, -3.16067813e-08],
        [ 7.58052897e-06, -5.24212302e-06,  9.63735606e-06,
         -2.58788924e-05,  8.32828164e-05, -1.98568971e-05,
          4.88189045e-06, -1.36374541e-06,  4.30980330e-07,
         -1.29796738e-07,  4.00430201e-08, -1.73587659e-08,
          1.24593033e-08, -1.63828863e-08,  1.01716168e-07],
        [-1.80740506e-06,  1.24986524e-06, -2.29780879e-06,
          6.17023444e-06, -1.98568971e-05,  6.65354536e-05,
         -1.63579836e-05,  4.56956688e-06, -1.44410638e-06,
          4.34916131e-07, -1.34174061e-07,  5.81648462e-08,
         -4.17479829e-08,  5.48949199e-08, -3.40825224e-07],
        [ 4.44357115e-07, -3.07283921e-07,  5.64924658e-07,
         -1.51697461e-06,  4.88189045e-06, -1.63579836e-05,
          5.37709678e-05, -1.50208021e-05,  4.74697858e-06,
         -1.42962983e-06,  4.41048804e-07, -1.91195942e-07,
          1.37231428e-07, -1.80447239e-07,  1.12033993e-06],
        [-1.24130187e-07,  8.58390902e-08, -1.57810466e-07,
          4.23763534e-07, -1.36374541e-06,  4.56956688e-06,
         -1.50208021e-05,  5.29303187e-05, -1.67274083e-05,
          5.03773114e-06, -1.55416825e-06,  6.73736466e-07,
         -4.83576254e-07,  6.35860176e-07, -3.94785505e-06],
        [ 3.92284870e-08, -2.71274677e-08,  4.98723635e-08,
         -1.33920706e-07,  4.30980330e-07, -1.44410638e-06,
          4.74697858e-06, -1.67274083e-05,  6.88234141e-05,
         -2.07272908e-05,  6.39448523e-06, -2.77202798e-06,
          1.98963092e-06, -2.61618940e-06,  1.62430939e-05],
        [-1.18142971e-08,  8.16987823e-09, -1.50198736e-08,
          4.03323994e-08, -1.29796738e-07,  4.34916131e-07,
         -1.42962983e-06,  5.03773114e-06, -2.07272908e-05,
          9.12426044e-05, -2.81488541e-05,  1.22026103e-05,
         -8.75845805e-06,  1.15166008e-05, -7.15029378e-05],
        [ 3.64477677e-09, -2.52045316e-09,  4.63371509e-09,
         -1.24427709e-08,  4.00430201e-08, -1.34174061e-07,
          4.41048804e-07, -1.55416825e-06,  6.39448523e-06,
         -2.81488541e-05,  1.14113138e-04, -4.94683777e-05,
          3.55060680e-05, -4.66873517e-05,  2.89867025e-04],
        [-1.58002135e-09,  1.09262379e-09, -2.00872899e-09,
          5.39397746e-09, -1.73587659e-08,  5.81648462e-08,
         -1.91195942e-07,  6.73736466e-07, -2.77202798e-06,
          1.22026103e-05, -4.94683777e-05,  2.35745850e-04,
         -1.69207250e-04,  2.22492629e-04, -1.38138648e-03],
        [ 1.13406479e-09, -7.84233815e-10,  1.44177092e-09,
         -3.87154258e-09,  1.24593033e-08, -4.17479829e-08,
          1.37231428e-07, -4.83576254e-07,  1.98963092e-06,
         -8.75845805e-06,  3.55060680e-05, -1.69207250e-04,
          1.30782103e-03, -1.71966945e-03,  1.06768846e-02],
        [-1.49119530e-09,  1.03119838e-09, -1.89580175e-09,
          5.09073744e-09, -1.63828863e-08,  5.48949199e-08,
         -1.80447239e-07,  6.35860176e-07, -2.61618940e-06,
          1.15166008e-05, -4.66873517e-05,  2.22492629e-04,
         -1.71966945e-03,  3.55129328e-02, -2.20488586e-01],
        [ 9.25836076e-09, -6.40238515e-09,  1.17704344e-08,
         -3.16067813e-08,  1.01716168e-07, -3.40825224e-07,
          1.12033993e-06, -3.94785505e-06,  1.62430939e-05,
         -7.15029378e-05,  2.89867025e-04, -1.38138648e-03,
          1.06768846e-02, -2.20488586e-01,  3.05600819e+01]])



bsad_min=np.pi-0.7
bsad_max=np.pi

bsad=np.array([0.00000000e+00, 0.00000000e+00, 4.35151786e+00, 1.82303678e+01,
       3.99210735e+01, 6.87760297e+01, 1.16431607e+02, 1.84354156e+02,
       2.77052180e+02, 3.99457640e+02, 5.50107078e+02, 7.22314550e+02,
       9.12316635e+02, 1.11394717e+03, 1.31460086e+03, 1.50849033e+03,
       1.69301524e+03, 1.85932448e+03, 1.99856322e+03, 2.10418894e+03,
       2.18372863e+03, 2.24187437e+03, 2.28839458e+03, 2.32144136e+03,
       2.35072215e+03, 2.38010377e+03, 2.42183019e+03, 2.47686745e+03,
       2.54753197e+03, 2.63751458e+03, 2.73830606e+03, 2.84275647e+03,
       2.94893186e+03, 3.05187632e+03, 3.13950989e+03, 3.20070594e+03,
       3.23193073e+03, 3.23296566e+03, 3.21900811e+03, 3.19283799e+03,
       3.15066367e+03, 3.09558395e+03, 3.02618308e+03, 2.95021183e+03,
       2.86922255e+03, 2.79300514e+03, 2.72041041e+03, 2.65716994e+03,
       2.60100211e+03, 2.55002281e+03, 2.50521481e+03, 2.46620615e+03,
       2.42253452e+03, 2.37550836e+03, 2.31138235e+03, 2.23581925e+03,
       2.15737702e+03, 2.07277274e+03, 1.97471867e+03, 1.86416677e+03,
       1.74040019e+03, 1.60154118e+03, 1.44978615e+03, 1.29139152e+03,
       1.12616825e+03, 9.54184680e+02, 7.76333768e+02, 5.91853426e+02,
       3.97024370e+02, 1.98867181e+02, 2.42560215e-12])

pb=np.array([7.52975769, 0.2199202 , 0.2907205 , 0.37738431, 0.42230994,
        0.39386076, 0.41584288, 0.44030149, 0.44815295, 0.4295675 ,
        0.4461027 , 0.45887599, 0.45618429, 0.43189816, 0.50320307])
pbcov=np.array([[ 4.08476876e+03, -4.40364169e+00,  1.87456015e-01,
         -2.19573809e-02,  4.40107962e-03, -1.04984916e-03,
          2.37097320e-04, -5.83538030e-05,  1.70572786e-05,
         -5.43341730e-06,  1.71378225e-06, -5.76016863e-07,
          2.32356404e-07, -1.39577966e-07,  2.18132948e-07],
        [-4.40364169e+00,  2.74244961e-02, -1.16741713e-03,
          1.36743665e-04, -2.74085404e-05,  6.53813054e-06,
         -1.47656758e-06,  3.63409140e-07, -1.06227368e-07,
          3.38376147e-08, -1.06728971e-08,  3.58725193e-09,
         -1.44704263e-09,  8.69247691e-10, -1.35846342e-09],
        [ 1.87456015e-01, -1.16741713e-03,  8.34704250e-04,
         -9.77718381e-05,  1.95971299e-05, -4.67476895e-06,
          1.05574708e-06, -2.59837847e-07,  7.59526595e-08,
         -2.41939236e-08,  7.63113055e-09, -2.56488821e-09,
          1.03463672e-09, -6.21512845e-10,  9.71302512e-10],
        [-2.19573809e-02,  1.36743665e-04, -9.77718381e-05,
          1.46152217e-04, -2.92943657e-05,  6.98798202e-06,
         -1.57816177e-06,  3.88413250e-07, -1.13536268e-07,
          3.61657880e-08, -1.14072382e-08,  3.83407028e-09,
         -1.54660538e-09,  9.29055668e-10, -1.45193154e-09],
        [ 4.40107962e-03, -2.74085404e-05,  1.95971299e-05,
         -2.92943657e-05,  7.22920633e-05, -1.72448055e-05,
          3.89455678e-06, -9.58518629e-07,  2.80182583e-07,
         -8.92492249e-08,  2.81505596e-08, -9.46164370e-09,
          3.81668254e-09, -2.29270542e-09,  3.58304828e-09],
        [-1.04984916e-03,  6.53813054e-06, -4.67476895e-06,
          6.98798202e-06, -1.72448055e-05,  5.84866601e-05,
         -1.32085931e-05,  3.25086609e-06, -9.50253891e-07,
          3.02693417e-07, -9.54740959e-08,  3.20896597e-08,
         -1.29444786e-08,  7.77583041e-09, -1.21520957e-08],
        [ 2.37097320e-04, -1.47656758e-06,  1.05574708e-06,
         -1.57816177e-06,  3.89455678e-06, -1.32085931e-05,
          4.50571300e-05, -1.10893488e-05,  3.24150444e-06,
         -1.03254726e-06,  3.25681072e-07, -1.09464192e-07,
          4.41561830e-08, -2.65248993e-08,  4.14532081e-08],
        [-5.83538028e-05,  3.63409140e-07, -2.59837847e-07,
          3.88413250e-07, -9.58518629e-07,  3.25086609e-06,
         -1.10893488e-05,  3.47642659e-05, -1.01618701e-05,
          3.23695721e-06, -1.02098541e-06,  3.43161924e-07,
         -1.38426278e-07,  8.31535438e-08, -1.29952658e-07],
        [ 1.70572785e-05, -1.06227368e-07,  7.59526595e-08,
         -1.13536268e-07,  2.80182583e-07, -9.50253891e-07,
          3.24150444e-06, -1.01618701e-05,  3.57188796e-05,
         -1.13778747e-05,  3.58875430e-06, -1.20621100e-06,
          4.86567091e-07, -2.92283939e-07,  4.56782392e-07],
        [-5.43341727e-06,  3.38376147e-08, -2.41939236e-08,
          3.61657880e-08, -8.92492249e-08,  3.02693417e-07,
         -1.03254726e-06,  3.23695721e-06, -1.13778747e-05,
          4.47077868e-05, -1.41015143e-05,  4.73963947e-06,
         -1.91189817e-06,  1.14848936e-06, -1.79486331e-06],
        [ 1.71378224e-06, -1.06728970e-08,  7.63113055e-09,
         -1.14072382e-08,  2.81505596e-08, -9.54740959e-08,
          3.25681072e-07, -1.02098541e-06,  3.58875430e-06,
         -1.41015143e-05,  5.62698560e-05, -1.89127796e-05,
          7.62912638e-06, -4.58286463e-06,  7.16211734e-06],
        [-5.76016860e-07,  3.58725193e-09, -2.56488821e-09,
          3.83407028e-09, -9.46164370e-09,  3.20896597e-08,
         -1.09464192e-07,  3.43161924e-07, -1.20621100e-06,
          4.73963947e-06, -1.89127796e-05,  7.27650230e-05,
         -2.93522988e-05,  1.76321121e-05, -2.75555283e-05],
        [ 2.32356403e-07, -1.44704263e-09,  1.03463672e-09,
         -1.54660538e-09,  3.81668254e-09, -1.29444786e-08,
          4.41561830e-08, -1.38426278e-07,  4.86567091e-07,
         -1.91189817e-06,  7.62912638e-06, -2.93522988e-05,
          1.23424175e-04, -7.41416851e-05,  1.15868892e-04],
        [-1.39577966e-07,  8.69247691e-10, -6.21512845e-10,
          9.29055668e-10, -2.29270542e-09,  7.77583041e-09,
         -2.65248993e-08,  8.31535438e-08, -2.92283939e-07,
          1.14848936e-06, -4.58286463e-06,  1.76321121e-05,
         -7.41416851e-05,  3.99754424e-04, -6.24737653e-04],
        [ 2.18132947e-07, -1.35846342e-09,  9.71302512e-10,
         -1.45193154e-09,  3.58304828e-09, -1.21520957e-08,
          4.14532081e-08, -1.29952658e-07,  4.56782392e-07,
         -1.79486331e-06,  7.16211734e-06, -2.75555283e-05,
          1.15868892e-04, -6.24737653e-04,  6.95755185e-03]])






