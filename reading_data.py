import numpy as np
import h5py
import matplotlib.pyplot as plt
#import ang_tof as at


#scipy.optimize.minimize

p='C:\\Users\\moell\\Desktop\\COMPTEL_Simulation_Project\\Werkstudent\\Data_from_Simulations\\Thesis_Simulations\\Simulation_Batch_3/'
h5_f_n="Simulation1"
# h5_f_n="Activation_fast"
tra_f_n="ActivationStep1.p1.inc1.id1.tra"
# tra_f_n="ActivationStep1.p1.inc2.id1.tra"
# tra_f_n="ActivationStep1.p1.inc3.id1.tra"
# tra_f_n="ActivationStep1.p1.inc4.id1.tra"
# tra_f_n="ActivationStep1.p1.inc5.id1.tra"

p2='C:\\Users\\moell\\Desktop\\COMPTEL_Simulation_Project\\Werkstudent\\Plots/'
p3='C:\\Users\\moell\\Desktop\\COMPTEL_Simulation_Project\\Werkstudent\\Data_from_Simulations\\Thesis_Simulations/'

p_create="Activation/"
h5_1="temp1"
h5_2="temp2"
h5_3="temp3"
h5_4="temp4"
h5_5="temp5"
h5_6="temp6"

parameters={
    0:"ID",
    1:"Energy of scattered gamma-ray [keV]",
    2:"Energy error of scattered gamma-ray [keV]",
    3:"Energy of recoil electron [keV]",
    4:"Energy error of recoil electron [keV]",
    5:"x position of first interaction [cm]",
    6:"x position error of first interaction [cm]",
    7:"y position of first interaction [cm]",
    8:"y position error of first interaction [cm]",
    9:"z position of first interaction [cm]",
    10:"z position error of first interaction [cm]",
    11:"x position of second interaction [cm]",
    12:"x position error of second interaction [cm]",
    13:"y position of second interaction [cm]",
    14:"y position error of second interaction [cm]",
    15:"z position of second interaction [cm]",
    16:"z position error of second interaction [cm]",
    17:"Time of flight between first and second interaction [s]",
    18:"Error of time of flight between first and second interaction [s]",
    19:"Absolute time [s]",
    20:"Length of compton sequence in 'detected interactions'",
    21:"First compton quality factor/PSD (for real data)",
    22:"Second compton quality factor/Zeta (for real data)",
    23:"Deposited energy of first track in its first layer [keV]/Rigidity (for real data)",
    24:"Shortest distance in the sequence [cm]"
    }

#Create and Read Data Files
def create_h5py_file(h5_file_name=h5_5,h5_dataset_name="Data",path=p_create):
    with h5py.File(path+h5_file_name,"w") as h5:
        h5.create_dataset(h5_dataset_name, data=create_np_array())
        
def read_h5py_file(path=p,h5_file_name=h5_f_n,h5_dataset_name="Data"):
    with h5py.File(path+h5_file_name,"r") as h5:
        data_array=np.array(h5.get(h5_dataset_name))
    #return slice_selection(data_array, 17, 0.375e-8, 0.85e-8)
    return data_array
        
def create_np_array(tra_filename=tra_f_n,path=p_create):   
    data_array=np.array([])
    temp_array=np.array([])
    
    with open(path+tra_filename, "r") as sim_data:
        for line in sim_data:
            
            line=line.strip()+" "
            
            indicator=line[0:2]
            
            if indicator=="SE":
                data_array=merge(data_array, temp_array)
                temp_array=np.array([0.0 for i in range(len(parameters))])
            
            elif indicator=="ID":
                insert_pos={1:0}
                temp_array=line_extracter(temp_array, line, insert_pos)
                if data_array.shape[0]%5000==0:
                    print(data_array.shape)
            
            elif indicator=="CE":
                insert_pos={1:1,2:2,3:3,4:4}
                temp_array=line_extracter(temp_array, line, insert_pos)
            
            elif indicator=="CD":
                insert_pos={1:5,2:7,3:9,4:6,5:8,6:10,7:11,8:13,9:15,10:12,11:14,12:16}
                temp_array=line_extracter(temp_array, line, insert_pos)
            
            elif indicator=="TF":
                insert_pos={1:17,2:18}
                temp_array=line_extracter(temp_array, line, insert_pos)
                
            elif indicator=="TI":
                insert_pos={1:19}
                temp_array=line_extracter(temp_array, line, insert_pos)
                
            elif indicator=="SQ":
                insert_pos={1:20}
                temp_array=line_extracter(temp_array, line, insert_pos)
                
            elif indicator=="CT":
                insert_pos={1:21,2:22}
                temp_array=line_extracter(temp_array, line, insert_pos)
                
            elif indicator=="TE":
                insert_pos={1:23}
                temp_array=line_extracter(temp_array, line, insert_pos)
                
            elif indicator=="LA":
                insert_pos={1:24}
                temp_array=line_extracter(temp_array, line, insert_pos)
            
        else:
            data_array=merge(data_array, temp_array)
            return data_array

def merge(data_array, temp_array):
    try:
        data_array=np.vstack((data_array,temp_array))
    except:
        data_array=temp_array
    return data_array

def line_extracter(temp_array,line,insert_pos):
    space_flag=False
    space_counter=0
    space_pos=0
    char_counter=0
    
    while space_counter<=len(insert_pos):
        
        if line[char_counter]==" ":
            
            if space_counter!=0 and not space_flag:
                temp_array[insert_pos[space_counter]]=float(line[space_pos:char_counter])
                
            space_counter+=1 if not space_flag else 0
            space_flag=True
            space_pos=char_counter
            
        else:
            space_flag=False
            
        char_counter+=1
        
    return temp_array

def combine_h5py_files():
    new_file_name=h5_f_n
    h5_dataset_name="Data"
    path=p_create
    
    old_file_name_1="temp1"
    old_file_name_2="temp2"
    old_file_name_3="temp3"
    old_file_name_4="temp4"
    old_file_name_5="temp5"
    # old_file_name_6="temp6"
    
    with h5py.File(path+old_file_name_1,"r") as h5:
        data_array1=np.array(h5.get(h5_dataset_name))
        
    with h5py.File(path+old_file_name_2,"r") as h5:
        data_array2=np.array(h5.get(h5_dataset_name))
        
    with h5py.File(path+old_file_name_3,"r") as h5:
        data_array3=np.array(h5.get(h5_dataset_name))
        
    with h5py.File(path+old_file_name_4,"r") as h5:
        data_array4=np.array(h5.get(h5_dataset_name))
        
    with h5py.File(path+old_file_name_5,"r") as h5:
        data_array5=np.array(h5.get(h5_dataset_name))
        
    # with h5py.File(path+old_file_name_6,"r") as h5:
    #     data_array6=np.array(h5.get(h5_dataset_name))
        
    data_array=np.vstack((data_array1,data_array2,data_array3,data_array4,data_array5))#,data_array6))
    
    with h5py.File(path+new_file_name,"w") as h5:
        h5.create_dataset(h5_dataset_name, data=data_array)
    
    
    
    

#Functions and Data Sets
def save_line(x_min,x_max,function_name,data_points=5):
    data=np.array([])
    for i in range(data_points):
        data=np.append(data,[function_name((x_max-x_min)/(data_points-1)*i+x_min)])
    return data

def continue_line(value, data,x_min,x_max,sig_dec=7):
    data_points=data.size
    increment=round((x_max-x_min)/(data_points-1),sig_dec)
    if type(value)==type(np.array([])):
        output=np.array([])
        for i in value:
            if i>x_max or i<x_min:
                output=np.append(output,0)
            else:
                index=int((i-x_min)//increment)
                progress=(i-x_min)%increment
                try:
                    output=np.append(output,data[index]+(data[index+1]-data[index])*progress/increment)
                except:
                    print(value)
                    output=np.append(output,0)
        return output
    else:
        if value>x_max or value<x_min:
            return 0
        else:
            index=int((value-x_min)//increment)
            progress=(value-x_min)%increment
            try:
                return data[index]+(data[index+1]-data[index])*progress/increment
            except:
                print(value,"rd.continue_line")
                return 0

def continue_log(value,x_data,y_data):
    if value<x_data[0] or value>x_data[-1]:
        return 0
    for i in range(len(x_data)-1):
        if value>=x_data[i] and value<x_data[i+1]:
            s=np.log(y_data[i+1]/y_data[i])/np.log(x_data[i+1]/x_data[i])
            f=y_data[i]/(x_data[i]**s)
            return f*value**s

def continue_linear_set(value,x_data,y_data):
    if value<x_data[0] or value>x_data[-1]:
        return 0
    for i in range(len(x_data)-1):
        if value>=x_data[i] and value<x_data[i+1]:
            return y_data[i]+(value-x_data[i])/(x_data[i+1]-x_data[i])*(y_data[i+1]-y_data[i])


def fit_func(a,y1,y2):
    return np.sum((y1-a*y2)**2)


#Create Plots
def plot_histogram_1D(data_array,parameter,num_bins=100, label=""):
    data=data_array[:,parameter].reshape(-1) if type(parameter)==type(1) else parameter
    n,bins,patches=plt.hist(data,num_bins)
    if type(parameter)==type(1):
        plt.xlabel(parameters[parameter])
    else:
        plt.xlabel(label)
    plt.ylabel("Frequency [Counts/Bin]")
    plt.savefig(p2+'plot.pdf',bbox_inches='tight')

def plot_parameter_correlation(data_array,parameter1,parameter2,*labels):
    x=data_array[:,parameter1:parameter1+1].reshape(-1) if type(parameter1)==type(1) else parameter1
    y=data_array[:,parameter2:parameter2+1].reshape(-1) if type(parameter2)==type(1) else parameter2
    plt.plot(x,y,"b+")
    if type(parameter1)==type(1):
        plt.xlabel(parameters[parameter1])
    else:
        plt.xlabel(labels[0])
    if type(parameter2)==type(1):
        plt.ylabel(parameters[parameter2])
    else:
        plt.ylabel(labels[1])
        
def plot_histogram_2D(data_array,parameter1,parameter2,size1=100, size2=100,*labels):
    x=data_array[:,parameter1:parameter1+1].reshape(-1) if type(parameter1)==type(1) else parameter1
    y=data_array[:,parameter2:parameter2+1].reshape(-1) if type(parameter2)==type(1) else parameter2
    plt.hist2d(x,y,bins=(size1,size2))
    # if type(parameter1)==type(1):
    #     plt.xlabel(parameters[parameter1])
    # else:
    #     plt.xlabel(labels[0])
    # if type(parameter2)==type(1):
    #     plt.ylabel(parameters[parameter2])
    # else:
    #     plt.ylabel(labels[1])

def plot_function(function_name,x_min,x_max, points=100):
    x=np.arange(x_min,x_max,(x_max-x_min)/points)
    y=np.array([function_name(i) for i in x])
    plt.plot(x,y)

def plot_points(data_points,x_min,x_max,linew=1.0):
    increment=(x_max-x_min)/(data_points.size-1)
    x=np.arange(x_min,x_max+increment,increment)
    if x.size>data_points.size:
        x=x[:-1]
    plt.plot(x,data_points,lw=linew)

def plot_points_error(data_points,error_points,x_min,x_max):
    increment=(x_max-x_min)/(data_points.size-1)
    x=np.arange(x_min,x_max+increment,increment)
    if x.size>data_points.size:
        x=x[:-1]
    plt.errorbar(x,data_points,error_points)

#Calculate Different Values
def calculate_velocity(data_array):
    return data_array[:,24:25].reshape(-1) / data_array[:,17:18].reshape(-1)

def calculate_angles(data_array,initial_direction=[0,0,-1]):
    x=-data_array[:,5].reshape(-1)+data_array[:,11].reshape(-1)
    y=-data_array[:,7].reshape(-1)+data_array[:,13].reshape(-1)
    z=-data_array[:,9].reshape(-1)+data_array[:,15].reshape(-1)
    distance=data_array[:,24].reshape(-1)
    return np.arccos((x*initial_direction[0]+y*initial_direction[1]+z*initial_direction[2])/distance)

def calculate_average(data_array,parameter):
    return np.average(data_array,0)[parameter]

def calculate_standard_devation(data_array,parameter):
    return np.std(data_array,0)[parameter]

def calculate_angles_energy(data_array):
    Ee=data_array[:,3].reshape(-1)
    Eg=data_array[:,1].reshape(-1)
    

#Make Data Selections
def slice_selection(data_array,parameter,greater_than,less_than):
    data_array=data_array[greater_than<data_array[:,parameter]]
    data_array=data_array[less_than>data_array[:,parameter]]
    return data_array

def cut_selection(data_array,parameter,cut_value,greater_than=True):
    if greater_than:
        return data_array[data_array[:,parameter]>cut_value]
    else:
        return data_array[data_array[:,parameter]<cut_value]

def extract_parameter(data_array,parameter):
    return data_array[:,parameter].reshape(-1)

def select_cell(data_array,cell_number,forward=True,D1=True):
    if forward:
        par1=5
        par2=7
        par3=11
        par4=13
    else:
        par1,par2,par3,par4=11,13,5,7
    if D1:
        return data_array[ (data_array[:,par1]-D1_points[cell_number,0])**2 
                          + (data_array[:,par2]-D1_points[cell_number,1])**2 <
                          (D1_radius+1)**2]
    else:
        return data_array[ (data_array[:,par3]-D2_points[cell_number,0])**2 
                          + (data_array[:,par4]-D2_points[cell_number,1])**2 <
                          (D2_radius+1)**2]
    
def velocity_cut(data_array,V_min,V_max):
    data_array=data_array[data_array[:,24]/data_array[:,17]>=V_min]
    data_array=data_array[data_array[:,24]/data_array[:,17]<=V_max]
    return data_array

def angular_cut(data_array,A_min,A_max):
    output=np.array([])
    angles=calculate_angles(data_array)
    for i in range(len(data_array)):
        if angles[i]>=A_min and angles[i]<=A_max:
            try:
                output=np.vstack((output,data_array[i]))
            except:
                output=data_array[i]
    return output



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
D1_D2_Distance=158
#D1_D2_Distance=157.5#################





# d=read_h5py_file()
# df=cut_selection(d, 9, 0)
# db=cut_selection(d, 9, 0, False)
# af=calculate_angles(df)
# ab=calculate_angles(db)


