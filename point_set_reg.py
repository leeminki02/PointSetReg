# solving point set registration problem

# import packages
import numpy as np
import matplotlib.pyplot as plt
import psr_tools as psr

# read data from triangulated_points.npy
point_data = np.load('triangulated_points1.npy')
point_clrs = np.load('triangulated_colors1.npy')
cloud1 = point_data
cloud1_color = point_clrs

# Method 1: CPD (Coherent Point Drift)
from pycpd import RigidRegistration

for i in range(20):
    seed = i + 1000
    np.random.seed(seed)
    cloud2, cloud2_color, tvec, rmat, scale = psr.make_transformation(cloud1, cloud1_color, 
                                                            noiseThrsh=0.003, noisePoints=20)

    # plot the data of cloud2
    # plot_cloud(cloud2, cloud2_color, show_eyes=False)
    psr.plot_clouds(cloud1, cloud2, cloud1_color, cloud2_color, 
                    savefig=True, file_id=str(seed)+'_1')

    target = cloud1.T
    source = cloud2.T

    reg = RigidRegistration(**{'X': target, 'Y': source})

    TY, (s_reg, R_reg, t_reg) = reg.register()
    # TY: transformed source points
    # s_reg: scale factor
    # R_reg: rotation matrix
    # t_reg: translation vector

    # plot the data of TY
    psr.plot_clouds(cloud1, TY.T, cloud1_color, cloud2_color, 
                    savefig=True, file_id=str(seed)+'_2')

    # Evaluation of the registration
    # Compute the registration error
    # save the registration error as txt file
    with open('results/clouds_'+str(seed)+'_regerr.txt', 'w') as f:
        f.write("true,estimated\n")
        f.write(str(np.round(1/scale, 4))+','+str(np.round( s_reg, 4))+'\n')
        f.write(str(np.round(   rmat.ravel(), 4))+','+str(np.round( R_reg.ravel(), 4))+'\n')
        f.write(str(np.round(   tvec, 4))+','+str(np.round(-t_reg, 4))+'\n')
    print(seed)

