
import numpy as np
import matplotlib.pyplot as plt
import os


# data_parentpath = '/home/gagarwal/Downloads/relative_LCs3/'
data_parentpath = '/home/gagarwal/Downloads/'
os.system(" ls -d {}*/ > /home/gagarwal/Downloads/temp.txt".format(data_parentpath))
stars = [starpath.split('/')[-2] for starpath in open('/home/gagarwal/Downloads/temp.txt', 'r').read().split('\n')[:-1]]
stars = ['Gaia_DR3_6375708771423327744']


# excl = [4,9, 11]
listt = list(range(0,len(stars)))
# for i in range(len(excl)):
#     listt.remove(excl[i])
# listt.reverse()

# listt = [13, 8, 7, 3,12, 0, 9, 1, 10, 11]

for k in [0]:
    # savefolder = '/home/gagarwal/Downloads/plots2/' + stars[k] + '/'
    savefolder = '/home/gagarwal/Downloads/Gaia_DR3_6375708771423327744/'
    os.system('mkdir -p '+savefolder)

    os.system(" ls -d {}*/ > /home/gagarwal/Downloads/temp.txt".format(data_parentpath + stars[k]+ '/'))
    dirs = [dirpath.split('/')[-2] for dirpath in open('/home/gagarwal/Downloads/temp.txt', 'r').read().split('\n')[:-1]]
    # if k==12: dirs = dirs[1:]

    reference_stars = [int(num) for num in open(data_parentpath + stars[k]+ '/chosen_ref_stars_index.txt', 'r').read().split('\n')[:-1]]
    stars_touse = [0]
    num_refstars = 5#len(reference_stars)
    for i in range(num_refstars): stars_touse.append(reference_stars[i+15])

    time_all = []
    rel_lc_all = []
    rel_lc_norm_all = []
    stares_means = []
    stares_stdev = []
    stares_times = []
    shift = 0.1

    overall_maxy =0.0
    overall_miny = 10.0

    for i in range(num_refstars+1):

        rel_lc_all.append([])
        rel_lc_norm_all.append([])

        stares_means.append([])
        stares_stdev.append([])

        for j in range(len(dirs)):
            lc_filepath = data_parentpath+ stars[k] + '/' + dirs[j]+ '/star{}.txt'.format(stars_touse[i])

            lc_file = open(lc_filepath, 'r').readlines()
            lines = lc_file[13:]

            time=[float(line.split('  ')[1]) for line in lines]
            rel_lc=[float(line.split('  ')[4]) for line in lines]
            rel_lc_norm = np.array(rel_lc/np.mean(rel_lc)) - i*shift

            subdivisions = 1
            # if(k==0 and j==3): subdivisions=4
            # if(k==4 and (j==1 or j==2)): subdivisions=2

            N = len(rel_lc)
            subdiv_idx = [0]
            for n in range(subdivisions): 
                idx = (N*(n+1))//subdivisions
                subdiv_idx.append(idx)

            for n in range(subdivisions):
                if(i==0): 
                    stares_times.append(np.mean(time[subdiv_idx[n]: subdiv_idx[n+1]])+0.5)
                stares_means[i].append(np.mean(rel_lc[subdiv_idx[n]: subdiv_idx[n+1]]))
                stares_stdev[i].append(np.std(rel_lc[subdiv_idx[n]: subdiv_idx[n+1]]))

            maxy = np.max(rel_lc_norm)
            miny = np.min(rel_lc_norm)
            if(maxy > overall_maxy): overall_maxy = maxy
            if(miny < overall_miny): overall_miny = miny

            rel_lc_all[i].append(rel_lc)
            rel_lc_norm_all[i].append(rel_lc_norm)

            if(i==0): time_all.append(time)


    for j in range(len(dirs)):

        plt.figure(figsize=(10,8))
        plt.scatter(time_all[j], rel_lc_norm_all[0][j], label='Target relative flux', zorder=10)
        
        for i in range(num_refstars):
            plt.scatter(time_all[j], rel_lc_norm_all[i+1][j], label='Reference star {}'.format(i+1))
        
        plt.xlabel('Time (J.D.)')
        plt.ylabel('normalized relative flux')
        plt.ylim(overall_miny-0.002, overall_maxy+0.002)
        plt.title('{}  {}'.format(stars[k], dirs[j]))
        # plt.legend()
        plt.savefig(savefolder+dirs[j]+'.png')
        plt.show()


    time_all = np.concatenate(time_all).ravel()
    rel_lc_all = [np.concatenate(rel_lc_all[i]).ravel() for i in range(num_refstars+1)]

    sort = np.argsort(time_all)
    time_all = time_all[sort]
    rel_lc_all = [rel_lc_all[i][sort] for i in range(num_refstars+1)]

    sort = np.argsort(stares_times)
    stares_times = np.array(stares_times)[sort]
    stares_means = [np.array(stares_means[i])[sort] for i in range(num_refstars+1)]

    plt.figure(figsize=(10,8))
    plt.scatter(time_all, rel_lc_all[0]/np.mean(rel_lc_all[0]), label='Target relative flux')
    plt.errorbar(stares_times, stares_means[0]/np.mean(rel_lc_all[0]), yerr=stares_stdev[0]/np.mean(rel_lc_all[0]), fmt='-ok', capsize=4)
    
    for i in range(num_refstars):
        plt.scatter(time_all, np.array(rel_lc_all[i+1]/np.mean(rel_lc_all[i+1])) - (i+1)*shift, label='Reference star {}'.format(i+1))
        plt.errorbar(stares_times, stares_means[i+1]/np.mean(rel_lc_all[i+1]) - (i+1)*shift, yerr=stares_stdev[i+1]/np.mean(rel_lc_all[i+1]), fmt='-ok', capsize=4)
    
    plt.xlabel('Time (J.D.)')
    plt.ylabel('normalized relative flux')
    plt.title('{} all data'.format(stars[k]))
    # plt.legend()
    plt.savefig(savefolder+'all_data.png')
    plt.show()


    # radec = [float(coord) for coord in lc_file[5].split(':')[1].split(',')]
    # alldata_combinedfile = open('/home/gagarwal/Downloads/finalsLCs_forperiodicity/{}_lightcurve.txt'.format(str(radec[0])+','+str(radec[1])), 'w')
    # alldata_combinedfile.write('Time(J.D),Relative Flux\n')
    # for i in range(rel_lc_all[0].shape[0]):
    #     alldata_combinedfile.write('%f,%f\n' % (time_all[i], rel_lc_all[0][i]))
    # alldata_combinedfile.close()