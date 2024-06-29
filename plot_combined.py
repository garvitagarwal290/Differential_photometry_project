
import numpy as np
import matplotlib.pyplot as plt
import os


data_parentpath = '/home/gagarwal/Downloads/relative_LCs/'
os.system(" ls -d {}*/ > /home/gagarwal/Downloads/temp.txt".format(data_parentpath))
stars = [starpath.split('/')[-2] for starpath in open('/home/gagarwal/Downloads/temp.txt', 'r').read().split('\n')[:-1]]


listt = list(range(0,len(stars)))
# listt.remove(5)

colors=['blue', 'orange', 'green','red', 'purple', 'brown']

for k in listt:
    savefolder = '/home/gagarwal/Downloads/plots/' + stars[k] + '/'
    os.system('mkdir -p '+savefolder)

    os.system(" ls -d {}*/ > /home/gagarwal/Downloads/temp.txt".format(data_parentpath + stars[k]+ '/'))
    dirs = [dirpath.split('/')[-2] for dirpath in open('/home/gagarwal/Downloads/temp.txt', 'r').read().split('\n')[:-1]]

    reference_stars = [int(num) for num in open(data_parentpath + stars[k]+ '/chosen_ref_stars_index.txt', 'r').read().split('\n')[:-1]]
    stars_touse = [0]
    num_refstars = 5#len(reference_stars)
    for i in range(num_refstars): stars_touse.append(reference_stars[i])

    time_all = []
    rel_lc_all = []
    rel_lc_norm_all = []
    stares_means = []
    stares_stdev = []
    stares_times = []
    shift = 0.05

    overall_maxy =0.0
    overall_miny = 10.0

    for i in range(num_refstars+1):

        rel_lc_all.append([])
        rel_lc_norm_all.append([])

        stares_means.append([])
        stares_stdev.append([])

        for j in range(len(dirs)):
            # reference_stars = [int(num) for num in open(data_parentpath + stars[k]+'/'+ dirs[j]+ '/chosen_ref_stars_index.txt', 'r').read().split('\n')[:-1]]
            # stars_touse = [0]
            # for n in range(num_refstars): stars_touse.append(reference_stars[n+0])

            lc_filepath = data_parentpath+ stars[k] + '/' + dirs[j]+ '/star{}.txt'.format(stars_touse[i])

            lc_file = open(lc_filepath, 'r').readlines()
            lines = lc_file[12:]

            time=[float(line.split('\t')[1]) for line in lines]
            rel_lc=[float(line.split('\t')[4]) for line in lines]
            rel_lc_norm = np.array(rel_lc/np.mean(rel_lc)) - i*shift

            subdivisions = 1
            if(k==0 and j==3): subdivisions=4
            if(k==4 and (j==1 or j==2)): subdivisions=2

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
        ax = plt.gca()
        plt.scatter(time_all[j], rel_lc_norm_all[0][j], label='Target relative flux', zorder=10)
        
        for i in range(num_refstars):
            plt.scatter(time_all[j], rel_lc_norm_all[i+1][j], label='Reference star {}'.format(i+1))
        
        plt.xlabel('Time (J.D.)', fontsize=16)
        plt.ylabel('normalized relative flux', fontsize=16)
        plt.ylim(overall_miny-0.002, overall_maxy+0.002)
        plt.title('Relative Flux of Target: {}\nand {} reference stars\nObservation label: {}'.format(stars[k],num_refstars, dirs[j]), fontsize=16)
        plt.legend(loc='lower center')
        ax.tick_params(which="major", size=8, labelsize=14, width=1.5, direction='in', pad=5)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
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
    ax = plt.gca()
    plt.scatter(time_all, rel_lc_all[0]/np.mean(rel_lc_all[0]), label='Target relative flux')
    plt.errorbar(stares_times, stares_means[0]/np.mean(rel_lc_all[0]), yerr=stares_stdev[0]/np.mean(rel_lc_all[0]), fmt='-o', capsize=4, ecolor='black', color=colors[0], markerfacecolor='black', markeredgecolor='black', label='mean + std dev')
    
    for i in range(num_refstars):
        plt.scatter(time_all, np.array(rel_lc_all[i+1]/np.mean(rel_lc_all[i+1])) - (i+1)*shift, label='Reference star {}'.format(i+1))
        plt.errorbar(stares_times, stares_means[i+1]/np.mean(rel_lc_all[i+1]) - (i+1)*shift, yerr=stares_stdev[i+1]/np.mean(rel_lc_all[i+1]), fmt='-o', capsize=4, ecolor='black', color=colors[i+1], markerfacecolor='black', markeredgecolor='black')
    
    plt.xlabel('Time (J.D.)', fontsize=16)
    plt.ylabel('normalized relative flux', fontsize=16)
    plt.title('Relative flux of Target: {}\n and {} reference stars\nAll data'.format(stars[k], num_refstars), fontsize=16)
    plt.legend(loc='lower center')
    ax.tick_params(which="major", size=8, labelsize=14, width=1.5, direction='in', pad=5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    plt.savefig(savefolder+'all_data.png')
    plt.show()


    # radec = [float(coord) for coord in lc_file[5].split(':')[1].split(',')]
    # alldata_combinedfile = open('/home/gagarwal/Downloads/finalsLCs_forperiodicity/{}_lightcurve.txt'.format(str(radec[0])+','+str(radec[1])), 'w')
    # alldata_combinedfile.write('Time(J.D),Relative Flux\n')
    # for i in range(rel_lc_all[0].shape[0]):
    #     alldata_combinedfile.write('%f,%f\n' % (time_all[i], rel_lc_all[0][i]))
    # alldata_combinedfile.close()