import lightkurve as lk
import numpy as np
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(flux[i])
    return [im]


obj = SkyCoord(224.161770871613,-50.985501453475, unit="deg")

search_result = lk.search_targetpixelfile(obj)
search_result

tpf_file = search_result[4].download(quality_bitmask='default')
time = tpf_file.time
flux = tpf_file.flux
flux= flux.value
# print(time)

plt.imshow(flux[0])
plt.show()

fig = plt.figure( figsize=(8,8) )
im = plt.imshow(flux[0], interpolation='none', aspect='auto')
fps = 300

# anim = animation.FuncAnimation(
#                                fig, 
#                                animate_func, 
#                                frames = flux.shape[0],
#                                interval = 1000 / fps, # in ms
#                                )
# plt.show()

# anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])