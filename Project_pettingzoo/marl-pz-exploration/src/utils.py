from matplotlib import animation
import matplotlib.pyplot as plt
"""
save_frames_as_gif
from: https://gist.github.com/botforge

"""
def save_frames_as_gif(frames, path='./', filename='out.gif',fps=1,dpi=50,size=50.0,step_length=1):
    #Mess with this to change frame  size
    plt.figure(figsize=(frames[0].shape[1] / size, frames[0].shape[0] / size), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        tt = " Episode: {} Step: {}".format((i // step_length)+1, (i % step_length) )
        plt.title(tt,fontsize=24)
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=fps)
