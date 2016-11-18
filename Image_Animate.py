# To animate png images from a python session
# animation_obj = Image_animate.Animation('User/My/Directory', 'png')
# animation_obj.peek()
# animaton_obj.save_as_mp4('User/My/Desktop', 'MyAnimation.mp4)


from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.image as mgimg
import numpy as np
import os
import fnmatch


class Animation:
    def __init__(self, directory, extension, frame_rate=10):
        """

        :param directory:
        :param extension:
        :param frame_rate:
        """
        self.directory = directory
        self.extension = extension
        self.frame_rate = frame_rate  # in milliseconds

        # To check that the directory exists
        if not os.path.isdir(directory):
            raise ValueError('No such directory found.')

        self.filelist = fnmatch.filter(os.listdir(self.directory), '*.' + self.extension.lower())

        # Is there at least one file in the directory with the required type?
        if not self.filelist:
            raise ValueError('No files found.')

        self.n = len(self.filelist)

        fig = plt.figure()
        ax = plt.gca()

        self.imshape = mgimg.imread(os.path.join(self.directory,self.filelist[0])).shape
        self.imobj = ax.imshow(np.zeros(self.imshape[:2]), origin='lower', alpha=1.0, zorder=1, aspect=1)
        self.anim = animation.FuncAnimation(fig, self._animate_func, init_func=self._init_func, repeat=True,
                                            frames=range(0,self.n-1), interval=self.frame_rate, blit=False, repeat_delay=1000)

    def _animate_func(self, frame_number):
        """

        :param frame_number:
        :return:
        """
        img = mgimg.imread(os.path.join(self.directory,self.filelist[frame_number]))[-1::-1]
        return self.imobj.set_data(img),

    def _init_func(self):
        return self.imobj.set_data(np.zeros(self.imshape[:2])),

    def save_as_mp4(self, odir, oname):
        """
        Save as an mp4 format file.
        :param odir:
        :param oname:
        :return:
        """
        self.anim.save(os.path.join(odir,oname))

    def peek(self):
        plt.show()

