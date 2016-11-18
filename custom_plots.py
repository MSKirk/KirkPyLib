"""
plotting wrapper to display an array with x,y and intensity values

"""

def array_plot(array):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(array, interpolation='nearest')

    numrows, numcols = array.shape
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = array[row,col]
            return 'x=%1.4f, y=%1.4f, value=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)

    ax.format_coord = format_coord
    plt.show()
