import numpy as np
from matplotlib import pyplot
import matplotlib as mpl


def show_image(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


def load_data():
    # load training data
    trainLabels = np.loadtxt('trainingLabels.gz', dtype=np.uint8, delimiter=',')
    trainData = np.loadtxt('trainingData.gz', dtype=np.uint8, delimiter=',')

    # Visualize sample image
    imgHeight = np.sqrt(trainData.shape[1])
    show_image(trainData[1].reshape((imgHeight, imgHeight)))

    # load test data
    testData = np.loadtxt('testData.gz', dtype=np.uint8, delimiter=',')

if __name__ == '__main__':
    load_data()
