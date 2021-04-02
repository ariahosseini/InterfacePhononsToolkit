# Generate plots

import matplotlib.pyplot as plt

plt.rc("font", size=18, family='sans-serif')
plt.rcParams["figure.figsize"] = (6, 5)
plt.rcParams['animation.html'] = 'html5'


def quickplot(x, y, xlab="x", ylab="y", plotlab=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-')
    ax.set(title=plotlab, xlabel=xlab, ylabel=ylab)
    plt.show()


def quickploterr(x, y, e, col='', xlab="x", ylab="y", plotlab=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-' + col)
    ax.fill_between(x, y - e, y + e, alpha=0.2)
    ax.set(title=plotlab, xlabel=xlab, ylabel=ylab)
    # plt.savefig('test.pdf')
    plt.show()


def quickmultiplot(data, xlab="x", ylab="y", plotlab=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x, y in data:
        ax.plot(x, y, '-')
        ax.set(title=plotlab, xlabel=xlab, ylabel=ylab)
    plt.show()


def quickmultierrplot(data, xlab="x", ylab="y", plotlab=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for x, y, e in data:
        ax.plot(x, y, '-')
        ax.fill_between(x, y - e, y + e, alpha=0.2)
        ax.set(title=plotlab, xlabel=xlab, ylabel=ylab)
    plt.show()
