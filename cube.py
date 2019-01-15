'''Made by Gio'''
import numpy as no
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

class Cube (object):
    facedict = {"U":0, "D":1, "F":2, "B":3, "R":4, "L":5}
    dictface = dict([(v, k) for k, v in facedict.items()])
    normals = [np.array([0., 1., 0.]), np.array([0., -1., 0.]),
               np.array([0., 0., 1.]), np.array([0., 0. -1.]),
               np.array([1., 0., 0.]), np.array([-1., 0., 0.])]
    xdirs = [np.array([1., 0., 0.]), np.array([1., 0., 0.]),
               np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
               np.array([0., 0., -1.]), np.array([0, 0., 1.])]
    colordict = {"w":0, "y":1, "b":2, "g":3, "o":4, "r":5}
    pltpos = [(0., 1.05), (0., -1.05), (0., 0.), (2.10, 0.), (1.05, 0.), (-1.05, 0.)]
    labelcolor = "#7f00ff"


    def __init__ (self, N, whiteplastic= False);
        self.N = N 
        self.stickers = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        self.stickercolors = ["w", "#ffcf00", "#00008f", "#009f0f", "#ff6f00", "#cf0000"]
        self.stickerthickness = 0.001
        self.stickerwidth - 0.9
        if whiteplastic:
            self.plasticcolor = "#dfdfdf"
        else:
            self.plasticcolor"#1f1f1f"
        self.fontsize = 12. * (self.N/5)
        return None
    def turn(self, f, d):
        for l in range(self.N):
            self.move(f, l, d)
            return None
    def move(self, f, l, d):
        i = self.facedict[f]
        12 = self.N - 1 - l 
        assert l < self.N 
        ds = range ((d+4) % 4)
        if f == "U":
             self._rotate([(self.facedict["U"], l2, range(self.N)),
                              (self.facedict["F"], l2, range(self.N)),
                              (self.facedict["D"], l2, range(self.N)),
                              (self.facedict["B"], l, range(self.N)[::-1])])
        if f == "L":
            return self.move("R", l2, -d)
        for d in ds:
            if l == 0:
                self.stickers[i] = np.rot90(self.stickers[i], 3)
            if l == self.N - 1:
                self.stickers[i2] = np.rot90(self.stickers[i2], 1)
        print "moved", f, l, len(ds)
        return None

    def _rotate(self, args):
      
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo
        return None

    def randomize(self, number):
       
        for t in range(number):
            f = self.dictface[np.random.randint(6)]
            l = np.random.randint(self.N)
            d = 1 + np.random.randint(3)
            self.move(f, l, d)
        return None

    def _render_points(self, points, viewpoint):
        
        v2 = np.dot(viewpoint, viewpoint)
        zdir = viewpoint / np.sqrt(v2)
        xdir = np.cross(np.array([0., 1., 0.]), zdir)
        xdir /= np.sqrt(np.dot(xdir, xdir))
        ydir = np.cross(zdir, xdir)
        result = []
        for p in points:
            dpoint = p - viewpoint
            dproj = 0.5 * dpoint * v2 / np.dot(dpoint, -1. * viewpoint)
            result += [np.array([np.dot(xdir, dproj),
                                 np.dot(ydir, dproj),
                                 np.dot(zdir, dpoint / np.sqrt(v2))])]
        return result

    def render_views(self, ax):
       
        csz = 2. / self.N
        x2 = 8.
        x1 = 0.5 * x2
        for viewpoint, shift in [(np.array([-x1, -x1, x2]), np.array([-1.5, 3.])),
                                 (np.array([x1, x1, x2]), np.array([0.5, 3.])),
                                 (np.array([x2, x1, -x1]), np.array([2.5, 3.]))]:
            for f, i in self.facedict.items():
                zdir = self.normals[i]
                if np.dot(zdir, viewpoint) < 0:
                    continue
                xdir = self.xdirs[i]
                ydir = np.cross(zdir, xdir) # insanity: left-handed!
                psc = 1. - 2. * self.stickerthickness
                corners = [psc * zdir - psc * xdir - psc * ydir,
                           psc * zdir + psc * xdir - psc * ydir,
                           psc * zdir + psc * xdir + psc * ydir,
                           psc * zdir - psc * xdir + psc * ydir]
                projects = self._render_points(corners, viewpoint)
                xys = [p[0:2] + shift for p in projects]
                zorder = np.mean([p[2] for p in projects])
                ax.add_artist(Polygon(xys, ec="none", fc=self.plasticcolor))
                for j in range(self.N):
                    for k in range(self.N):
                        corners = self._stickerpolygon(xdir, ydir, zdir, csz, j, k)
                        projects = self._render_points(corners, viewpoint)
                        xys = [p[0:2] + shift for p in projects]
                        ax.add_artist(Polygon(xys, ec="none", fc=self.stickercolors[self.stickers[i, j, k]]))
                x0, y0, zorder = self._render_points([1.5 * self.normals[i], ], viewpoint)[0]
                ax.text(x0 + shift[0], y0 + shift[1], f, color=self.labelcolor,
                        ha="center", va="center", rotation=20, fontsize=self.fontsize / (-zorder))
        return None

    def _stickerpolygon(self, xdir, ydir, zdir, csz, j, k):
        small = 0.5 * (1. - self.stickerwidth)
        large = 1. - small
        return [zdir - xdir + (j + small) * csz * xdir - ydir + (k + small + small) * csz * ydir,
                zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + small) * csz * ydir,
                zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + small) * csz * ydir,
                zdir - xdir + (j + large) * csz * xdir - ydir + (k + small + small) * csz * ydir,
                zdir - xdir + (j + large) * csz * xdir - ydir + (k + large - small) * csz * ydir,
                zdir - xdir + (j + large - small) * csz * xdir - ydir + (k + large) * csz * ydir,
                zdir - xdir + (j + small + small) * csz * xdir - ydir + (k + large) * csz * ydir,
                zdir - xdir + (j + small) * csz * xdir - ydir + (k + large - small) * csz * ydir]

    def render_flat(self, ax):
     
        for f, i in self.facedict.items():
            x0, y0 = self.pltpos[i]
            cs = 1. / self.N
            for j in range(self.N):
                for k in range(self.N):
                    ax.add_artist(Rectangle((x0 + j * cs, y0 + k * cs), cs, cs, ec=self.plasticcolor,
                                            fc=self.stickercolors[self.stickers[i, j, k]]))
            ax.text(x0 + 0.5, y0 + 0.5, f, color=self.labelcolor,
                    ha="center", va="center", rotation=20, fontsize=self.fontsize)
        return None

    def render(self, flat=True, views=True):
        
        assert flat or views
        xlim = (-2.4, 3.4)
        ylim = (-1.2, 4.)
        if not flat:
            ylim = (2., 4.)
        if not views:
            xlim = (-1.2, 3.2)
            ylim = (-1.2, 2.2)
        fig = plt.figure(figsize=((xlim[1] - xlim[0]) * self.N / 5., (ylim[1] - ylim[0]) * self.N / 5.))
        ax = fig.add_axes((0, 0, 1, 1), frameon=False,
                          xticks=[], yticks=[])
        if views:
            self.render_views(ax)
        if flat:
            self.render_flat(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return fig

def adjacent_edge_flip(cube):
    """
    Do a standard edge-flipping algorithm.  Used for testing.
    """
    ls = range(cube.N)[1:-1]
    cube.move("R", 0, -1)
    for l in ls:
        cube.move("U", l, 1)
    cube.move("R", 0, 2)
    for l in ls:
        cube.move("U", l, 2)
    cube.move("R", 0, -1)
    cube.move("U", 0, -1)
    cube.move("R", 0, 1)
    for l in ls:
        cube.move("U", l, 2)
    cube.move("R", 0, 2)
    for l in ls:
        cube.move("U", l, -1)
    cube.move("R", 0, 1)
    cube.move("U", 0, 1)
    return None

def swap_off_diagonal(cube, f, l1, l2):
    """
    A big-cube move that swaps three cubies (I think) but looks like two.
    """
    cube.move(f, l1, 1)
    cube.move(f, l2, 1)
    cube.move("U", 0, -1)
    cube.move(f, l2, -1)
    cube.move("U", 0, 1)
    cube.move(f, l1, -1)
    cube.move("U", 0, -1)
    cube.move(f, l2, 1)
    cube.move("U", 0, 1)
    cube.move(f, l2, -1)
    return None

def checkerboard(cube):
    """
    Dumbness.
    """
    ls = range(cube.N)[::2]
    for f in ["U", "F", "R"]:
        for l in ls:
            cube.move(f, l, 2)
    if cube.N % 2 == 0:
        for l in ls:
            cube.move("F", l, 2)
    return None

if __name__ == "__main__":
    """
    Functional testing.
    """
    np.random.seed(42)
    c = Cube(6, whiteplastic=False)

    for m in range(32):
        c.render(flat=False).savefig("test%02d.png" % m, dpi=865 / c.N)
        c.randomize(1)