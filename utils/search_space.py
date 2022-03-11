import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def parallel_coordinates(frame, class_column, cols=None, ax=None, color=None,use_columns=False, xticks=None, colormap=None,**kwds):

    n = len(frame)
    class_col = frame[class_column]
    class_min = np.amin(class_col)
    class_max = np.amax(class_col)

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends = set([])

    ncols = len(df.columns)

    # determine values to use for xticks
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError('Columns must be numeric to be used as xticks')
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError('xticks specified must be numeric')
        elif len(xticks) != ncols:
            raise ValueError('Length of xticks must match number of columns')
        x = xticks
    else:
        x = range(ncols)

    fig = plt.figure()
    ax = plt.gca()

    Colorm = plt.get_cmap(colormap)

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        ax.plot(x, y, color=Colorm((kls - class_min)/(class_max-class_min)), alpha=1-(kls - class_min)/(class_max-class_min), **kwds)

    for i in x:
        ax.axvline(i, linewidth=1, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc='upper right')
    ax.grid()

    bounds = np.linspace(class_min,class_max,10)
    cax,_ = mpl.colorbar.make_axes(ax)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=Colorm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%.2f')

    return fig

# TYPES: REAL: R, DISCRETE: D, CATEGORICAL: C
class Searchspace:

    # Initialize the search space
    def __init__(self, label, type, values, neighborhood = 0.10):

        self.label = label
        self.type = type
        self.values = values
        self.sub_values = None

        self.n_variables = len(label)

        # if list is given do nothing, if a percentage is given, compute the neighborhood of a solution is located in an area corresponding to x% of the search space
        self.create_neighborhood(neighborhood)

    # Create the neighborhood for each variables
    def create_neighborhood(self,neighborhood):

        self.neighborhood = []

        if type(neighborhood) == float:

            for i in range(self.n_variables):
                if self.type[i] == "R":
                    self.neighborhood.append((self.values[i][1]-self.values[i][0])*neighborhood)
                elif self.type[i] == "D":
                    self.neighborhood.append(int(np.ceil((self.values[i][1]-self.values[i][0])*neighborhood)))
                else:
                    self.neighborhood.append(-1)
        else:
            self.neighborhood = neighborhood

    # Return a or size=n random attribute from the search space, can exclude one attribute
    def random_attribute(self,size=1,replace=True, exclude=None):

        try:
            index = self.label.index(exclude)
            p = np.full(self.n_variables, 1/(self.n_variables-1))
            p[index] = 0

        except ValueError:
            p = np.full(self.n_variables, 1/self.n_variables)
        except IndexError:
            p = np.full(self.n_variables, 1/self.n_variables)

        return np.random.choice(self.label,size=size,replace=replace,p=p)

    # Return a or size=n random value from an attribute, can exclude one value
    def random_value(self, attribute, size=1, replace = True, exclude=None):

        try:
            index = self.label.index(attribute)
        except ValueError:
            print("Attribute doesn't exist: "+attribute)

        if self.type[index]=="R":
            return np.random.uniform(self.values[index][0],self.values[index][1],size=size)

        elif self.type[index]=="D":
            return np.random.randint(self.values[index][0],self.values[index][1],size=size)

        elif self.type[index]=="C":

            try:
                idx = self.label.index(exclude)
                p = np.full(len(self.values[index]), 1/(len(self.values[index])-1))
                p[idx] = 0

            except ValueError:
                p = np.full(len(self.values[index]), 1/len(self.values[index]))
            except IndexError:
                p = np.full(len(self.values[index]), 1/len(self.values[index]))


            return np.random.choice(self.values[index],size=size,replace=replace,p=p)

    # Return a neighbor of a given point in the search space, can select neighbor of a particular attribute
    def get_neighbor(self, point, size=1, attribute=None, replace=True):

        points = []

        if attribute == None:
            for i in range(size):
                attribute = self.random_attribute()
                index = self.label.index(attribute)
                neighbor = point[:]

                if self.type[index]=="R":
                    neighbor[index] = np.random.uniform(np.max([point[index]-self.neighborhood[index],self.values[index][0]]),
                    np.min([point[index]+self.neighborhood[index], self.values[index][1]]))

                elif self.type[index]=="D":

                    search_list = list(range(int(np.max([point[index]-self.neighborhood[index],self.values[index][0]])),
                                             int(np.min([point[index]+self.neighborhood[index],self.values[index][1]])+1)))

                    try:
                        search_list.remove(point[index])
                    except ValueError:
                        pass
                    except IndexError:
                        pass
                    neighbor[index] = np.random.choice(search_list)


                elif self.type[index]=="C":

                    try:
                        idx = self.label.index(point[index])
                        p = np.full(len(self.values[index]), 1/(len(self.values[index])-1))
                        p[idx] = 0

                    except ValueError:
                        p = np.full(len(self.values[index]), 1/len(self.values[index]))
                    except IndexError:
                        p = np.full(len(self.values[index]), 1 / len(self.values[index]))

                    neighbor[index] = np.random.choice(self.values[index],p=p)

                points.append(neighbor[:])
        else:

            index = self.label.index(attribute)

            if self.type[index]=="R":
                for i in range(size):
                    points.append(np.random.uniform(np.max([point[index]-self.neighborhood[index],self.values[index][0]]),
                    np.min([point[index]+self.neighborhood[index],self.values[index][1]])))

            elif self.type[index]=="D":

                search_list = list(range(int(np.max([point[index]-self.neighborhood[index],self.values[index][0]])),
                                         int(np.min([point[index]+self.neighborhood[index],self.values[index][1]])+1)))

                try:

                    if len(search_list)>1:
                        search_list.remove(point[index])

                except ValueError:
                    pass
                except IndexError:
                    pass

                for i in range(size):
                    points.append(np.random.choice(search_list))

            elif self.type[index]=="C":

                try:
                    idx = self.label.index(point[index])
                    p = np.full(len(self.values[index]), 1/(len(self.values[index])-1))
                    p[idx] = 0

                except ValueError:
                    p = np.full(len(self.values[index]), 1/len(self.values[index]))
                except IndexError:
                    p = np.full(len(self.values[index]), 1 / len(self.values[index]))
                for i in range(size):
                    points.append(np.random.choice(self.values[index], p=p))

        return points

    # Return a random point of the search space
    def random_point(self,size=1):
        points = []

        for i in range(size):
            new_point = []
            for l in self.label:
                new_point.append(self.random_value(l)[0])
            points.append(new_point[:])

        return points

    # Convert a point to continuous, or convert a continuous point to a point from the search space
    def convert_to_continuous(self,points,reverse=False,sub_values=False):

        if sub_values and self.sub_values != None:
            val = self.sub_values
        else:
            val = self.values

        res = []

        if reverse:

            for point in points:
                converted = []
                for att in range(self.n_variables):

                    if self.type[att]=="R":

                        converted.append(point[att]*(val[att][1]-val[att][0])+val[att][0])

                    elif self.type[att]=="D":

                        converted.append(int(point[att]*(val[att][1]-val[att][0])+val[att][0]))

                    elif self.type[att]=="C":

                        n_values = len(val[att])-1
                        converted.append(val[att][int(point[att]*n_values)])

                res.append(converted[:])
        else:

            for point in points:
                converted = []
                for att in range(self.n_variables):

                    if self.type[att]=="R" or self.type[att]=="D":

                        converted.append((point[att]-val[att][0])/(val[att][1]-val[att][0]))

                    elif self.type[att]=="C":

                        idx = self.values[att].index(point[att])
                        n_values = len(val[att])

                        converted.append(idx/n_values)

                res.append(converted[:])

        return res

    def general_convert(self, in_place = False):

        if in_place:

            neighborhood = []

            for att in range(self.n_variables):

                if self.type[att]=="R" or self.type[att]=="D":

                    neighborhood.append(self.neighborhood[att]/(self.values[att][1]-self.values[att][0]))

                elif self.type[att]=="C":

                    neighborhood.append(1)

            self.type = ["R"]*self.n_variables
            self.values = [[0,1]*self.n_variables]

            self.neighborhood = neighborhood[:]

            return self

        else:

            label = self.label[:]
            type = ["R"]*self.n_variables
            values = [[0,1]]*self.n_variables
            neighborhood = []

            for att in range(self.n_variables):

                if self.type[att]=="R" or self.type[att]=="D":

                    neighborhood.append(self.neighborhood[att]/(self.values[att][1]-self.values[att][0]))

                elif self.type[att]=="C":

                    neighborhood.append(1)

            sp = Searchspace(label, type, values, neighborhood)

            return sp

    def subspace(self,lo_bounds,up_bounds):

        new_values = []
        new_neighborhood = []

        for i in range(len(lo_bounds)):

            if self.type[i] == "R":
                new_values.append([np.max([lo_bounds[i],self.values[i][0]]),np.min([up_bounds[i],self.values[i][1]])])
                new_neighborhood.append(self.neighborhood[i]*(new_values[-1][1]-new_values[-1][0])/(self.values[i][1]-self.values[i][0]))
            elif self.type[i] == "D":
                new_values.append([np.max([lo_bounds[i],self.values[i][0]]),np.min([up_bounds[i],self.values[i][1]])])
                new_neighborhood.append(int(self.neighborhood[i]*(new_values[-1][1]-new_values[-1][0])/(self.values[i][1]-self.values[i][0])))
            else:
                new_neighborhood.append(-1)

                lo_idx = self.values[i].index(lo_bounds[i])
                up_idx = self.values[i].index(up_bounds[i])

                if lo_idx > up_idx:
                    inter = lo_idx
                    up_idx = lo_idx
                    lo_idx = inter

                new_values.append(self.values[i][lo_idx:up_idx+1])

        subspace = Searchspace(self.label,self.type, new_values, new_neighborhood)
        if subspace.sub_values == None:
            subspace.sub_values = self.values
        else:
            subspace.sub_values = self.sub_values

        return subspace

    def show(self,X,Y):

        f, plots = plt.subplots(self.n_variables,self.n_variables)

        if len(X) < 100:
            s = 1
        else:
            s = 2500/len(X)

        if self.type[0] == "C":
            X.iloc[:,0].value_counts().plot(kind="bar", ax=plots[0,0])
            plots[0,0].set_yticks([])
            plots[0,0].xaxis.tick_top()
            plots[0,0].tick_params(axis='x', labelsize= 7/len(self.type[0]))
        else:
            plots[0,0].hist(X.iloc[:,0], 20, density=True, facecolor='g', alpha=0.75)
            plots[0,0].set_yticks([])
            plots[0,0].xaxis.tick_top()
            plots[0,0].tick_params(axis='x', labelsize= 7)

        for i in range(self.n_variables):

            if i > 0:

                if self.type[i] == "C":

                    sorter = self.values[i]
                    sorterIndex = dict(zip(sorter, range(len(sorter))))

                    new = X.iloc[:,i].value_counts().rename_axis('unique_values').reset_index(name='counts')
                    new["Rank"] = new["unique_values"].map(sorterIndex)
                    new.sort_values('Rank',inplace=True)
                    new.drop('Rank', 1, inplace = True)
                    new = new.set_index('unique_values')

                    new['counts'].plot.barh(ax=plots[i,i], facecolor='g')
                    plots[i,i].yaxis.tick_right()
                    plots[i,i].tick_params(axis='y', labelsize= 7/len(self.type[i]))
                    plots[i,i].set_ylabel("")

                else:
                    plots[i,i].hist(X.iloc[:,i], 20, density=True, facecolor='g', alpha=0.75,orientation='horizontal')
                    plots[i,i].yaxis.tick_right()
                    plots[i,i].tick_params(axis='y', labelsize= 7)

            for j in range(self.n_variables-1,i,-1):

                plots[i,j].axis('off')

                if self.type[i] == "C" or self.type[j] == "C":

                    if self.type[i] == self.type[j]:
                        pass
                    else:
                        if self.type[i] == "C":
                            idx = i
                            idx2 = j
                            vert=True
                        else:
                            idx = j
                            idx2 = i
                            vert=False

                        data = []
                        for val in self.values[idx]:
                            data.append(X.iloc[:,idx2].loc[X.iloc[:,idx]==val])

                        plots[j,i].boxplot(data,vert=vert,\
                        flierprops = dict(marker='o', markerfacecolor='green', markersize=0.1,markeredgecolor='green'),labels=self.values[idx])

                else:
                    try:
                        plots[j,i].tricontourf(X.iloc[:,i],X.iloc[:,j], Y, 10, cmap="Greys_r")
                    except:
                        print("Triangularisation failed")
                    plots[j,i].scatter(X.iloc[:,i],X.iloc[:,j],c=Y,s=s, alpha=0.4, cmap='plasma_r')

                if i == 0:
                    plots[j,i].set_ylabel(self.label[j])


                if j == self.n_variables - 1:
                    plots[j,i].set_xlabel(self.label[i])

                plots[j,i].set_xticks([])
                plots[j,i].set_yticks([])

        plt.subplots_adjust(left=0.050, bottom=0.050, right=0.980, top=0.980, wspace=0, hspace=0)
        plt.show()


        argmin = np.argmin(Y)
        print("Best individual")
        print(X.iloc[argmin,:])
        print(np.array(Y)[argmin])

        for i in range(self.n_variables):
            if self.type[i] =="C":
                inter = X[self.label[i]].astype('category').cat.codes
                X.drop(self.label[i],axis=1)
                X[self.label[i]] = (inter-inter.min())/(inter.max()-inter.min())
            else:
                X[self.label[i]] = (X[self.label[i]]-self.values[i][0])/(self.values[i][1]-self.values[i][0])

        dataf= X.iloc[:,:self.n_variables]
        dataf["loss_value"] = Y
        parallel_coordinates(dataf, "loss_value", colormap="viridis_r")
        plt.show()
        plt.savefig("search_space.png")
