import csv
from visdom import Visdom
viz = Visdom(port=8850)
import sys
import numpy as np
from scipy.spatial.distance import directed_hausdorff
sys.path.append("..")
sys.path.append(".")

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def hd(u,v):
    a=directed_hausdorff(u, v)[0]
    b=directed_hausdorff(u, v)[0]
    c=max(a,b)
    print('abc', a,b,c)
    return hd95
f=.4308
v=0.6225
auto = 0.5649
cf=[f,f,f,f,f]
cv=[v,v,v,v,v]
cauto = [auto,auto,auto,auto,auto]
x=[0.3,0.4, 0.5,0.6,0.7,0.8]
x2=[0.4, 0.5,0.6,0.7]
                                    # L=200

y0=[0.5356, 0.5137245  , 0.5145 ,0.5155 , 0.5173, 0.51993]   #L=100
y1=[0.50263, 0.519257, 0.5899, 0.5843 , 0.56977, 0.51289]   #L=200

y2=[0.440652, 0.54095,  0.55286 , 0.56019, 0.56458 , 0.561486]#L=300

y3=[ 0.39640,0.4805, 0.53245, 0.541263, 0.54702, 0.5531150 ]#L=400


import tikzplotlib

import matplotlib.pyplot as plt
import os
plt.figure(1)
plt.plot(x, y1,color='blue', linestyle='-',marker='o', label='L=200')

plt.plot(x, y3, color='forestgreen', linestyle='-',marker='x', label='L=400')
plt.plot(x, y2, color='darkred', linestyle='-',marker='D', label='L=300')
plt.plot(x, y0, color='orange', linestyle='-',marker='v', label='L=100')


#plt.plot(x, cf,color='darkorange', linestyle='--',  label='LDM')

#plt.plot(x, cv,color='lightpink', linestyle='--', label='AnoDDPM')
#plt.plot(x, cauto, color='y', linestyle='--', label='AutoDDPM')

#plt.title('Dice Score')
plt.xlabel('Threshold value P for the histogram masking')
plt.ylabel('average Dice score')
#plt.hlines(0.693, 0, 750, colors='g', linestyles='dashed', label='FP-GAN')
#plt.hlines(0.222, 0, 750, colors=None, linestyles='dashed', label='VAE')
plt.legend(loc='center right')
viz.matplot(plt)
plt.savefig('./dice.png')
tikzplotlib.save("./dice.tex")


H=[]
D=[]
D2=[]
H2=[]

D3=[]
H3=[]
scale=100/131072
# h3 = open("./dummylist_healthy")
# list_of_column_names = []
# for row in h3:
#     list_of_column_names.append(row)
# x = list_of_column_names[0].split(",")
# for i in range(len(x)):
#     value =x[i]
#     H3.append(np.array(value).astype(float))
#
# d3 = open("./dummylist_diseased")
# list_of_column_names = []
# for row in d3:
#         list_of_column_names.append(row)
# x = list_of_column_names[0].split(",")
# for i in range(len(x)):
#         value = x[i]
#         D3.append(np.array(value).astype(float))
#
# data = [H3, D3]
#
# plt.figure(2)
# plt.title("Number of voxels with a flipping probability higher than p")
# labels = ['h_200_0.7', 'd_200_0.7']
#
# plt.boxplot(data, labels=labels)
# viz.matplot(plt)
# plt.savefig('./boxplot2.png')

# loop to iterate through the rows of csv


healthy_data = open("./healthylist_300_0.5")
list_of_column_names = []

# loop to iterate through the rows of csv
for row in healthy_data:
    # adding the first row
    list_of_column_names.append(row)

x = list_of_column_names[0].split(",")

for i in range(len(x)):
    value =x[i]

    H.append(np.array(value).astype(float)*scale)


healthy_data2 = open("./healthylist_200_0.5")
list_of_column_names = []

# loop to iterate through the rows of csv
for row in healthy_data2:
    # adding the first row
    list_of_column_names.append(row)

x = list_of_column_names[0].split(",")

for i in range(len(x)):
    value =x[i]
    H2.append(np.array(value).astype(float)*scale)

H3=[]
healthy_data3= open("./healthylist_200_0.7")
list_of_column_names = []

# loop to iterate through the rows of csv
for row in healthy_data3:
    # adding the first row
    list_of_column_names.append(row)

x = list_of_column_names[0].split(",")

for i in range(len(x)):
    value =x[i]
    H3.append(np.array(value).astype(float)*scale)





PathDicomstripped = "./Reconstruction_300_0.5/"
  #  PathDicomstripped = "scaling" + str(scale) + "_ddim1000_t500/"

for dirName, subdirList, fileList in os.walk(PathDicomstripped):#used to be dicomstripped
      s = dirName.split("/", -1)

#     if 't1n_3d' in subdirList:
#         path=os.path.join(dirName, 't1n_3d'))
      for filename in fileList:
            s = filename.split("(", 1)
            number=s[1]
            i+=1
            s2 = number.split(".", 1)
            D.append(np.array( s2[0]).astype(float)*scale)



PathDicomstripped2 = "./Reconstruction_200_0.5/"
  #  PathDicomstripped = "scaling" + str(scale) + "_ddim1000_t500/"

for dirName, subdirList, fileList in os.walk(PathDicomstripped2):#used to be dicomstripped
      s = dirName.split("/", -1)

#     if 't1n_3d' in subdirList:
#         path=os.path.join(dirName, 't1n_3d'))
      for filename in fileList:
            s = filename.split("(", 1)
            number=s[1]
            i+=1
            s2 = number.split(".", 1)
            D2.append(np.array( s2[0]).astype(float)*scale)


D3=[]


PathDicomstripped2 = "./Reconstruction_200_0.7/"
  #  PathDicomstripped = "scaling" + str(scale) + "_ddim1000_t500/"

for dirName, subdirList, fileList in os.walk(PathDicomstripped2):#used to be dicomstripped
      s = dirName.split("/", -1)

#     if 't1n_3d' in subdirList:
#         path=os.path.join(dirName, 't1n_3d'))
      for filename in fileList:
            s = filename.split("(", 1)
            number=s[1]
            i+=1
            s2 = number.split(".", 1)
            D3.append(np.array( s2[0]).astype(float)*scale)
plt.figure(3)
fig = plt.figure(figsize=(10, 7))

# Creating plot

print('shape', len(H),len(H2),len(H3))
data = [H,D,H2, D2, H3, D3]
#ax = fig.add_axes([0, 0, 1, 1])# x-axis labels



#data = [H,D,H2, D2, H3, D3]

# H=H*scale
# H2=H2*scale
# H3=H3*scale
# D=D*scale
# D2=D2*scale
# D3=D3*scale
data = [H, H2, H3, D, D2,  D3]
plt.figure(4)



# option 1, specify props dictionaries
c = "darkblue"
box1=plt.boxplot(data[ :3], positions=[1, 2, 3], notch=True, patch_artist=True,
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            )

# option 2, set all colors individually
c2 = "darkorange"
box2 = plt.boxplot(data[3:] , positions=[1.5, 2.5, 3.5], notch=True, patch_artist=True)
for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
    plt.setp(box2[item], color=c2)
plt.setp(box2["boxes"], facecolor=c2)
plt.setp(box2["fliers"], markeredgecolor=c2)
plt.legend([box1["boxes"][0], box2["boxes"][0]], ['healthy', 'diseased'], loc='upper right')

labels = ["Jan\n2009", "Feb\n2009", "Mar\n2009", "Apr\n2009",
          "May\n2009"]


plt.xlim(0.5, 4)
plt.xticks([1.25, 2.25, 3.25], ["P=0.5\nL=300", "P=0.5\n L=200", "P=0.7\nL=200"])
plt.ticklabel_format(axis='y', scilimits=[-3, 3])
#plt.title('Summed mask')
#plt.xlabel('histogram masking')
plt.ylabel('Percentage of masked voxels')

#plt.show()

plt.savefig('./boxplot3.png')
tikzplotlib.save("./boxplot3.tex")
viz.matplot(plt)