
# coding: utf-8

# In[1]:

import pandas as pd

index = 0
gtlst = []
imgLlst = []
imgRlst = []
for num in range(194):
    dir_name = '000{}'.format(num)
    if len(dir_name) ==4 :
        dir_name = '00'+dir_name
    elif len(dir_name) == 5:
        dir_name = '0'+dir_name
    gt = './disp_noc/'+dir_name+'_10.png'.format(num)
    imgL = './colored_0/'+dir_name+'_10.png'.format(num)
    imgR ='./colored_0/'+dir_name+'_11.png'.format(num)
    gtlst.append((index,index,gt))
    imgLlst.append((index,index,imgL))
    imgRlst.append((index,index,imgR))
    index+=1

gt_df = pd.DataFrame(gtlst)
l_df = pd.DataFrame(imgLlst)
r_df = pd.DataFrame(imgRlst)

gt_df.to_csv('dis.lst',sep='\t',index=False,header=False)
l_df.to_csv('left.lst',sep ='\t',index=False,header = False)
r_df.to_csv('right.lst',sep ='\t',index=False,header = False)

