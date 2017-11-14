

import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import copy
from matplotlib.font_manager import FontProperties
# rc('font',**{'family':'sans-serif','sans-serif':['Palatino']})
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Palatino']
    



def plot_attention(attentions,input_sentence,output_sentence,filename='./'):

        
#         if attentions!=[]:

#             print(attentions[0][0])
            attentions = attentions[0][0].cpu().numpy()
#             print(attentions)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            row_sums = attentions.sum(axis=1)
#             attentions = attentions / row_sums[:, np.newaxis]
#             attentions = attentions / attentions.sum(axis=0)
#             attentions = attentions / attentions.max(axis=0)            
#             attentions = attentions.sum(axis=1)
        
            
            cax = ax.matshow(attentions/1.0, cmap='Blues')
#             print(attentions)
        #         fig.colorbar(cax)
        
            # Set up axes
            ax.set_yticklabels(['']+ output_sentence+ ['<EOS>'])
            ax.set_xticklabels(['']+input_sentence.split() +['<EOS>',''], rotation=75)
            
            # Show label at every tick
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            
            for (i, j), z in np.ndenumerate(attentions):
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',fontsize=3)
        
        #         plt.show()
            plt.xlabel('Lyric')
            plt.ylabel('Annotation')
            plt.tight_layout()
            plt.savefig('../attention_plots/newplots/'+str(filename)+'.png',dpi=240)
            plt.close()
#             plt.savefig('./plot.png',dpi=240)            
#             if 'ak-47' in input_sentence:
#                 plt.savefig('plots/'+input_sentence+'.eps')            
                
        
        
    
