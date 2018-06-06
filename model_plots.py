import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# From https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/

def range01(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

class model_plots(object):
    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, model, feature_data = [], label_data = [], description = [], nbins = 10, seed = 999, colors = []):
        """Return a Customer object whose name is *name*.""" 
        self.feature_data = feature_data
        self.label_data = label_data
        self.description = description
        self.model = model
        self.nbins = nbins
        self.seed = seed
        self.colors = colors

    def dataprep_modevalplots(self):
        # create an empty output
        final = pd.DataFrame()
        # loop over different files
        for i in range(0,len(self.description)):
            # real target
            y_true = self.label_data[i]
            y_true = y_true.rename('target')
            # probabilities and rename them
            y_pred = self.model.predict_proba(self.feature_data[i])
            probabilities = pd.DataFrame(data=y_pred, index=self.feature_data[i].index)
            probabilities.columns = 'prob_' + self.model.classes_
            # combine the datasets
            dataset = pd.concat([self.feature_data[i], probabilities, y_true], axis=1, join_axes=[self.feature_data[i].index])
            dataset['modelset'] = self.description[i]
            # remove the feature columns
            dataset = dataset.drop(list(self.feature_data[i].columns), axis=1)
            # maak decielen
            # loop over de verschillende uitkomsten heen
            n = dataset.shape[0]
            for j in self.model.classes_:
                #! Added small proportion to prevent equal decile bounds and reset to 0-1 range (to prevent probs > 1.0)
                np.random.seed(self.seed)
                prob_plus_smallrandom = range01(dataset[['prob_' + j]] + (np.random.uniform(size=(n, 1))/1000000))
                dataset["dec_" + j] = self.nbins - pd.DataFrame(pd.qcut(prob_plus_smallrandom, self.nbins, labels=False), index=self.feature_data[i].index)
            # append the different datasets
            final = final.append(dataset)
        return final

    def input_modevalplots(self):
        eval_tot = self.dataprep_modevalplots()
        eval_tot['all'] = 1
        eval_t_tot = pd.DataFrame()
        for i in self.model.classes_:
            for j in self.description:
                eval_t_agg = []
                eval_t_agg = pd.DataFrame(index=range(1,11))
                eval_t_agg['modelset'] = j
                eval_t_agg['target_value'] = i
                eval_t_agg['decile'] = range(1,11,1)
                # 1e relvars werkt
                relvars = ['dec_%s' % i,'all']
                eval_t_agg['tot']=eval_tot[eval_tot.modelset==j][relvars].groupby(('dec_%s' % i)).agg('sum')
                #print(i)
                #print(j)
                # 2e relvars
                eval_tot['pos'] = eval_tot.target == i
                relvars = ['dec_%s' % i, 'pos']
                #print(relvars)
                eval_t_agg['pos']=eval_tot[eval_tot.modelset==j][relvars].groupby('dec_%s' % i).agg('sum')
                # 3e relvars voor neg
                eval_tot['neg'] = eval_tot.target != i
                relvars = ['dec_%s' % i, 'neg']
                #print(relvars)
                eval_t_agg['neg']=eval_tot[eval_tot.modelset==j][relvars].groupby('dec_%s' % i).agg('sum')
                eval_t_agg['pct']=eval_t_agg.pos/eval_t_agg.tot
                eval_t_agg['cumpos']=eval_t_agg.pos.cumsum()
                eval_t_agg['cumneg']=eval_t_agg.neg.cumsum()
                eval_t_agg['cumtot']=eval_t_agg.tot.cumsum()
                eval_t_agg['cumpct']=eval_t_agg.cumpos/eval_t_agg.cumtot
                eval_t_agg['gain']=eval_t_agg.pos/eval_t_agg.pos.sum()
                eval_t_agg['cumgain']=eval_t_agg.cumpos/eval_t_agg.pos.sum()
                eval_t_agg['gain_ref']=eval_t_agg.decile/10
                eval_t_agg['pct_ref']=eval_t_agg.pos.sum()/eval_t_agg.tot.sum()
                eval_t_agg['gain_opt']= 1.0
                eval_t_agg['gain_opt'][(np.ceil(eval_t_agg.pct_ref.astype(float) * 10) / 10.0) >= eval_t_agg.gain_ref]= \
                (1.0*eval_t_agg.gain_ref)/(np.ceil(eval_t_agg.pct_ref * 10) / 10.0)
                eval_t_agg['lift']=eval_t_agg.pct/(eval_t_agg.pos.sum()/eval_t_agg.tot.sum())
                eval_t_agg['cumlift']=eval_t_agg.cumpct/(eval_t_agg.pos.sum()/eval_t_agg.tot.sum())
                eval_t_agg['cumlift_ref']=1
                eval_t_tot = eval_t_tot.append(eval_t_agg,ignore_index=True)
        return eval_t_tot

    def cumulative_lift_chart(self):
        eval_t_tot = self.input_modevalplots()
        for i in self.model.classes_:
            plt.figure(figsize=(20,10))
            plt.subplot(222)
            # Cumulative lift chart
            for col, j in enumerate(self.description):
                plt.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.cumlift_ref[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)]\
                         , color='black', linestyle='dashed',label='no model')
                plt.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.cumlift[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)], \
                         color=self.colors[col], label='%sset' % j)
            plt.title("Cumulative lift chart on the %s outcome dataset" % i)
            plt.xlabel("Decile (1=10% with highest probability, 10=10% lowest prob.) ")
            plt.ylabel("Cumulative lift (factor better than random)")
            plt.legend(loc='upper right', shadow=False,frameon=False)
            plt.grid(True)
        return plt

    def response_chart(self):
        eval_t_tot = self.input_modevalplots()
        for i in self.model.classes_:
            plt.figure(figsize=(20,10))
            plt.subplot(223)
            # Response chart
            for col, j in enumerate(self.description):
                plt.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.pct_ref[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)]\
                         , color='black', linestyle='dashed',label='no model')
                plt.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.pct[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)]\
                         , color=self.colors[col],label='%sset' % j)
            plt.title("Response chart on the %s outcome dataset" % i)
            plt.xlabel("Decile (1=10% with highest probability, 10=10% lowest prob.) ")
            plt.ylabel("Response (% target observations in decile)")
            plt.ylim(ymin = 0)
            plt.legend(loc='upper right', shadow=False,frameon=False)
            plt.grid(True)
        return plt

    def cumulative_response_chart(self):
        eval_t_tot = self.input_modevalplots()
        for i in self.model.classes_:
            plt.figure(figsize=(20,10))
            plt.subplot(224)
            # Cumulative response chart
            for col, j in enumerate(self.description):
                plt.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.pct_ref[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)], \
                         color='black', linestyle='dashed',label='no model')
                plt.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.cumpct[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)], \
                         color=self.colors[col],label='%sset' % j)
            plt.title("Cumulative Response chart on the %s outcome dataset" % i)
            plt.xlabel("Decile (1=10% with highest probability, 10=10% lowest prob.) ")
            plt.ylabel("Cumulative % target obs, until decile")
            plt.ylim(ymin = 0)
            plt.legend(loc='upper right', shadow=False,frameon=False)
            plt.grid(True)
        return plt

    def cumulative_gains_chart(self):
        eval_t_tot = self.input_modevalplots()
        for i in self.model.classes_:
            plt.figure(figsize=(20,10))
            plt.subplot(221)
            # Cumulative Gains chart
            add_origin = pd.DataFrame()
            for col, j in enumerate(self.description):
                for tv in self.model.classes_:
                    add_origin_add = pd.DataFrame({'modelset':[j],'target_value':[tv],'decile':[0],'tot':[0],'pos':[0],'neg':[0],'pct':[0],
                                                   'cumpos':[0],'cumneg':[0],'cumtot':[0],'cumpct':[0],'gain':[0],'cumgain':[0],'gain_ref':[0],
                                                   'pct_ref':[0],'gain_opt':[0],'lift':[None],'cumlift':[None],'cumlift_ref':[1]})
                    add_origin = pd.concat([add_origin,add_origin_add],axis=0)
                    eval_t_tot_origin = pd.concat([add_origin,eval_t_tot],axis=0)
                    eval_t_tot_origin = eval_t_tot_origin.sort_values(['target_value','modelset','decile'])
                    eval_t_tot_origin = eval_t_tot_origin.set_index(keys=['target_value','modelset','decile'],drop = False)
                plt.plot(eval_t_tot_origin.decile[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)],\
                             eval_t_tot_origin.gain_ref[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)], \
                             color='black', linestyle='dashed',label='no model')
                plt.plot(eval_t_tot_origin.decile[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)],\
                             eval_t_tot_origin.gain_opt[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)], \
                             color='black', linestyle='dashed',label='optimal')
                plt.plot(eval_t_tot_origin.decile[(eval_t_tot_origin.modelset==j)&(eval_t_tot_origin.target_value==i)],\
                             eval_t_tot_origin.cumgain[(eval_t_tot_origin.modelset==j)&(eval_t_tot_origin.target_value==i)], \
                             color=self.colors[col],label='%sset' % j)
                plt.title("Cumulative gains on the %s outcome dataset" % i)
                plt.xlabel("decile (1=10% with highest probability, 10=10% lowest prob.) ")
                plt.ylabel("gains (cum. % of all target observations)")
                plt.xticks(np.arange(0, 11, 1))
                plt.grid(True)
                plt.legend(loc='lower right', shadow=False,frameon=False)
        return plt

    def multiplot(self):
        eval_t_tot = self.input_modevalplots()
        for i in self.model.classes_:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(15,10))
            fig.suptitle("Target %s in dataset" % i, fontsize=18)
    
            ax1.title.set_text('Cumulative gains chart')
            ax1.set_ylabel('Gains (cum. % of all target observations)')
            ax1.set_xlabel('Decile (1=10% with highest probability, 10=10% lowest prob.)')
            ax1.set_ylim(0, 1)
            ax1.set_xlim(0, 10)
            ax1.set_xticks(np.arange(0, 11, 1))
            ax1.grid(True)
    
            ax2.title.set_text('Cumulative lift chart')
            ax2.set_ylabel('Cumulative lift (factor better than random)')
            ax2.set_xlabel('Decile (1=10% with highest probability, 10=10% lowest prob.)')
            ax2.set_xlim(1, 10)
            ax2.grid(True)
    
            ax3.title.set_text('Response chart')
            ax3.set_ylabel('Response (% target observations in decile)')
            ax3.set_xlabel('Decile (1=10% with highest probability, 10=10% lowest prob.)')
            ax3.set_xlim(1, 10)
            ax3.set_ylim(0, 1)
            ax3.grid(True)
    
            ax4.title.set_text('Cumulative response chart')
            ax4.set_ylabel('Cumulative % target obs, until decile')
            ax4.set_xlabel('Decile (1=10% with highest probability, 10=10% lowest prob.)')
            ax4.set_xlim(1, 10)
            ax4.set_ylim(0, 1)
            ax4.grid(True)
    
            for col, j in enumerate(self.description):
                # cumulative gains
                add_origin = pd.DataFrame()
                for tv in self.model.classes_:
                    add_origin_add = pd.DataFrame({'modelset':[j],'target_value':[tv],'decile':[0],'tot':[0],'pos':[0],'neg':[0],'pct':[0],
                                                   'cumpos':[0],'cumneg':[0],'cumtot':[0],'cumpct':[0],'gain':[0],'cumgain':[0],'gain_ref':[0],
                                                   'pct_ref':[0],'gain_opt':[0],'lift':[None],'cumlift':[None],'cumlift_ref':[1]})
                    add_origin = pd.concat([add_origin,add_origin_add],axis=0)
                    eval_t_tot_origin = pd.concat([add_origin,eval_t_tot],axis=0)
                    eval_t_tot_origin = eval_t_tot_origin.sort_values(['target_value','modelset','decile'])
                    eval_t_tot_origin = eval_t_tot_origin.set_index(keys=['target_value','modelset','decile'],drop = False)
                ax1.plot(eval_t_tot_origin.decile[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)],\
                         eval_t_tot_origin.gain_ref[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)], \
                         color='black', linestyle='dashed',label='no model')
                ax1.plot(eval_t_tot_origin.decile[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)],\
                         eval_t_tot_origin.gain_opt[(eval_t_tot_origin.modelset==self.description[col])&(eval_t_tot_origin.target_value==i)], \
                         color='black', linestyle='dashed',label='optimal')
                ax1.plot(eval_t_tot_origin.decile[(eval_t_tot_origin.modelset==j)&(eval_t_tot_origin.target_value==i)],\
                         eval_t_tot_origin.cumgain[(eval_t_tot_origin.modelset==j)&(eval_t_tot_origin.target_value==i)], \
                         color=self.colors[col],label='%sset' % j)
                ax1.legend(loc='lower right', shadow=False, frameon=False, fontsize='large')
                
                # cumulative lift
                ax2.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.cumlift_ref[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)]\
                         , color='black', linestyle='dashed',label='no model')
                ax2.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.cumlift[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)], \
                         color=self.colors[col], label='%sset' % j)
                ax2.legend(loc='upper right', shadow=False, frameon=False, fontsize='large')
                
                # response
                ax3.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.pct_ref[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)]\
                         , color='black', linestyle='dashed',label='no model')
                ax3.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.pct[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)]\
                         , color=self.colors[col],label='%sset' % j)
                ax3.legend(loc='upper right', shadow=False, frameon=False, fontsize='large')
                
                # cumulative response
                ax4.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.pct_ref[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)], \
                         color='black', linestyle='dashed',label='no model')
                ax4.plot(eval_t_tot.decile[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)],\
                         eval_t_tot.cumpct[(eval_t_tot.modelset==j)&(eval_t_tot.target_value==i)], \
                         color=self.colors[col],label='%sset' % j)
                ax4.legend(loc='upper right', shadow=False, frameon=False, fontsize='large')
        return plt