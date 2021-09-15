from pulp import *
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import datetime
from dateutil.relativedelta import *
import xlwings as xw
from collections import Counter
import ctypes
import os
import math
class Entity(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """ 
		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # constructor  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self,name, verbose=False, fx_position = False):	  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        Constructor method  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
        """  		  	   		   	 			
        self.entity = defaultdict(int)
        self.entity['Product'] = {}
        self.entity['FX'] = {}
        self.entity['name'] = name
        self.cur_lst = []
        
        self.var_bal = {}
        self.var_flow_in = {}
        self.var_flow_out = {}
        self.var_boo = {}
        self.var_spot_bal = {}
        self.var_swap_bal = {}
        self.var_flow_spot_in = {}
        self.var_flow_swap_in = {}
        self.var_flow_swap_out = {}
        self.var_fx_bal = {}
        self.var_fx_flow_in  = {}
        self.fx_gl_var = []
        self.out_mat = []
        self.obj = ''
        self.data = pd.DataFrame(data = []) #product dataframe
        self.fx_position = fx_position #if this entity represent fx balance sheet positions/ should only include balance & flows of foreign currencies.
        
        #self.date_lst = {}

    def construct_bal_cf(self,df):
        '''
        input: balance with columns: ['Date', currencies, currencies+_in or _out]
        output: {'Balance': {currencies: daily_balance},'Flow:{currencies:daily_flows}}
        '''
        self.cur_lst = [i for i in df.columns if all([i != 'Date', '_' not in i])]
        daily_bal = {df.loc[k,'Date']:{i:df.loc[k,i] for i in self.cur_lst} for k in range(len(df.index))}
        daily_flow = {df.loc[k,'Date']:{i:df.loc[k,i.lower()+'_in']+df.loc[k,i.lower()+'_out'] \
                                        for i in self.cur_lst} for k in range(len(df.index))}
        #return {'Balance':daily_bal,'Flow':daily_flow}
        self.entity['Balance'] = daily_bal
        self.entity['Flow'] = daily_flow
        
        
    def construct_fx_vars(self,fx, in_bal_days, balance_daily, balance, week_index,by_day = False):

        '''

        input:

        1. fx dataframe with Date, type (spot or swap), convertible currecies - from & to, GL (hedging G/L), Tenor

        2. in_bal_days: actual tenor considering holidays & weekend

        3. balance_daily --> dataframe with a column Date for list of dates

        output:



        {var_keys: list of variable keys, rate: np array (interest rate by day) , tenor: np array , fx_info: store fx product info for later use}

        '''
        # assert sum([i not in self.cur_lst for i in list(set(fx['from']))]) == 0,'currency in fx table not in cur_lst'
        # assert sum([i not in self.cur_lst for i in list(set(fx['to']))]) == 0,'currency in fx table not in cur_lst'

        df_fx_product = fx[['type','from','to','Tenor']].drop_duplicates()

        df_fx_product = df_fx_product.reset_index()

        m = len(df_fx_product.index)

        n = len(balance_daily['Date'])

        fx_tenor = np.zeros((m,n))

        fx_rate_mat = np.zeros((m,n))

        for i in range(len(df_fx_product.index)):

            tenor = df_fx_product.loc[i,'Tenor']

            date_series = df_fx_product.loc[[i],:].merge(fx, on = ['type','from','to','Tenor']).merge(balance_daily[['Date']], on = ['Date'], how = 'outer')

            date_series = date_series.sort_values(['Date'], ascending = True).reset_index()

            date_series['GL'] = np.array(date_series['GL'].ffill())

            date_series['GL'] = np.array(date_series['GL'].fillna(-100))

            #date_series['calc_tenor'] = [min(max(date_series.index)-i+1, tenor) for i in date_series.index]



            #saving:假日算進去，TD 類要變成實際天期

            if tenor == 0:

                date_series['calc_tenor'] = in_bal_days.copy()

            else:

                remaining_days = list(date_series['Date'].apply(lambda x: (max(date_series['Date'])-x).days+1))

                date_series['calc_tenor'] = [min(i,tenor) for i in remaining_days]



            if tenor == 0:

                fx_rate_mat[i,:] = np.array(date_series['GL']/360)

            else:

                #fx_rate_mat[i,:] = np.array(date_series['GL']*tenor/360)

                fx_rate_mat[i,:] = np.array(date_series['GL'])*np.array(date_series['calc_tenor'])/360

            fx_tenor[i,:] = np.array(date_series['calc_tenor'])

        ###########################################################

        if by_day == False:

            for i in range(len(week_index)):

                if i == 0:

                    int_index = np.argmax(fx_rate_mat[:,0:week_index[i]+1], axis = 1)

                    #int_rate_mat[:,0:week_index[i]+1]

                else:

                    int_index = np.vstack((int_index,np.argmax(fx_rate_mat[:,week_index[i-1]+1:week_index[i]+1], axis = 1)+week_index[i-1]+1))

                    #int_rate_mat[:,week_index[i-1]+1:week_index[i]+1]

            int_index = int_index.T



            #construct new int_rate_mat and td_tenor from the old ones.

            for c in range(int_index.shape[0]):

                if c == 0:

                    new_fx_rate_mat = fx_rate_mat[c,int_index[c,:]]

                    new_fx_tenor = fx_tenor[c,int_index[c,:]]

                else:

                    new_fx_rate_mat = np.vstack((new_fx_rate_mat,fx_rate_mat[c,int_index[c,:]]))

                    new_fx_tenor = np.vstack((new_fx_tenor,fx_tenor[c,int_index[c,:]]))

        else:

            new_fx_rate_mat = fx_rate_mat.copy()

            new_fx_tenor = fx_tenor.copy()

        ###########################################################



        fx_info = defaultdict(int)

        for row in range(len(df_fx_product.index)):

            k = '_'.join([str(df_fx_product.loc[row,col]) for col in ['type','from','to','Tenor']])

            fx_info[k] = [df_fx_product.loc[row,col] for col in ['type','from','to','Tenor']]

        #Spot 先用flow 算如果要加hedging cost 就要加天期

        spot_bal = []

        flow_spot_in = []

        swap_bal = []

        flow_swap_in = []

        flow_swap_out = []

        objective_basis_fx = []

        for k in fx_info.keys():

            for date in balance['Date']:

                if 'spot' in k:

                    spot_bal.append('{}_{}'.format(k,date))

                    flow_spot_in.append('{}_{}_in'.format(k,date))

                    objective_basis_fx.append('{}_{}_in'.format(k,date))

                else:

                    swap_bal.append('{}_{}'.format(k,date))

                    flow_swap_in.append('{}_{}_in'.format(k,date))

                    flow_swap_out.append('{}_{}_out'.format(k,date))

                    objective_basis_fx.append('{}_{}_in'.format(k,date))

        self.entity['FX'] =  {'objective_basis':objective_basis_fx,'var_keys_bal':[spot_bal,swap_bal],\

                'var_keys_flow':[flow_spot_in,flow_swap_in,flow_swap_out],\

                'rate':new_fx_rate_mat, 'tenor':new_fx_tenor,'all_rate':fx_rate_mat, 'all_tenor': fx_tenor,'fx_info':fx_info}

    def construct_product_vars(self,data, in_bal_days, balance_daily,balance, rate_path, week_index,by_day = False):

        '''

        input:

        1. fx dataframe with Date, type (spot or swap), convertible currecies - from & to, GL (hedging G/L), Tenor

        2. in_bal_days: actual tenor considering holidays & weekend

        3. balance_daily --> dataframe with a column Date for list of dates

        4. rate_path: rate rise/fall pattern over time by currency

        output:



        {var_keys: list of variable keys, rate: np array (interest rate by day) , tenor: np array }

        '''

        #Interest rate matrix logic: adjusted for the tenor

        #
        self.data = data.copy()

        df_product = data[['Bank','Curr','product','Tenor','Company']].drop_duplicates()

        #

        df_product = df_product.reset_index()

        m = len(df_product.index)

        n = len(balance_daily['Date'])

        td_tenor = np.zeros((m,n))

        int_rate_mat = np.zeros((m,n))

        for i in range(len(df_product.index)):

            tenor = df_product.loc[i,'Tenor']

            cur = df_product.loc[i,'Curr']

            bk = df_product.loc[i,'Bank']

            #

            comp = self.entity['name']

            #

            date_series = df_product.loc[[i],:].merge(data, on = ['Bank','Curr','product','Company']).merge(balance_daily[['Date']], on = ['Date'], how = 'outer')

            #

            date_series = date_series.sort_values(['Date'], ascending = True).reset_index()

            date_series[['rate','Daily_max','Daily_min','average']] = date_series[['rate','Daily_max','Daily_min','average']].ffill()

            date_series['rate'] = date_series['rate'].fillna(-100)
            



            ##############

            #Add rate movement

            ##############
            if 'Loan' not in bk:

                date_series['rate'] = date_series['rate']+rate_path[cur]['Rate']
            else:
                date_series['rate'] = date_series['rate']-rate_path[cur]['Rate']

        ##########################################################################################   

            #假日沒有算進去

            #date_series['calc_tenor'] = [min(max(date_series.index)-i+1, tenor) for i in date_series.index]

            #saving:假日算進去，TD 類要變成實際天期(才不會最後一股腦往長天期放)

            if tenor == 0:

                date_series['calc_tenor'] = in_bal_days.copy()

            else:

                remaining_days = list(date_series['Date'].apply(lambda x: (max(date_series['Date'])-x).days+1))

                date_series['calc_tenor'] = [min(i,tenor) for i in remaining_days]



            if cur == 'TWD':

                if tenor == 0:

                    #int_rate_mat[i,:] = np.array(date_series['rate']/365)

                    int_rate_mat[i,:] = np.array(date_series['rate']) * np.array(date_series['calc_tenor'])/365

                    #print(np.array(date_series['rate']) * np.array(date_series['calc_tenor'])/365)

                else:

                    #int_rate_mat[i,:] = np.array(date_series['rate']*tenor/365)

                    int_rate_mat[i,:] = np.array(date_series['rate']) * np.array(date_series['calc_tenor'])/365

            else:

                if tenor == 0:

                    int_rate_mat[i,:] = np.array(date_series['rate']) * np.array(date_series['calc_tenor'])/360

                else:

                    int_rate_mat[i,:] = np.array(date_series['rate']) * np.array(date_series['calc_tenor'])/360

            td_tenor[i,:] = np.array(date_series['calc_tenor'])





    ######################################################################

        #Find the max int rate for each week

        if by_day == False:

            for i in range(len(week_index)):

                if i == 0:
                    # add absolute to avoid obtaining too favorable Loan rates 
                    int_index = np.argmax(abs(int_rate_mat[:,0:week_index[i]+1]), axis = 1)
                else:
                    int_index = np.vstack((int_index,np.argmax(abs(int_rate_mat[:,week_index[i-1]+1:week_index[i]+1]), axis = 1)+week_index[i-1]+1))
                    
            int_index = int_index.T



            #construct new int_rate_mat and td_tenor from the old ones.

            for c in range(int_index.shape[0]):

                if c == 0:

                    new_int_rate_mat = int_rate_mat[c,int_index[c,:]]

                    new_td_tenor = td_tenor[c,int_index[c,:]]

                else:

                    new_int_rate_mat = np.vstack((new_int_rate_mat,int_rate_mat[c,int_index[c,:]]))

                    new_td_tenor = np.vstack((new_td_tenor,td_tenor[c,int_index[c,:]]))

        else:

            new_int_rate_mat = int_rate_mat.copy()

            new_td_tenor = td_tenor.copy()

    #######################################################################

        bal_per_bank = []

        flow_per_bank_in = []

        flow_per_bank_out = []

        objective_basis = []

        for i in range(len(df_product.index)):

            b = df_product.loc[i,'Bank']

            c = df_product.loc[i,'Curr']

            pro = df_product.loc[i,'product']

            tenor = df_product.loc[i,'Tenor']
            
            comp = self.entity['name']

            #

            for date in balance['Date']:

                bal_per_bank.append('{}_{}_{}_{}_{}_bal'.format(comp,b,c,pro,date))

                flow_per_bank_in.append('{}_{}_{}_{}_{}_in'.format(comp,b,c,pro,date))

                flow_per_bank_out.append('{}_{}_{}_{}_{}_out'.format(comp,b,c,pro,date))

                if tenor == 0:

                    objective_basis.append('{}_{}_{}_{}_{}_bal'.format(comp,b,c,pro,date))

                else:

                    objective_basis.append('{}_{}_{}_{}_{}_in'.format(comp,b,c,pro,date))

           #

        #Construct boolean variable for special accounts

        #11/19/2020 to make the program more flexible --> add booleans by month for average max

        boo_lst = []

        boo = data.loc[(data['Daily_min'].isnull() == False) | ((data['average'].isnull() == False) & (data['average'].apply(lambda x: '>' in str(x))))]

        #

        boo = boo[['Bank','Curr','product','Company']].drop_duplicates()

        #

        for i in boo.index:

            b = boo.loc[i,'Bank']

            c = boo.loc[i,'Curr']

            pro = boo.loc[i,'product']

            #

            comp = self.entity['name']

            for m in set([str(i.year)+str(i.month) for i in balance['Date']]):

                boo_lst.append('{}_{}_{}_{}_{}_boo'.format(comp,b,c,pro,m))

            #

        self.entity['Product'] = {'objective_basis':objective_basis,'var_keys_bal':[bal_per_bank,boo_lst],\
                'var_keys_flow':[flow_per_bank_in,flow_per_bank_out],\
                'rate':new_int_rate_mat, 'tenor':new_td_tenor, 'all_rate':int_rate_mat,\
                'all_tenor':td_tenor,'df_product':df_product}

    def var_init(self):

        '''

         return {'objective_basis':objective_basis,'var_keys_bal':[bal_per_bank,boo_lst],\

                'var_keys_flow':[flow_per_bank_in,flow_per_bank_out],\

                'rate':new_int_rate_mat, 'tenor':new_td_tenor, 'all_rate'; int_rate_mat, 'all_tenor':td_tenor}

        '''

        bal_per_bank = self.entity['Product']['var_keys_bal'][0]

        boo_lst = self.entity['Product']['var_keys_bal'][1]

        flow_per_bank_in = self.entity['Product']['var_keys_flow'][0]

        flow_per_bank_out = self.entity['Product']['var_keys_flow'][1]


        objective_basis = self.entity['Product']['objective_basis']

        new_int_rate_mat = self.entity['Product']['rate']

        



        self.var_bal = {i:j for i,j in zip(bal_per_bank,\

                                      LpVariable.matrix(self.entity['name']+"Balance",list(range(len(bal_per_bank))),lowBound=0, upBound=None,cat='Continuous'))}

        self.var_flow_in = {i:j for i,j in zip(flow_per_bank_in,\

                                          LpVariable.matrix(self.entity['name']+"Flow_in",list(range(len(flow_per_bank_in))),0, None,cat='Continuous'))}

        self.var_flow_out = {i:j for i,j in zip(flow_per_bank_out,\

                                           LpVariable.matrix(self.entity['name']+"Flow_out",list(range(len(flow_per_bank_out))),None, 0,cat='Continuous'))}

        ######################################

        #如果solver跑很久，可能是因為branch and bound 找不到答案，把boolean variables 拿掉，跑LP，會快些結束，再找哪個constraint 出錯

        ######################################

        self.var_boo = LpVariable.dicts("Boolean",boo_lst,0, 1, LpInteger)
        
        fx_vec = []
        
        if len([i for i in self.entity['FX'].keys()]) != 0:

            spot_bal = self.entity['FX']['var_keys_bal'][0]

            swap_bal = self.entity['FX']['var_keys_bal'][1]

            flow_spot_in = self.entity['FX']['var_keys_flow'][0]

            flow_swap_in = self.entity['FX']['var_keys_flow'][1]

            flow_swap_out = self.entity['FX']['var_keys_flow'][2]

            objective_basis_fx = self.entity['FX']['objective_basis']

            new_fx_rate_mat = self.entity['FX']['rate']
            
            self.var_spot_bal = {i:j for i,j in zip(spot_bal,\

                                               LpVariable.matrix(self.entity['name']+"Balance_spot",list(range(len(spot_bal))),0, None,cat='Continuous'))}

            self.var_swap_bal = {i:j for i,j in zip(swap_bal,\

                                               LpVariable.matrix(self.entity['name']+"Balance_swap",list(range(len(swap_bal))),0, None,cat='Continuous'))}

            self.var_flow_spot_in = {i:j for i,j in zip(flow_spot_in,\

                                                   LpVariable.matrix(self.entity['name']+"Flow_spot_in",list(range(len(flow_spot_in))),0, None,cat='Continuous'))}

            self.var_flow_swap_in = {i:j for i,j in zip(flow_swap_in,\

                                                   LpVariable.matrix(self.entity['name']+"Flow_swap_in",list(range(len(flow_swap_in))),0, None,cat='Continuous'))}

            self.var_flow_swap_out = {i:j for i,j in zip(flow_swap_out,\

                                                    LpVariable.matrix(self.entity['name']+"Flow_swap_out",list(range(len(flow_swap_out))),None, 0,cat='Continuous'))}

            self.var_fx_bal = dict(self.var_spot_bal, **self.var_swap_bal)

            self.var_fx_flow_in = dict(self.var_flow_spot_in, **self.var_flow_swap_in)

            self.fx_gl_var = [self.var_fx_flow_in[i] for i in objective_basis_fx]
            
            fx_vec = new_fx_rate_mat.reshape(np.array(self.fx_gl_var).shape, order = 'C')



        for i in objective_basis:

            if len(re.findall('_bal$',i)) > 0:

                self.out_mat.append(self.var_bal[i])

            else:

                self.out_mat.append(self.var_flow_in[i])



        self.out_mat = np.array(self.out_mat)

        int_vec = new_int_rate_mat.reshape(self.out_mat.shape,order = 'C')

        #################################################################################

        self.obj = LpAffineExpression([(self.out_mat[i],int_vec[i]) for i in range(len(self.out_mat))])+ \
        LpAffineExpression([(self.fx_gl_var[i],fx_vec[i]) for i in range(len(self.fx_gl_var))])

    #     LpAffineExpression([(s,-2) for _,s in var_s.items()]),"Total interest income"

    def bal_constraint(self, prob, intercomp_constraint = {},fx_pos_constraints = []):
        #sum(bank balances) = sum(balance excluding loan)+loan balance -fx_out +fx_in
        #company 之間只有intercompany loan 有流通

        comp = self.entity['name']
        daily_bal = self.entity['Balance']
        date_lst = [i for i in self.entity['Balance'].keys()]
        d_prior = min(date_lst)
        fx_info = {}
        if len([i for i in self.entity['FX'].keys()]) != 0:
            fx_info = self.entity['FX']['fx_info']

            #print(d)
        for cur in self.cur_lst:
            #print(cur)
            if cur in intercomp_constraint.keys():
                intercomp = True
            else:
                intercomp = False
            for d in daily_bal.keys():
                #print(d)
                if self.fx_position == False:
                    fx_out = [self.var_fx_bal['_'.join([i,str(d)])] for i,j in fx_info.items() if j[1] == cur]
                    fx_in = [self.var_fx_bal['_'.join([i,str(d)])] for i,j in fx_info.items() if j[2] == cur]
                    fx_out = LpAffineExpression([(fx_out[i],1) for i in range(len(fx_out))])
                    fx_in = LpAffineExpression([(fx_in[i],1) for i in range(len(fx_in))])
                    #adding spot & forward that impact balance sheet position  
                    if len(fx_pos_constraints) != 0:
                        assert len([i for _,i in fx_info.items() if i[0] == 'spot']) == 0, 'spot should be a variable in fx position entity!'
                        fx_pos_info = fx_pos_constraints[0]
                        #fx_pos_spot_in = fx_pos_constraints[1]
                        fx_pos_fwd_out = fx_pos_constraints[2]
                        fx_pos_spot_bal = fx_pos_constraints[3]

                        for t,_ in {i+'_':j[3] for i,j in fx_pos_info.items() if all([j[0] != 'spot', j[1] == cur])}.items():

                            lst_out = [i for i,j in fx_pos_fwd_out.items() if t in i]
                            boo_out = [self.within_date_range(i,d,d_prior) for i in lst_out]*np.array([-1]*len(lst_out))
                            fx_out += lpSum([fx_pos_fwd_out[i] for i in lst_out]*boo_out)
                        #to 
                        for t,_ in {i+'_':j[3] for i,j in fx_pos_info.items() if all([j[0] !='spot', j[2] == cur])}.items():

                            lst_out = [i for i,j in fx_pos_fwd_out.items() if t in i]
                            boo_out = [self.within_date_range(i,d,d_prior) for i in lst_out]*np.array([-1]*len(lst_out))
                            fx_in += lpSum([fx_pos_fwd_out[i] for i in lst_out]*boo_out)

                        #for spot
                        spot_out = [fx_pos_spot_bal['_'.join([i,str(d)])] for i,j in fx_pos_info.items() if all([j[1] ==cur, j[0] == 'spot'])]
                        spot_in = [fx_pos_spot_bal['_'.join([i,str(d)])] for i,j in fx_pos_info.items() if all([j[2] == cur, j[0] == 'spot'])]
                        fx_out += LpAffineExpression([(spot_out[i],1) for i in range(len(spot_out))])
                        fx_in += LpAffineExpression([(spot_in[i],1) for i in range(len(spot_in))])

                    if intercomp == True:
                        prob += LpAffineExpression([(self.var_bal[i],1) for i in self.var_bal.keys() if all([comp in i,'{}'.format(d) in i, cur in i, 'Loan' not in i])])-\
                        LpAffineExpression([(self.var_bal[i],1) for i in self.var_bal.keys() if all([comp in i,'{}'.format(d) in i, cur in i, 'Loan' in i])]) +\
                        fx_out - fx_in +  intercomp_constraint[cur][d]['bal']\
                        == daily_bal[d][cur], '{}_{}_{}_bal'.format(comp,d,cur)
                
                    else:
                        prob += LpAffineExpression([(self.var_bal[i],1) for i in self.var_bal.keys() if all([comp in i,'{}'.format(d) in i, cur in i, 'Loan' not in i])])-\
                        LpAffineExpression([(self.var_bal[i],1) for i in self.var_bal.keys() if all([comp in i,'{}'.format(d) in i, cur in i, 'Loan' in i])]) +\
                        fx_out - fx_in \
                        == daily_bal[d][cur], '{}_{}_{}_bal'.format(comp,d,cur)
                
                else:
                    #fx position only cares about flow in of spot or forward(named swap in this case)
                    fx_in = None
                    fx_out = None
                    for t,_ in {i+'_':j[3] for i,j in fx_info.items() if all([j[1] == cur])}.items():
                        lst_out = [i for i,j in self.var_fx_flow_in.items() if t in i]
                        boo_out = [self.within_date_range(i,d,d_prior) for i in lst_out]*np.array([1]*len(lst_out))
                        fx_out += lpSum([self.var_fx_flow_in[i] for i in lst_out]*boo_out)
                        
                    for t,_ in {i+'_':j[3] for i,j in fx_info.items() if all([j[2] == cur])}.items():
                        lst_out = [i for i,j in self.var_fx_flow_in.items() if t in i]
                        boo_out = [self.within_date_range(i,d,d_prior) for i in lst_out]*np.array([1]*len(lst_out))
                        fx_in += lpSum([self.var_fx_flow_in[i] for i in lst_out]*boo_out)
                    
                    prob += fx_out - fx_in \
                        == daily_bal[d][cur], '{}_{}_{}_bal'.format(comp,d,cur)

          #                     fx_in = [self.var_fx_flow_in['_'.join([i,str(d),'in'])] for i,j in fx_info.items() if j[2] == cur]
#                     fx_out = [self.var_fx_flow_in['_'.join([i,str(d),'in'])] for i,j in fx_info.items() if j[1] == cur]             
#                     fx_in = LpAffineExpression([(fx_in[i],1) for i in range(len(fx_in))])
#                     fx_out = LpAffineExpression([(fx_out[i],1) for i in range(len(fx_out))])                    



                


    def within_date_range(self, s, d, d_prior):
        d_string = s.rsplit('_',1)[0].split('_')[-1]
        d_datetime = datetime.datetime.strptime(d_string,'%Y-%m-%d %H:%M:%S')
        return all([d_datetime <= d, d_datetime >= d_prior])

    def cf_constraint(self, prob , intercomp_constraint = {}, fx_pos_constraints = []):
        #'FX Position does not have to add cash flow constraints.'
               
        daily_flows = self.entity['Flow']
        fx_info = {}
        if len([i for i in self.entity['FX'].keys()]) != 0:
            fx_info = self.entity['FX']['fx_info']

        comp = self.entity['name']
        for cur in self.cur_lst:
            if cur in intercomp_constraint.keys():
                intercomp = True
            else:
                intercomp = False
            for d in daily_flows.keys():
                if self.fx_position == False: 
                #print(d)
                    #正數
                    fx_in = [self.var_fx_flow_in['_'.join([i,str(d),'in'])] for i,j in fx_info.items() if j[2] == cur]
                    fx_out = [self.var_fx_flow_in['_'.join([i,str(d),'in'])] for i,j in fx_info.items() if j[1] == cur]
                    fx_in = LpAffineExpression([(fx_in[i],1) for i in range(len(fx_in))])
                    fx_out = LpAffineExpression([(fx_out[i],1) for i in range(len(fx_out))])
                    #下面是負數
                    fx_due_in = [self.var_flow_swap_out['_'.join([i,str(d),'out'])] for i,j in fx_info.items() if all([j[1] == cur,j[0] != 'spot'])]
                    fx_due_out = [self.var_flow_swap_out['_'.join([i,str(d),'out'])] for i,j in fx_info.items() if all([j[2] == cur,j[0] != 'spot'])]
                    fx_due_in = LpAffineExpression([(fx_due_in[i],1) for i in range(len(fx_due_in))])
                    fx_due_out = LpAffineExpression([(fx_due_out[i],1) for i in range(len(fx_due_out))])

                    if len(fx_pos_constraints) != 0:
                        assert len([i for _,i in fx_info.items() if i[0] == 'spot']) == 0, 'spot should be a variable in fx position entity!'
                        fx_pos_info = fx_pos_constraints[0]
                        fx_pos_spot_in = fx_pos_constraints[1]
                        fx_pos_fwd_out = fx_pos_constraints[2]

                        spot_in = [fx_pos_spot_in['_'.join([i,str(d),'in'])] for i,j in fx_pos_info.items() if all([j[2] == cur, j[0] == 'spot'])]
                        spot_out = [fx_pos_spot_in['_'.join([i,str(d),'in'])] for i,j in fx_pos_info.items() if all([j[1] == cur,j[0] == 'spot'])]
                        fx_in += LpAffineExpression([(spot_in[i],1) for i in range(len(spot_in))])
                        fx_out += LpAffineExpression([(spot_out[i],1) for i in range(len(spot_out))])

                        fwd_in = [fx_pos_fwd_out['_'.join([i,str(d),'out'])] for i,j in fx_pos_info.items() if all([j[2] == cur, j[0] != 'spot'])]
                        fwd_out = [fx_pos_fwd_out['_'.join([i,str(d),'out'])] for i,j in fx_pos_info.items() if all([j[1] == cur,j[0] != 'spot'])]
                        fx_in += LpAffineExpression([(fwd_in[i],-1) for i in range(len(fwd_in))])
                        fx_out += LpAffineExpression([(fwd_out[i],-1) for i in range(len(fwd_out))])


                    f = LpAffineExpression([(self.var_flow_in[i],1) for i in self.var_flow_in.keys() if all([comp in i,'{}'.format(d) in i, cur in i,'Loan' not in i])])+\
                    LpAffineExpression([(self.var_flow_out[i],1) for i in self.var_flow_out.keys() if all([comp in i,'{}'.format(d) in i, cur in i,'Loan' not in i])])
                    # 要加intercompany loan

                    f_loan = LpAffineExpression([(self.var_flow_in[i],1 ) for i in self.var_flow_in.keys() if all(['{}'.format(d) in i, cur in i,'Loan' in i])])+\
                    LpAffineExpression([(self.var_flow_out[i],1) for i in self.var_flow_out.keys() if all(['{}'.format(d) in i, cur in i,'Loan' in i])])

                    if intercomp == True:
                        prob += f - f_loan - fx_in + fx_due_in + fx_out - fx_due_out + intercomp_constraint[cur][d]['flow'] == daily_flows[d][cur], \
                            '{}_{}_{}_flow'.format(comp,d,cur)
                    else:
                        prob += f - f_loan - fx_in + fx_due_in + fx_out - fx_due_out == daily_flows[d][cur], \
                            '{}_{}_{}_flow'.format(comp,d,cur)
                else: # fx_position = True
                    fx_in = [self.var_fx_flow_in['_'.join([i,str(d),'in'])] for i,j in fx_info.items() if all([j[2] == cur, j[0] == 'spot'])]
                    fx_out = [self.var_fx_flow_in['_'.join([i,str(d),'in'])] for i,j in fx_info.items() if all([j[1] == cur,j[0] == 'spot'])]
                    fx_in = LpAffineExpression([(fx_in[i],1) for i in range(len(fx_in))])
                    fx_out = LpAffineExpression([(fx_out[i],1) for i in range(len(fx_out))])
                    #下面是負數
                    fx_due_in = [self.var_flow_swap_out['_'.join([i,str(d),'out'])] for i,j in fx_info.items() if all([j[1] == cur,j[0] != 'spot'])]
                    fx_due_out = [self.var_flow_swap_out['_'.join([i,str(d),'out'])] for i,j in fx_info.items() if all([j[2] == cur,j[0] != 'spot'])]
                    fx_due_in = LpAffineExpression([(fx_due_in[i],-1) for i in range(len(fx_due_in))])
                    fx_due_out = LpAffineExpression([(fx_due_out[i],-1) for i in range(len(fx_due_out))])
                    
                    f = LpAffineExpression([(self.var_flow_in[i],1) for i in self.var_flow_in.keys() if all([comp in i,'{}'.format(d) in i, cur in i,'Loan' not in i])])+\
                    LpAffineExpression([(self.var_flow_out[i],1) for i in self.var_flow_out.keys() if all([comp in i,'{}'.format(d) in i, cur in i,'Loan' not in i])])
                    
                    prob += f - fx_in + fx_due_in + fx_out - fx_due_out == 0, \
                            '{}_{}_{}_flow'.format(comp,d,cur)

    def fx_constraint(self, prob):
        ###############
        #swap
        ################
        fx_info = self.entity['FX']['fx_info']
        date_lst = [i for i in self.entity['Balance'].keys()]
        d_prior = min(date_lst)
        d_last = max(date_lst)
        for t,tenor in {i+'_':j[3] for i,j in fx_info.items() if j[0] != 'spot'}.items():
            count = 0
            for weeknum , d in enumerate(date_lst):

                lst_in = [i for i,j in self.var_fx_flow_in.items() if t in i]
                lst_out = [i for i,j in self.var_flow_swap_out.items() if t in i]
                boo_in = [self.within_date_range(i,d,d_prior) for i in lst_in]*np.array([1]*len(lst_in))
                boo_out = [self.within_date_range(i,d,d_prior) for i in lst_out]*np.array([1]*len(lst_out))
                bal = self.var_fx_bal[''.join([t,str(d)])]
                prob += lpSum([self.var_fx_flow_in[i] for i in lst_in]*boo_in +\
                              [self.var_flow_swap_out[i] for i in lst_out]*boo_out) == lpSum(bal),\
                '{}_swap_{}_{}'.format(self.entity['name'],t, count)
                ##########################################
                weeks = max(1,math.floor(tenor/7))
                in_week = weeknum - weeks
        #         out_date = d + relativedelta(days = +tenor)
        #         check_out = all([out_date >= d_prior,out_date <= d_last,sum([i == out_date for i in balance['Date']]) == 0])
        #         check_in = all([d_in >= d_prior, sum([i == d_in for i in balance['Date']]) > 0])
                if in_week >= 0: # 如果遇假日到期就不能做 (不然要多判斷式)
                    d_in = date_lst[in_week]
                    prob += self.var_fx_flow_in[''.join([t,str(d_in),'_in'])] == \
                    -1*self.var_flow_swap_out[''.join([t,str(d),'_out'])],\
                    'no_neg_flows_{}_{}_{}'.format(self.entity['name'],t,count)
                if in_week < 0:
                    prob += self.var_flow_swap_out[''.join([t,str(d),'_out'])] == 0,\
                    'no_neg_flows_{}_{}_{}'.format(self.entity['name'],t,count)

        #         if check_out:
        #             prob += var_fx_flow_in[''.join([t,str(d),'_in'])] == 0, \
        #             'no_inflow_{}_{}'.format(t,count)

                count += 1

        ###############
        #spot
        ################
        #d_prior = min(balance['Date'])
        for t,tenor in {i+'_':j[3] for i,j in fx_info.items() if j[0] == 'spot'}.items():
            count = 0
            for d in date_lst:
                #d_in = d + relativedelta(days = float(-tenor))
                lst_in = [i for i,j in self.var_fx_flow_in.items() if t in i]

                boo_in = [self.within_date_range(i,d,d_prior) for i in lst_in]*np.array([1]*len(lst_in))

                bal = self.var_fx_bal[''.join([t,str(d)])]
                prob += lpSum([self.var_fx_flow_in[i] for i in lst_in]*boo_in) == lpSum(bal),\
                '{}_spot_{}_{}'.format(self.entity['name'],t, count)

                count += 1
        #total transaction limit
        count = 0
        for d in date_lst:
            spot = [j for i, j in self.var_fx_bal.items() if all(['spot' in i, str(d) in i])]
            swap = [j for i, j in self.var_fx_bal.items() if all(['swap' in i, str(d) in i])]
            if len(spot) >0:
                prob += lpSum(spot) <= 1000000, '{}_spot_limit_{}'.format(self.entity['name'],count)
            if len(swap) >0:
                prob += lpSum(swap) <= 1000000, '{}_swap_limit_{}'.format(self.entity['name'],count)
            count += 1


    def product_constraint(self, prob):
        comp = self.entity['name']
        df_product = self.entity['Product']['df_product']
        date_lst = [i for i in self.entity['Balance'].keys()]
        d_prior = min(date_lst)
        d_last = max(date_lst)
        for bank,cur,prod,tenor in zip(df_product['Bank'],df_product['Curr'],\
                                            df_product['product'],df_product['Tenor']):
            #for cur in ['USD','TWD','JPY','EUR']:
            count = 0
            for weeknum , d in enumerate(date_lst):
                #d_in = d + relativedelta(days = -tenor)
                #print(d_in)
                lst_in = [self.var_flow_in['{}_{}_{}_{}_{}_in'.format(comp,bank,cur,prod,di)] for di in date_lst]

                lst_out = [self.var_flow_out['{}_{}_{}_{}_{}_out'.format(comp,bank,cur,prod,di)] for di in date_lst]

                boo_in = [all([d_datetime <= d, d_datetime >= d_prior]) for d_datetime in date_lst]*np.array([1]*len(lst_in))
                boo_out = [all([d_datetime <= d, d_datetime >= d_prior]) for d_datetime in date_lst]*np.array([1]*len(lst_out))
                #print(bank, cur,boo)
                bal = LpAffineExpression([(self.var_bal['{}_{}_{}_{}_{}_bal'.format(comp,bank,cur,prod,d)],1)])
                #bal = [var_bal[i] for i in var_bal.keys() if all([comp in i,bank in i, cur in i, prod in i,'{}'.format(d) in i])]
                ###########################################
                #balance of a date is cumsum of flows in previous dates.
                ###########################################
                #if len(lst_in) >0:
                prob += LpAffineExpression([(lst_in[i],boo_in[i]) for i in range(len(lst_in))]) +\
                LpAffineExpression([(lst_out[i],boo_out[i]) for i in range(len(lst_out))]) == bal,\
                'TD_{}_{}_{}_{}_{}'.format(comp,prod,bank,cur,count)

                ###########################################
                # flow_in and flow_out closely connected: flow in accompanied by flow out after n days
                ###########################################
                if prod not in ['SpecialS', 'Saving']:
                    weeks = max(1,math.floor(tenor/7))
                    in_week = weeknum - weeks            

                    if in_week >= 0: # 如果遇假日到期就不能做 (不然要多判斷式)
                        d_in = date_lst[in_week]
                        prob += LpAffineExpression([(self.var_flow_in['{}_{}_{}_{}_{}_in'.format(comp,bank,cur,prod,d_in)],1)]) == \
                        LpAffineExpression([(self.var_flow_out['{}_{}_{}_{}_{}_out'.format(comp,bank,cur,prod,d)],-1)]),\
                        'no_neg_flows_{}_{}_{}_{}_{}'.format(comp, prod,bank,cur,count)
                    if in_week < 0:
                        prob += self.var_flow_out['{}_{}_{}_{}_{}_out'.format(comp,bank,cur,prod,d)] == 0,\
                        'no_neg_flows_{}_{}_{}_{}_{}'.format(comp,prod,bank,cur,count)

                count += 1

    def product_specific_constraint(self, prob):
        #############
        #把日期改成周的日期
        #############
        data = self.data.copy()
        comp = self.entity['name']
        date_lst = [i for i in self.entity['Balance'].keys()]
        data['Date'] = data['Date'].apply(lambda x: date_lst[np.argmax([(i-x).days >= 0 for i in date_lst])])
        data = data.drop_duplicates(['Company','Bank','Curr','product','Tenor','Date'],keep = 'first')
        date_frame = pd.DataFrame(data = date_lst, columns = ['Date'])

        ##############################
        #Individual account constrains - daily maximum amount allowed (有tenor就是單點 flow限制，沒有tenor就是未來期間的bal)
        ##############################
        daily_max = data.loc[(data['Daily_max'].isnull() == False) & (data['Daily_min'].isnull()) & (data['average'].isnull())]
        df = daily_max.groupby(['Company','Bank','Curr','product'])['Tenor'].count().reset_index()
        for indx in range(len(df.index)):
            prod = df.loc[indx,'product']
            bank = df.loc[indx,'Bank']
            curr = df.loc[indx,'Curr']
            #comp = df.loc[indx,'Company']
            #if prod in ['Saving','SpecialS']:
            temp = df.loc[[indx],:].merge(daily_max, on = ['Company','Bank','Curr','product']).merge(date_frame, how = 'outer')
            temp = temp.sort_values(['Date'], ascending = True).reset_index()
            temp['Daily_max'] = temp['Daily_max'].ffill()
            if prod in ['Saving','SpecialS']:
                for d, vol in zip(temp['Date'],temp['Daily_max']):
                    if np.isnan(vol) == False:
                        prob += lpSum(self.var_bal['{}_{}_{}_{}_{}_bal'.format(comp,bank,curr,prod,d)]) <= vol,\
                                'daily_max{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,d)
            else:
                for d, vol in zip(temp['Date'],temp['Daily_max']):
                    if np.isnan(vol) == False:
                        prob += lpSum(self.var_flow_in['{}_{}_{}_{}_{}_in'.format(comp,bank,curr,prod,d)]) <= vol,\
                                'daily_max{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,d)
        #     else:
        #         temp = df.loc[[indx],:].merge(daily_max, on = ['Company','Bank','Curr','product'])
        #         for d, vol in zip(temp['Date'],temp['Daily_max']):
        #             if np.isnan(vol) == False:
        #                 prob += lpSum(var_flow_in['{}_{}_{}_{}_{}_in'.format(comp,bank,curr,prod,d)]) <= vol,\
        #                         'daily_max{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,d)

        ##############
        #如果有balance 要大於數值的(Daily_min)，要用boolean/ 目前假設會有這種條件是 saving 類, average 先假設 通通都設
        ##############
        min_cap = data.loc[(data['Daily_min'].isnull() == False) | (data['average'].isnull() == False)]
        df = min_cap.groupby(['Company','Bank','Curr','product'])['Tenor'].count().reset_index()
        for indx in df.index:
            prod = df.loc[indx,'product']
            bank = df.loc[indx,'Bank']
            bank_group = re.findall('\d*([a-zA-Z]+)',bank)[0]
            curr = df.loc[indx,'Curr']
            comp = df.loc[indx,'Company']

            temp = df.loc[[indx],:].merge(min_cap, on = ['Company','Bank','Curr','product']).merge(date_frame, how = 'outer')
            temp = temp.sort_values(['Date'], ascending = True).reset_index()
            temp[['Daily_max','Daily_min']] = temp[['Daily_max','Daily_min']].ffill()
            df_array = np.array(temp[['Date','Daily_max','Daily_min','average']])
            d_boo_format = [str(d.year)+str(d.month) for d in temp['Date']]
            boo_effective = [] # for daily min
        #     boo_effective_1 = [] # for average >
        #     boo_effective_2 = [] # for average <
            #max_limit = max([j for _,j in bank_limit[bank_group].items()])
            max_limit = 1000000
            for i in range(df_array.shape[0]):
                if np.isnan(df_array[i,1]) == False:
                    prob += lpSum(self.var_bal['{}_{}_{}_{}_{}_bal'.format(comp,bank,curr,prod,df_array[i,0])]) <= df_array[i,1],\
                            'daily_max{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,i)  


                if np.isnan(df_array[i,2]) == False:

                    prob += lpSum(self.var_bal['{}_{}_{}_{}_{}_bal'.format(comp,bank,curr,prod,df_array[i,0])]) >= df_array[i,2], \
                    'daily_min{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,i) 

                if isinstance(df_array[i,3], str):
                    if '>' in df_array[i,3]:
                        amt = float(str(df_array[i,3]).replace('>',''))
                        for ym in set(d_boo_format): #by month making boolean effective
                            num = [i for i,j in enumerate(d_boo_format) if j == ym]
                            prob += lpSum([self.var_bal['{}_{}_{}_{}_{}_bal'.format(comp,bank,curr,prod,df_array[d,0])] for d in num]) <= \
                            self.var_boo['{}_{}_{}_{}_{}_boo'.format(comp,bank,curr,prod,ym)]*max_limit*len(num),\
                            'make_boolean_effective{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,ym)

                            #每月平均大於這個金額 
                            prob += lpSum([self.var_bal['{}_{}_{}_{}_{}_bal'.format(comp,bank,curr,prod,df_array[d,0])] for d in num]) \
                            >= self.var_boo['{}_{}_{}_{}_{}_boo'.format(comp,bank,curr,prod,ym)]*amt*len(num), \
                            'average_larger_than{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,ym)

                    if '<' in df_array[i,3]:
                        amt = float(str(df_array[i,3]).replace('<',''))
                        #for ym in set(d_boo_format): # by month
                            #num = [i for i,j in enumerate(d_boo_format) if j == ym]
                        num = [i for i,j in enumerate(d_boo_format)]
                        prob += lpSum([self.var_bal['{}_{}_{}_{}_{}_bal'.format(comp,bank,curr,prod,df_array[d,0])] for d in num]) \
                        <= amt*len(num), 'average_less_than{}_{}_{}_{}_{}'.format(comp,bank,curr,prod,i)
