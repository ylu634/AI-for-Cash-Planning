from pulp import *
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
from dateutil.relativedelta import *

class Intercompany(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  	intecompany loan variables and constraints		  		 			     			  	  		 	  	 		 			  		  			
    """ 
		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # constructor  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    def __init__(self):
        self.var_intercompany_bal = {}
        self.var_intercompany_flow_in  = {}
        self.var_intercompany_flow_out = {}
        self.var_w = {}
        self.date_lst = []
        self.fr =  ''
        self.to =  ''
        self.curr = ''
    def var_init(self,Entity_from, Entity_to, curr ,low_bnd = 0, up_bnd = None):
        assert sum([f != t for f, t in zip([i for i in Entity_from.entity['Balance'].keys()],\
            [i for i in Entity_to.entity['Balance'].keys()])]) == 0, 'Dates are not the same!'
        assert all([curr in Entity_from.cur_lst, curr in Entity_to.cur_lst]), 'Currency not in entities!'
        self.curr = curr
        bal_intercompany = []
        flow_intercompany_in = []
        flow_intercompany_out = []
        self.date_lst= [i for i in Entity_from.entity['Balance'].keys()]
        for date in self.date_lst:
            bal_intercompany.append('intercompany_{}_bal'.format(date))
            flow_intercompany_in.append('intercompany_{}_in'.format(date))
            flow_intercompany_out.append('intercompany_{}_out'.format(date))
        ###################################################################

        #如果是capital injection 就是有去無回, like spot
        #var_intercompany_bal low bound = -current intercompany loan outstanding (可還的部分)
        self.fr =  Entity_from.entity['name']
        self.to =  Entity_to.entity['name']
        self.var_intercompany_bal = {i:j for i,j in zip(bal_intercompany,\
                                                LpVariable.matrix(self.curr+self.fr+'_'+self.to+'_'+"Balance_intercomp",list(range(len( bal_intercompany))),lowBound=low_bnd, upBound=up_bnd, cat='Continuous'))}
        
        self.var_intercompany_flow_in =  {i:j for i,j in zip(flow_intercompany_in,\
                                                        LpVariable.matrix(self.curr+self.fr+'_'+self.to+'_'+"Flow_intercomp_in",list(range(len(flow_intercompany_in))), lowBound=0, upBound=None, cat='Continuous'))}
        self.var_intercompany_flow_out = {i:j for i,j in zip(flow_intercompany_out,\
                                                        LpVariable.matrix(self.curr+self.fr+'_'+self.to+'_'+"Flow_intercomp_out",list(range(len(flow_intercompany_out))), lowBound=None, upBound=0, cat='Continuous'))}
        ##############
        # To make sure no simultaneous intercomp inflow and outflow in the same week. 
        ##############
        self.var_w = {i:j for i,j in zip(flow_intercompany_in,LpVariable.matrix(self.curr+self.fr+'_'+self.to+'_'+"w",list(range(len(flow_intercompany_in))), 0, 1, LpInteger))}
    
    
    def within_date_range(self, s, d, d_prior):
        d_string = s.rsplit('_',1)[0].split('_')[-1]
        d_datetime = datetime.datetime.strptime(d_string,'%Y-%m-%d %H:%M:%S')
        return all([d_datetime <= d, d_datetime >= d_prior])    

    def intercomp_constraint(self, prob):
        cons_for_comp = defaultdict(int)
        #bal constraint
        for d in self.date_lst:
            cons_for_comp[d] = defaultdict(int)
            cons_for_comp[d]['bal'] = LpAffineExpression([(self.var_intercompany_bal[i],1) for i in self.var_intercompany_bal.keys() if '{}'.format(d) in i])
            #flow constraint
            cons_for_comp[d]['flow'] = LpAffineExpression([(self.var_intercompany_flow_out[i],1) for i in self.var_intercompany_flow_out.keys() if all(['{}'.format(d) in i])])+\
            LpAffineExpression([(self.var_intercompany_flow_in[i],1) for i in self.var_intercompany_flow_in.keys() if all(['{}'.format(d) in i])])

        d_prior = min(self.date_lst)
        count = 0
        # tgl_total_flows = sum([i['USD'] for _,i in daily_flows_tgl.items()])
        for d in self.date_lst:
            lst_in = [i for i in self.var_intercompany_flow_in.keys()]
            lst_out = [i for i in self.var_intercompany_flow_out.keys()]
            boo_in = [self.within_date_range(i,d,d_prior) for i in lst_in]*np.array([1]*len(lst_in))
            boo_out = [self.within_date_range(i,d,d_prior) for i in lst_out]*np.array([1]*len(lst_out))
            #print(bank, cur,boo)
            bal = LpAffineExpression([(self.var_intercompany_bal['intercompany_{}_bal'.format(d)],1)])
            ###########################################
            #balance of a date is cumulative sum of flows in previous dates.
            ###########################################
            prob += LpAffineExpression([(self.var_intercompany_flow_in[lst_in[i]],boo_in[i]) for i in range(len(lst_in))]) + \
            LpAffineExpression([(self.var_intercompany_flow_out[lst_out[i]],boo_out[i]) for i in range(len(lst_out))])\
            == bal,\
            '{}_{}_{}_intercompany_{}'.format(self.curr,self.fr, self.to,count)
            
            ########################
            #For intercompany loan
            #如果有inflow 就不能有outflow
            #######################
            prob += self.var_w['intercompany_{}_in'.format(d)]*1000000 >= \
                self.var_intercompany_flow_in['intercompany_{}_in'.format(d)],\
            'w_intercomp_{}_{}_{}_{}'.format(self.curr,self.fr, self.to,count)
            prob += self.var_intercompany_flow_out['intercompany_{}_out'.format(d)] >= \
                self.var_w['intercompany_{}_in'.format(d)]*1000000 - 1000000,\
            'w_intercomp_another_leg_{}_{}_{}_{}'.format(self.curr,self.fr, self.to,count)

            count += 1
        return cons_for_comp