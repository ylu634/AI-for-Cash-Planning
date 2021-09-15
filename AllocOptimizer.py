from pulp import *
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import datetime
from dateutil.relativedelta import *
from collections import Counter
import xlwings as xw
import CrossEntityLink as lk

class Optimizer(object):  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    """  		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
  	intecompany loan variables and constraints		  		 			     			  	  		 	  	 		 			  		  			
    """ 
		  	   		   	 			  		 			     			  	  		 	  	 		 			  		  			
    # constructor  		  
    def __init__(self, prob ,entity_lst, intercomp_lst, bank_limit_lst, fx_cf_pos = []):
        '''
        prob: pulp linear maximization problem, already initiated and objective function defined. (here assumes intercompany loan does not incur any gain or loss.)
        entity_lst: entity from Entity.py, with variables already initiated
        intercomp_lst: [entity_from, entity_to, curr , low_bnd (default should be 0) , up_bnd (default should be None)]
        bank_limit_lst: contains [[entities to share the bank limits], bank_limit = {x:y for x, y in zip(constrains['Bank'],constrains['Limit'])}]
        fx_cf_pos: list providing which entity represents cash flow and which entity represents fx balance sheet position [cash_flow_entity, fx_entity]
        '''
        self.prob = prob    
        self.entity_lst = entity_lst
        self.intercomp_lst = intercomp_lst
        self.intercomp_vars = []
        self.intercomp_comps = {c.entity['name']:{} for c in entity_lst}
        self.bank_limit_lst = bank_limit_lst
        self.fx_pos_constraints = {c.entity['name']:[] for c in entity_lst}
        self.main_entity = {c.entity['name']:c.entity['name'] for c in entity_lst}
        ###########################
        # intercompany constraints:
        ############################
        for lst in intercomp_lst:
            intercomp = lk.Intercompany()
            intercomp.var_init(lst[0], lst[1],lst[2], low_bnd = lst[3], up_bnd = lst[4])
            
            intercomp_constraints = intercomp.intercomp_constraint(self.prob)
            self.intercomp_vars.append(intercomp)
            #to 是負的
            constraints_to = {i:{'bal': -intercomp_constraints[i]['bal'],'flow': -intercomp_constraints[i]['flow']} for i in intercomp_constraints.keys()}
            #from entity: 
            from_comp = lst[0].entity['name']
            curr = lst[2]
            if curr not in self.intercomp_comps[from_comp].keys():
                self.intercomp_comps[from_comp][curr] = intercomp_constraints
            else:
                for k in self.intercomp_comps[from_comp][curr].keys():
                    self.intercomp_comps[from_comp][curr][k]['bal'] += intercomp_constraints[k]['bal']
                    self.intercomp_comps[from_comp][curr][k]['flow'] += intercomp_constraints[k]['flow']
            to_comp = lst[1].entity['name']

            if curr not in self.intercomp_comps[to_comp].keys():
                self.intercomp_comps[to_comp][curr] = constraints_to
            else:
                for k in self.intercomp_comps[from_comp][curr].keys():
                    self.intercomp_comps[to_comp][curr][k]['bal'] += constraints_to[k]['bal']
                    self.intercomp_comps[to_comp][curr][k]['flow'] += constraints_to[k]['flow']
        #####################################
        # Add fx position
        #####################################
        for link in fx_cf_pos:
            cf = link[0]
            pos = link[1]
            assert sum([i not in cf.cur_lst for i in pos.cur_lst]) == 0, 'Currencies in {} not included in {}'.format(pos.entity['name'], cf.entity['name'])
            fx_pos_info = pos.entity['FX']['fx_info']
            fx_pos_spot_in = pos.var_flow_spot_in
            fx_pos_fwd_out = pos.var_flow_swap_out
            fx_pos_spot_bal = pos.var_spot_bal        
            self.fx_pos_constraints[cf.entity['name']] = [fx_pos_info,fx_pos_spot_in,fx_pos_fwd_out,fx_pos_spot_bal]
            self.main_entity[pos.entity['name']] = cf.entity['name']
        
        
        ##################################
        # run all constraints
        ##################################
        for ent in self.entity_lst:
            const = self.intercomp_comps[ent.entity['name']]
            fx_const = self.fx_pos_constraints[ent.entity['name']]
            ent.bal_constraint(self.prob,intercomp_constraint = const, fx_pos_constraints = fx_const)
            ent.cf_constraint(self.prob,intercomp_constraint = const,fx_pos_constraints = fx_const)
            if len([i for i in ent.entity['FX'].keys()]) > 0:
                ent.fx_constraint(self.prob)
            ent.product_constraint(self.prob)
            ent.product_specific_constraint(self.prob)
        
        #####################
        # Bank limit
        #11/17/2020 add slack 1 to avoid infeasible result in divide and conquer
        #limit 不能是負的
        #####################
        #for bank in set(data['Bank']):
        
        count = 0
        for bk_lst in self.bank_limit_lst:
            entities = bk_lst[0]
            bank_limit = bk_lst[1]
            banks = pd.concat([ent.data for ent in entities], ignore_index = True)['Bank']
            date_lst = [i for i in entities[0].entity['Balance'].keys()]
            for bank in set([re.findall('\d*([a-zA-Z]+)',i)[0] for i in banks]):
                for d in date_lst:
                    strf_time = datetime.datetime.strftime(d,'%Y%m%d')
                    sum_entities = None
                    for ent in entities:
                        sum_entities += LpAffineExpression([(ent.var_bal[i],1) for i in ent.var_bal.keys() if\
                                                all([len(re.findall('(_\d?{}_)'.format(bank), i))>0,\
                                                    '{}'.format(d) in i])])
                    self.prob += sum_entities \
                    <= max(bank_limit[bank][strf_time],0),\
                    'Bank_limit_{}_{}'.format(bank,count)

                    count += 1 


    def solver(self):
        self.prob.solve()
        print("Status:", LpStatus[self.prob.status])

    def output(self, to_excel = False):

        df_bal_all = pd.DataFrame()
        df_flow_all = pd.DataFrame()
        df_fx_bal_all = pd.DataFrame()
        df_fx_flow_all = pd.DataFrame()
        for ent in self.entity_lst:
            df_bal = pd.DataFrame(data = [i.split('_')[:-1]+[j.value()] for i,j in ent.var_bal.items()], columns = ['Company','Bank','Curr','Product','Date','Value'])
            df_flowin =pd.DataFrame(data =  [i.split('_')+[j.value()] for i,j in ent.var_flow_in.items()], columns = ['Company','Bank','Curr','Product','Date','INorOut','Value'])
            df_flowout =pd.DataFrame(data =  [i.split('_')+[j.value()] for i,j in ent.var_flow_out.items()], columns = ['Company','Bank','Curr','Product','Date','INorOut','Value'])
            df_flow = pd.concat([df_flowin, df_flowout], ignore_index = True)
            df_bal_all = pd.concat([df_bal_all, df_bal], ignore_index = True)
            df_flow_all = pd.concat([df_flow_all, df_flow], ignore_index = True)
            if len([i for i in ent.entity['FX'].keys()]) > 0:
                df_fx_bal = pd.DataFrame(data = [i.split('_')+[j.value()] for i,j in ent.var_fx_bal.items()],columns = ['product','Curr_from','Curr_to','Tenor','Date','Value'])
                df_fx_bal['Company'] = np.repeat(self.main_entity[ent.entity['name']], len(df_fx_bal.index))

                df_fx_flowin = pd.DataFrame(data = [i.split('_')+[j.value()] for i,j in ent.var_fx_flow_in.items()],columns = ['product','Curr_from','Curr_to','Tenor','Date','INorOut','Value'])
                df_fx_flowout = pd.DataFrame(data = [i.split('_')+[j.value()] for i,j in ent.var_flow_swap_out.items()],columns = ['product','Curr_from','Curr_to','Tenor','Date','INorOut','Value'])

                df_fx_flow = pd.concat([df_fx_flowin, df_fx_flowout], ignore_index = True)
                df_fx_flow['Company'] = np.repeat(self.main_entity[ent.entity['name']], len(df_fx_flow.index))
                if ent.fx_position == True:
                    df_fx_bal['product'] = df_fx_bal['product'].apply(lambda x: x.replace('swap','fwd'))
                    df_fx_flow['product'] = df_fx_flow['product'].apply(lambda x: x.replace('swap','fwd'))
                df_fx_bal_all = pd.concat([df_fx_bal_all, df_fx_bal], ignore_index = True)
                df_fx_flow_all = pd.concat([df_fx_flow_all, df_fx_flow], ignore_index = True)

        df_intercomp_all = pd.DataFrame()
        for pairs in self.intercomp_vars:
            df_intercomp = pd.concat([\
                    pd.DataFrame(data = [i.split('_')+[j.value()] for i,j in pairs.var_intercompany_flow_in.items()],columns = ['product','Date','INorOut','Value']),\
            pd.DataFrame(data = [i.split('_')+[j.value()] for i,j in pairs.var_intercompany_flow_out.items()],columns = ['product','Date','INorOut','Value']),\
            pd.DataFrame(data = [i.split('_')+[j.value()] for i,j in pairs.var_intercompany_bal.items()],columns = ['product','Date','INorOut','Value'])])
            df_intercomp = pd.pivot_table(df_intercomp, columns = ['INorOut'], values = ['Value'], index = ['product','Date']).reset_index()
            df_intercomp.columns = ['product','Date','bal','in','out']
            df_intercomp['from'] = np.repeat(pairs.fr, len(df_intercomp.index))
            df_intercomp['to'] = np.repeat(pairs.to, len(df_intercomp.index))
            df_intercomp['Curr'] = np.repeat(pairs.curr, len(df_intercomp.index))
            df_intercomp_all = pd.concat([df_intercomp_all, df_intercomp], ignore_index = True)
        #contribution
        contribution = pd.DataFrame()
        for ent in self.entity_lst:
            
            new_int_rate_mat = ent.entity['Product']['rate']
            objective_basis = ent.entity['Product']['objective_basis']
            new_td_tenor = ent.entity['Product']['tenor']
            if len([i for i in ent.entity['FX'].keys()]) > 0:
                new_fx_rate_mat = ent.entity['FX']['rate']
                objective_basis_fx = ent.entity['FX']['objective_basis']
                new_fx_tenor = ent.entity['FX']['tenor']

                int_out = np.hstack((new_fx_rate_mat.reshape(np.array(ent.fx_gl_var).shape, order = 'C'),new_int_rate_mat.reshape(ent.out_mat.shape,order = 'C')))
                pos_out = np.hstack((np.array(ent.fx_gl_var),ent.out_mat))
                final_output = pd.DataFrame({'Var':pos_out,'rate':int_out})
                final_output['Position'] = final_output['Var'].apply(lambda x: x.value())
                final_output['contribution'] = final_output['Position'] * final_output['rate']
                final_output['Var'] = np.hstack((np.array(objective_basis_fx),objective_basis))

                tenor_out = np.hstack((new_fx_tenor.reshape(np.array(ent.fx_gl_var).shape, order = 'C'),new_td_tenor.reshape(ent.out_mat.shape,order = 'C')))
                final_output['Tenor'] = tenor_out
            else:
                int_out = new_int_rate_mat.reshape(ent.out_mat.shape,order = 'C')
                pos_out = ent.out_mat
                final_output = pd.DataFrame({'Var':pos_out,'rate':int_out})
                final_output['Position'] = final_output['Var'].apply(lambda x: x.value())
                final_output['contribution'] = final_output['Position'] * final_output['rate']
                final_output['Var'] = objective_basis

                tenor_out = new_td_tenor.reshape(ent.out_mat.shape,order = 'C')
                final_output['Tenor'] = tenor_out                

            details = []
            for i in final_output.Var:
                s = i.split('_')
                if any(['spot' in s, 'swap' in s]):
                    details.append([ent.entity['name']]+s[0:1]+[''.join([s[1],s[2]])]+s[3:4]+s[4:5])
                else:
                    details.append(s[0:5])

            final_output = pd.concat([final_output,pd.DataFrame(columns = ['n','Bank','Curr','product','Date'],data = details)], axis=1)
            final_output = final_output.drop(['Var'], axis = 1)
            final_output.loc[final_output['Tenor'] == 0,'Tenor'] = final_output.loc[final_output['Tenor'] == 0,'Tenor'].apply(lambda x: 1)
            final_output['Rate_real'] = np.where(final_output['Curr'] != 'TWD',final_output['rate']*360/final_output['Tenor'],final_output['rate']*365/final_output['Tenor'])
            final_output['Company'] = final_output['n'].copy()
            if ent.fx_position == True:
                final_output['Company'] = np.where((final_output['product'] == 'spot') | (final_output['product'] == 'swap'),\
                    self.main_entity[ent.entity['name']], final_output['Company'])
                final_output['product'] = final_output['product'].apply(lambda x: x.replace('swap','fwd'))
            contribution = pd.concat([contribution,final_output], ignore_index = True)
            
            for df in df_bal_all, df_flow_all, contribution,df_fx_bal_all,df_fx_flow_all, df_intercomp_all:
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])

        if to_excel:
            wb = xw.Book('final_result.xlsx')
            sheet_main = wb.sheets['df_bal']
            sheet_main.clear_contents()
            sheet_main.range('A1').options(index=False).value = df_bal_all

            sheet_flow = wb.sheets['df_flow']
            sheet_flow.clear_contents()
            sheet_flow.range('A1').options(index=False).value = df_flow_all

            cont = wb.sheets['contribution']
            cont.clear_contents()
            cont.range('A1').options(index=False).value = contribution

            sheet_fx_bal = wb.sheets['df_fx_bal']
            sheet_fx_bal.clear_contents()
            if df_fx_bal_all.empty == False:
                sheet_fx_bal.range('A1').options(index=False).value = df_fx_bal_all

            sheet_fx_flow = wb.sheets['df_fx_flow']
            sheet_fx_flow.clear_contents()
            if df_fx_flow_all.empty == False:
                sheet_fx_flow.range('A1').options(index=False).value = df_fx_flow_all
            
            sheet_intercomp = wb.sheets['intercompany']
            sheet_intercomp.clear_contents()
            if df_intercomp_all.empty == False:
                sheet_intercomp.range('A1').options(index=False).value = df_intercomp_all

            wb.save()
            return True
        return df_bal_all, df_flow_all, contribution,df_fx_bal_all,df_fx_flow_all, df_intercomp_all