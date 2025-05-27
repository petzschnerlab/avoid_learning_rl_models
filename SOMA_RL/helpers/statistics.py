import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy as sp
import statsmodels.api as sm


class Statistics:

    """
    Class to run statistical analyses
    """

    def linear_model_continous(self, formula: str, data: pd.DataFrame):
        
        """
        Fit a linear model to continuous data

        Parameters
        ----------
        formula : str
            The formula to fit the model on
        data : pd.DataFrame
            The data to fit the model on

        Returns
        -------
        smf.ols
            The fitted model
        """        

        return smf.ols(formula=formula, data=data).fit()

    def linear_model_categorical(self, formula: str, data: pd.DataFrame) -> dict:
        
        """
        Fit a linear model to categorical data and return the coefficients

        Parameters
        ----------
        formula : str
            The formula to fit the model on
        data : pd.DataFrame
            The data to fit the model on

        Returns
        -------
        dict
            The coefficients of the model
        """   

        #Check anova assumptions
        #Check normality of each group
        metric_name = formula.split(' ~ ')[0]
        group_name = formula.split(' ~ ')[-1]
        normality = {group: [] for group in data[group_name].unique()}
        for group in data[group_name].unique():
            group_data = data[data[group_name] == group][metric_name]
            normality[group] = 'met' if sp.stats.shapiro(group_data).pvalue > 0.05 else 'violated'
        normality_assumption = 'met' if all([assumption == 'met' for assumption in normality.values()]) else 'violated'

        #Check homogeneity of variance
        metric_data = data[[group_name, metric_name]]
        metric_list = [metric_data[metric_data[group_name] == group][metric_name] for group in metric_data[group_name].unique()]
        homogeneity = sp.stats.levene(*metric_list)
        homogeneity_assumption = 'met' if homogeneity.pvalue > 0.05 else 'violated'

        #Use ols to get omnibus p-value for category pain_group
        regression_model = smf.ols(formula=formula, data=data).fit()

        #Extract F, , df.res, df.model, p-value (Prob (F-statistic)), and R^2
        regression_dict = {'F': regression_model.fvalue,
                          'df_res': regression_model.df_resid,
                          'df_model': regression_model.df_model,
                          'p_value': regression_model.f_pvalue,
                          'r_squared': regression_model.rsquared,
                          'normality': normality_assumption,
                          'homogeneity': homogeneity_assumption}
        
        #Extract coefficients
        model_summary = pd.DataFrame(regression_dict, index=[0])
        
        metadata = {'formula': formula,
                    'test':'F'}

        return {'metadata': metadata, 'model_summary': model_summary}
    
    def planned_ttests(self, metric: str, factor: str, comparisons: list[list[str]], data: pd.DataFrame) -> dict:
            
            """
            Perform planned t-tests on the data

            Parameters
            ----------
            metric : str
                The metric to perform the t-tests on
            factor : str
                The factor to perform the t-tests on
            comparisons : list[list[str]]
                The comparisons to perform the t-tests on
            data : pd.DataFrame
                The data to perform the t-tests on

            Returns
            -------
            dict
                The metadata and the model summary
            """          

            #Wrap data into a list of dataframes
            data = [data] if type(data) is not list else data
            data = data*len(comparisons) if len(data) < len(comparisons) else data

            #Run the t-tests
            model_summary = pd.DataFrame()
            for dataframe, comparison in zip(data, comparisons):

                #Get data
                if '~' in comparison[0]:
                    condition1_index = (dataframe[factor[0]] == comparison[0].split('~')[0]) & (dataframe[factor[1]] == comparison[0].split('~')[1])
                    condition2_index = (dataframe[factor[0]] == comparison[1].split('~')[0]) & (dataframe[factor[1]] == comparison[1].split('~')[1])
                    condition1_data = dataframe[condition1_index]
                    condition2_data = dataframe[condition2_index]
                else:
                    condition1_data = dataframe[dataframe[factor] == comparison[0]]
                    condition2_data = dataframe[dataframe[factor] == comparison[1]]

                #Are there the same participant ids in condition1_data and condition2_data?
                if len(set(condition1_data['participant_id']).intersection(set(condition2_data['participant_id']))) == condition1_data['participant_id'].shape[0]:
                    condition1_data = condition1_data.sort_values('participant_id')[metric].reset_index(drop=True)
                    condition2_data = condition2_data.sort_values('participant_id')[metric].reset_index(drop=True)
                    
                    assumption_check = self.ttest_assumption_check(condition1_data, condition2_data, test_type='paired')
                    ttest = sp.stats.ttest_rel(condition1_data, condition2_data)
                    cohens_d = self.cohens_d(condition1_data, condition2_data, test_type='paired')

                else:
                    condition1_data = condition1_data[metric].astype(float)
                    condition2_data = condition2_data[metric].astype(float)

                    assumption_check = self.ttest_assumption_check(condition1_data, condition2_data, test_type='independent')
                    equal_var = assumption_check['homogeneity_assumption'] == 'met'
                    ttest = sp.stats.ttest_ind(condition1_data, condition2_data, equal_var=equal_var)
                    cohens_d = self.cohens_d(condition1_data, condition2_data, test_type='independent')

                ttest = pd.DataFrame({'condition1': comparison[0], 
                                    'condition2': comparison[1], 
                                    'comparison': f'{comparison[0]} vs {comparison[1]}',
                                    't_value': ttest.statistic, 
                                    'p_value': ttest.pvalue,
                                    'cohens_d': cohens_d,
                                    'df': ttest.df,
                                    'homogeneity_assumption': assumption_check['homogeneity_assumption'],
                                    'normality_assumption': assumption_check['normality_assumption']},
                                    index=[0])
                model_summary = pd.concat([model_summary, ttest], axis=0)

            metadata = {'metric': metric,
                        'factor': factor,
                        'comparisons': comparisons,
                        'test':'t'}

            return {'metadata': metadata, 'model_summary': model_summary}

    def ttest_assumption_check(self, group1: pd.Series, group2: pd.Series, test_type: str = 'independent') -> dict:

        """
        Check the assumptions of the t-test

        Parameters
        ----------
        group1 : pd.Series
            The first group to check the assumptions on
        group2 : pd.Series
            The second group to check the assumptions on
        test_type : str
            The type of t-test, either 'independent' or 'paired'

        Returns
        -------
        dict
            The assumption results
        """

        #Test for homogeneity of variance
        levene_results = sp.stats.levene(group1, group2)
        homogeneity_assumption = 'violated' if levene_results.pvalue < 0.05 else 'met'
        homogeneity_assumption = 'met' if test_type == 'paired' else homogeneity_assumption #Paired t-tests do not require homogeneity of variance
        assumption_results = {'homogeneity_assumption': homogeneity_assumption}

        #Test for normality
        if test_type == 'independent':
            normality_results_1, normality_results_2 = sp.stats.shapiro(group1), sp.stats.shapiro(group2)
            normality_met = normality_results_1.pvalue > 0.05 and normality_results_2.pvalue > 0.05
            assumption_results['normality_assumption'] = 'met' if normality_met else 'violated'
        else:
            normality_results = sp.stats.shapiro(group1-group2)
            normality_assumption = 'violated' if normality_results.pvalue < 0.05 else 'met'
            assumption_results['normality_assumption'] = normality_assumption

        return assumption_results
    
    def cohens_d(self, group1: pd.Series, group2: pd.Series, test_type: str = 'independent') -> float:

        """
        Calculate Cohen's d

        Parameters
        ----------
        group1 : pd.Series
            The first group to calculate Cohen's d on
        group2 : pd.Series
            The second group to calculate Cohen's d on
        test_type : str
            The type of t-test, either 'independent' or 'paired'

        Returns
        -------
        float
            Cohen's d
        """
        
        #Calculating statistics
        n1, n2 = len(group1), len(group2)

        mean1, mean2 = np.mean(group1), np.mean(group2)
        mean_diff = mean1 - mean2

        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        std_diff = np.std(group1 - group2, ddof=1)
        s1, s2 = (std1 ** 2) * (n1 - 1), (std2 ** 2) * (n2 - 1)
        npooled = n1 + n2 - 2
        pooled_std = np.sqrt((s1 + s2) / npooled)
        
        #Calculating Cohen's d
        if test_type == 'independent':
            d = mean_diff / pooled_std
        elif test_type == 'paired':
            d = mean_diff / std_diff
        else:
            raise ValueError('test_type must be either independent or paired')
    
        return d

    def post_hoc_tests(self, metric: str, factor: str, data: pd.DataFrame) -> pd.DataFrame:

        """
        Perform post-hoc tests on the data

        Parameters
        ----------
        metric : str
            The metric to perform the post-hoc tests on
        factor : str
            The factor to perform the post-hoc tests on
        data : pd.DataFrame
            The data to perform the post-hoc tests on

        Returns
        -------
        pd.DataFrame
            The post-hoc test results
        """
        
        #Create combined factor
        if type(factor) is list:
            data['factor'] = data.apply(lambda x: ' & '.join([str(x[f]) for f in factor]), axis=1)
            factor = 'factor'

        #Remove any nans
        if data[metric].astype(float).isnull().sum() > 0:
            data = data.dropna(subset=[metric])

        #Run the post-hoc tests
        tukey = sm.stats.multicomp.pairwise_tukeyhsd(data[metric].astype(float), data[factor].astype("string"))._results_table.data
        tukey_table = pd.DataFrame(tukey[1:], columns=tukey[0])
        tukey_table['factor'] = tukey_table.apply(lambda x: str(x['group1']) + ' vs ' + str(x['group2']), axis=1)
        tukey_table = tukey_table.drop(columns=['group1', 'group2'])
        tukey_table.set_index('factor', inplace=True)
        tukey_table = tukey_table.drop(columns=['lower', 'upper'])

        return tukey_table
