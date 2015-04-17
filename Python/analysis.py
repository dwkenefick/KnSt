### SETUP
### SETUP
# Libraries
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas import DataFrame, Series
from math import sqrt
import statsmodels.formula.api as sm
import os

#paths
root = os.path.normpath(r'C:/Users/DKenefick/Desktop/Student/')

data_path = root+os.path.normpath('/astudentData.csv')
model_path = root+os.path.normpath('\model_results.csv')

out_path = root+os.path.normpath('/Output/')

##############
### IMPORT ###
##############
data = pd.read_csv(data_path)

########################
### GLOBAL VARIABLES ###
########################

#the quantiles of students that we examine, whcich we will examin questions in.
STUDENT_QUANTILES = np.arange(.1,1,.1)

# maximum question confidence interval width
MAX_CI_WIDTH = .1

#reject quetions that are in the bottom quantile of discrimination
DISCRIMINATION_QUANTILE_SELECTION = .7

###################
### EXPLORATORY ###
###################

### Data integrity
# correct exclusivly one and zero?
data.correct.max()
data.correct.min()
data[data.correct <1].max()

### Basic counts and plots
# how many questions?
total_questions = len(data.question_id.unique())    

# how many times is each question asked?
data.groupby('question_id').agg({'question_id':'count'}).hist(bins=30) 
plt.title("Number of Responses by Question")
plt.xlabel("Number of Responses")
plt.ylabel("Frequency")
plt.savefig(out_path+os.path.normpath('/questions.pdf'))
plt.clf()

# how many users?
len(data.user_id.unique())

# how many questions per user?
data.groupby('user_id').agg({'question_id':'count'}).hist(bins=30)
plt.title("Number of Responses by Student")
plt.xlabel("Number of Responses")
plt.ylabel("Frequency")
plt.savefig(out_path+os.path.normpath('/users.pdf'))
plt.clf()

### number of qs vs averag score
# get the number of questions correct and pct correct by user.
# clearly variable number of qs per user,
# which may complicate some basic statistics
user_sum = data.groupby(by='user_id')['correct'].agg(['mean','count']).sort(columns='count')
user_sum['mean'].hist(bins=30)
plt.title("Average Score by User")
plt.xlabel("Proportion of Questions Answered Correctly")
plt.ylabel("Frequency")
plt.savefig(out_path+os.path.normpath('/avg_score.pdf'))
plt.clf()

# any relationship between test length and num correct?
# there does seem to be a slight positive relationship between number of qs asked
# and score.
user_sum.plot(y='mean',x='count',ls='',marker='.')
results = sm.ols(formula='mean~count', data = user_sum).fit()
results.summary()
intercept = results.params[0]
slope = results.params[1]
plt.plot(user_sum['count'], intercept + slope*user_sum['count'],'-')
plt.title("Average Score vs. Number of Questions Answered by Student")
plt.xlabel("Number of Questions")
plt.ylabel("Proportion of Questions Answered Correctly")
plt.savefig(out_path+os.path.normpath('/count_vs_score.pdf'))
plt.clf()

user_sum[user_sum['count']>30].plot(y='mean',x='count',ls='',marker='.')
results = sm.ols(formula='mean~count', data = user_sum[user_sum['count']>30]).fit()
results.summary()
intercept = results.params[0]
slope = results.params[1]
plt.plot(user_sum['count'], intercept + slope*user_sum['count'],'-')
plt.title("Average Score vs. Number of Questions Answered by Student, N>30")
plt.xlabel("Number of Questions")
plt.ylabel("Proportion of Questions Answered Correctly")
plt.savefig(out_path+os.path.normpath('/count_vs_score_30.pdf'))
plt.clf()

### Duplicate questions for a user
# do any users answer duplicate questions?
# yes, they do.  for some analyses, we should discard these.
# These will seem more consistent than they actually are. 
temp_avg = data.groupby(by=['question_id','user_id']).agg(['mean','count'])
dup_questions = temp_avg[temp_avg.correct['count'] > 1]
len(dup_questions)

# of the duplicate questions, is there any variance in answer?
# yes there is.  This could mean one of two things:
# 1. These people dont know the anser, then guessed differently on the
#    two iterations of the question to max their chance of getting one right
# 2. Human error.
dup_questions['diff_answers'] = (dup_questions.correct['mean'] < 1) & (dup_questions.correct['mean'] > 0)
dup_questions.diff_answers.max()

# merge duplicate questions onto primary dataset
dup_questions['dup'] = True
dup_questions = dup_questions.drop('correct',1)
dup_questions.columns = dup_questions.columns.get_level_values(0)
data = pd.merge(data,dup_questions, how='left', left_on=['question_id','user_id'],right_index=True).fillna(value=False)

# first drop all user*questions with different answers,
# and one copy of user*questions with same answers
filtered_data = data[data.diff_answers == False]
filtered_data = filtered_data[filtered_data.duplicated(subset=['user_id','question_id'])==False]
filtered_data = filtered_data.drop(['dup','diff_answers'], axis = 1)

correct_avg = filtered_data.groupby(by='question_id')['correct'].agg(['mean','count'])
correct_avg.columns = ['avg_correct','total']

################
### ANALYSIS ###
################

### Confidence intervals
# 1. conf. intervals:  mark any qs we are not sure about in terms of difficulty.
# calculate the exact conf. interval
correct_avg['num_correct'] = correct_avg['total']*correct_avg['avg_correct']

correct_avg['f1'] = stats.f.ppf(
    .975,
    dfn = 2*(correct_avg.total - correct_avg.num_correct+1),
    dfd = (2*correct_avg.num_correct)
    )

correct_avg['f2'] = stats.f.ppf(
    .975,
    dfn = (2*(correct_avg.num_correct+1)),
    dfd = (2*(correct_avg.total - correct_avg.num_correct))
    )

correct_avg['CI_LB'] = (
    1 / (
        1+ correct_avg.f1*(
            (correct_avg.total - correct_avg.num_correct+1)/correct_avg.num_correct
            )
        )
    )

correct_avg['CI_LB'] = correct_avg['CI_LB'].fillna(value=0)

correct_avg['CI_UB'] = (
    correct_avg.f2*(correct_avg.num_correct+1)/
    (
        (correct_avg.total - correct_avg.num_correct)*
        (1 + correct_avg.f2*(correct_avg.num_correct+1)/
            (correct_avg.total - correct_avg.num_correct)
        )
    )
    )

correct_avg['CI_UB'] = correct_avg['CI_UB'].fillna(value=1)

correct_avg = correct_avg.sort(columns = ['CI_UB','CI_LB'])

### Create the results matrix to hold all of the analyses
results = DataFrame(correct_avg[['CI_UB','CI_LB','avg_correct', 'total','num_correct']],columns=['CI_UB','CI_LB','avg_correct', 'total','num_correct']+
                    ['good_'+str(int(x*100)) for x in STUDENT_QUANTILES]+
                    ['bad_'+str(int(x*100)) for x in STUDENT_QUANTILES])

### Percentiles
# we want to know the students' percentiles in terms of their average scores - we want to choose questions that
# are highly discriminating at these points, so we can sort the students who take the midterm

# throw students who answer very few questions, may skew the percentiles
users=filtered_data.groupby(by='user_id')['correct'].agg(['mean','count'])
count_percentiles = users['count'].quantile(q=np.arange(.05,1,.05))
users=users[users['count']>count_percentiles[.05]]
filtered_data = filtered_data[filtered_data.user_id.isin(users.index)]

# Get the score percentiles
# seems to be a cutoff around .2 for the scores - evidence of guessing?
users[users['mean']<users['mean'].quantile(q=.05)]
users[users['mean']<users['mean'].quantile(q=.95)]
score_percentiles = users['mean'].quantile(q=STUDENT_QUANTILES)

# now, at each decile, we want to caluclate each question's discrimination
# or p(correct | good student) / p(correct | bad student)
# we dont have these probabilities, so we take them from average scores in the data.

# first we establish the various cutoffs for users
users =users.sort(columns='mean')
users['percentile'] = users['mean'].rank(ascending=True)/len(users)
users['user_id']=users.index

# merge the percentiles backto the users
filtered_data= filtered_data.merge(users[['user_id','percentile']], on='user_id',how='left')

for qid in results.index:
    # get users' answers to a given question, and their average score
    answers = filtered_data[filtered_data.question_id ==qid][['user_id','correct']]
    answers.columns = ['user_id','answer']
    users_subset = users.merge(answers,how='inner',on='user_id')

    #small adjustment to mean to remove the effect of the question being analized
    users_subset['mean'] = (users_subset['mean']*
                            users_subset['count']-
                            users_subset['answer'])/(users_subset['count']-1)
    
    for quant in STUDENT_QUANTILES:
        quant2 = score_percentiles[quant]
        means = users_subset.groupby(by=[users_subset['percentile'] > quant2]).agg({'answer':'mean'})
        t = str(int(quant*100))
        try:
            prob_good =means.get_value(True,'answer')
            results.set_value(qid,'good_'+t,prob_good)
        except:
            pass
        try:
            prob_bad = means.get_value(False,'answer')
            results.set_value(qid,'bad_'+t,prob_bad)
        except:
            pass

### Plot the resulting ratios
for quant in STUDENT_QUANTILES:
    t = str(int(quant*100))
    plt.plot(results['bad_'+t],results['good_'+t],'b.')
    plt.plot(np.arange(0,1.1,.1),np.arange(0,1.1,.1),'g-',alpha=.5)
    plt.title("Discrimination: "+t+"th Percentile")
    plt.ylabel("Proportion Right, Good Students")
    plt.xlabel("Proportion Right, Bad Students")
    plt.savefig(out_path+os.path.normpath('/discrim')+t+'.pdf')
    plt.clf()

#######################
### Descicion Rules ###
#######################

# where we will keep the rejection results
# start rejecting all, select the good ones
results['keep'] = False

### Discrimination
# remove questions that do the worst at discrimination in each quantile, ordered by odds ratio
for quant in STUDENT_QUANTILES:
    t = str(int(quant*100))    
    results['disc_'+t]= ((results['good_'+t])/
                         (1-results['good_'+t]))*(
                             (1- results['bad_'+t])/
                             (results['bad_'+t]))
    results['disc_'+t][~np.isfinite(results['disc_'+t])]=np.nan
    results.keep = (results.keep | (   (results['disc_'+t]>results['disc_'+t].quantile(DISCRIMINATION_QUANTILE_SELECTION)) ) )

### Confidence interval
# we remove questions with a wide CI:
# If we do not know how difficult the question is, then
# we should not include it
# also eliminates small sample sizes.
results['keep'] = ~((~results.keep) | (results['CI_UB']-results['CI_LB'] > MAX_CI_WIDTH))
results['reject']=~results['keep']

final_selection = results.index[results.keep]
num_chosen = len(final_selection)

#################################
### Comparison With IRT Model ###
#################################

try:
    model = pd.read_csv(model_path)
    model[model.Dffclt <10].Dffclt.hist(bins=30)
    model['ord'] = model.Dscrmn.rank()
    res = model[model.Dscrmn > model.Dscrmn.quantile((total_questions-num_chosen)/total_questions)]

    # model, analysis drop different questions
    merged= results.merge(model, left_index=True, right_on = 'question_id',how = 'outer')
    merged['model_rej'] = merged['ord']<= total_questions-num_chosen

    # how many rejected by us/model, but included in other
    mismatch = merged[(~merged['model_rej']) 
            &(merged['reject'])]

    mismatch2 = merged[(merged['model_rej']) 
            &(~merged['reject'])]

    # 39 mismatches
    len(mismatch)

    # 18 are non-confidence-interval related
    mismatch_not_CI = merged[(~merged['model_rej']) 
            &(merged['reject'])
            &(merged['CI_UB']-merged['CI_LB'] < MAX_CI_WIDTH)]

    mismatch_not_CI2 = merged[(merged['model_rej']) 
            &(~merged['reject'])
            &(merged['CI_UB']-merged['CI_LB'] < MAX_CI_WIDTH)]

    # mismatches are near median - middle 20% in terms of CI
    mismatch_not_CI.ord.describe()

    merged['mismatched']= merged.index.isin(mismatch.index) |merged.index.isin(mismatch2.index)
    merged['mismatched_not_ci']= merged.index.isin(mismatch_not_CI.index) |merged.index.isin(mismatch_not_CI2.index)
    merged['chart_order'] = merged['disc_'+t].rank()

    ### Final Plots
    quant=.5
    t = str(int(quant*100))
    merged['chart_order'] = merged['disc_'+t].rank()
    plt.clf()
    #plot order vs discrimination - showing method of rejection
    plt.plot(merged['chart_order'],merged['disc_'+t],'b.')
    plt.plot(merged[merged.reject]['chart_order'],merged[merged.reject]['disc_'+t],'y.')
    plt.plot(merged[merged['CI_UB']-merged['CI_LB'] > MAX_CI_WIDTH]['chart_order'],merged[merged['CI_UB']-merged['CI_LB'] > MAX_CI_WIDTH]['disc_'+t],'g.')
    plt.title("Question Selection and Method of Rejection")
    plt.xlabel("Order")
    plt.ylabel("Odds Ratio")
    plt.savefig(out_path+os.path.normpath('/model_drops_method.pdf'))
    plt.clf()

    #plot order vs. discrimination - showing disagreement
    plt.plot(merged['chart_order'],merged['disc_'+t],'b.')
    plt.plot(merged[merged.reject]['chart_order'],merged[merged.reject]['disc_'+t],'y.')
    plt.plot(merged[merged.mismatched]['chart_order'],merged[merged.mismatched]['disc_'+t],'r.')
    plt.title("Question Selection and Model Disagreement")
    plt.xlabel("Order")
    plt.ylabel("Odds Ratio")
    plt.savefig(out_path+os.path.normpath('/model_disagreement.pdf'))
    plt.clf()

    #plot order vs. discrimination - showing disagreement (non CI)
    plt.plot(merged['chart_order'],merged['disc_'+t],'b.')
    plt.plot(merged[merged.reject]['chart_order'],merged[merged.reject]['disc_'+t],'y.')
    plt.plot(merged[merged.mismatched_not_ci]['chart_order'],merged[merged.mismatched_not_ci]['disc_'+t],'r.')
    plt.title("Model Disagreement, No CI Mismatches")
    plt.xlabel("Order")
    plt.ylabel("Odds Ratio")
    plt.savefig(out_path+os.path.normpath('/model_disagreement_no_ci.pdf'))
    plt.clf()

    # check tail rejection
    merged= merged.sort(columns='chart_order')
    merged[['reject','mismatched','mismatched_not_ci','question_id','chart_order']]

    q=merged[['disc_'+str(int(x*100)) for x in STUDENT_QUANTILES]].quantile(q=DISCRIMINATION_QUANTILE_SELECTION)

    # check q 13274 - lonely in lower tail
    # very good at discriminating students in 90th percentile.  
    i=merged[merged.question_id ==13274 ][['disc_'+str(int(x*100)) for x in STUDENT_QUANTILES]]

    # check 165, highest mismatch - we keep, model rejects.  Good at discriminating several middle percentiles.
    i=merged[merged.question_id ==165 ][['disc_'+str(int(x*100)) for x in STUDENT_QUANTILES]]
except:
    print "Warning:  IRT Model Results not found. Make sure the file is at "+model_path
    

##############
### EXPORT ###
##############

DataFrame(final_selection).to_csv(path_or_buf=out_path+os.path.normpath('/final_selection.csv'))

plt.close('all')
