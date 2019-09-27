import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, binom, lognorm
from scipy.optimize import minimize
from math import log, sqrt

# FUNCTION: collapses_per_IM ---------------------------------------------------
# reads dakotaTab to extract number of collapses and simulations for each IM
# ------------------------------------------------------------------------------

def collapses_per_IM(dT_path):
	# initialize lists of outputs
	num_records = []
	num_collapses = []

	# read
	df = pd.read_csv(dT_path,delim_whitespace=True)

	# collect IM values in set
	all_IMs = []
	for ID in df['MultipleEvent']:
		all_IMs.append(ID.split('_')[0])
	IM = list(set(all_IMs))

	# collect number of records for each IM
	for m in IM:
		count = 0
		for n in all_IMs:
			if n == m:
				count += 1
		num_records.append(count)

	# collect number of collapses in each IM
	PID_columns = [col for col in list(df) if 'PID' in col] # list of column headers with PID
	num_rows = df.shape[0]
	for m in IM: # iterate through each IM
		count = 0
		# iterate through each row which corresponds to IM level m
		for row in range(num_rows):
			collapse_flag = 0
			if df['MultipleEvent'][row].split('_')[0] == m:
				# iterate through each column with PID EDPs
				for col in PID_columns:
					if df[col][row] >= 0.20: # If PID at any story exceeds collapse limit, is considered a collapse case
						collapse_flag = 1
				if collapse_flag == 1: # count number of collapse cases
					count += 1
		num_collapses.append(count)

	IM = [float(m) for m in IM]
	return IM, num_records, num_collapses



# FUNCTION: neg_log_likelihood -------------------------------------------------
# objective function for evaluating negative log likelihood of observing the given collapses
# ------------------------------------------------------------------------------

def neg_log_likelihood(params, IM, num_records, num_collapses):
	theta = params[0]
	beta = params[1]

	log_IM = [log(m) for m in IM]
	p = norm.cdf(log_IM, loc=theta, scale=beta)

	# likelihood of observing num_collapse(i) collapses, given num_records observations, using the current parameter estimates
	likelihood = np.maximum(binom.pmf(num_collapses, num_records, p),
							np.nextafter(0,1))

	neg_loglik = -np.sum(np.log(likelihood))

	return neg_loglik



# FUNCTION: lognormal_MLE ------------------------------------------------------
# returns maximum likelihood estimation (MLE) of lognormal fragility function parameters
# ------------------------------------------------------------------------------
# algorithm obtained from Baker, J. W. (2015). “Efficient analytical fragility function fitting
# using dynamic structural analysis.” Earthquake Spectra, 31(1), 579-599.

def lognormal_MLE(IM,num_records,num_collapses):
	# initial guess for parameters
	params0 = [np.log(1.0), 0.4]
	#params = minimize(neg_log_likelihood, params0, args=(IM, num_records, num_collapses), method='Nelder-Mead',
    #					options={'maxfev': 400*2,
	#						 'adaptive': True})

	params = minimize(neg_log_likelihood, params0, args=(IM, num_records, num_collapses), bounds=((None, None), (1e-10, None)))
	theta = np.exp(params.x[0])
	beta = params.x[1]

	return theta, beta



# FUNCTION: update_collapsep ---------------------------------------------------
# creates copy of BIM.json for each IM with updated collapse probability
# ------------------------------------------------------------------------------

def update_collapsep(BIMfile, RPi, theta, beta, num_collapses):
	with open(BIMfile, 'r') as f:
		BIM = json.load(f)
		Pcol = norm.cdf(np.log(num_collapses/theta)/beta)
		BIM['LossModel']['BuildingResponse']['CollapseProbability'] = Pcol
	f.close()

	outfilename = 'BIM_{}.json'.format(RPi)
	with open(outfilename, 'w') as g:
		json.dump(BIM,g,indent=4)

	return outfilename