###
# Density - Queue Simulation

'''
This code simulates a random queue subject to randomly happening deterioration. The system has infinite servers.

Please change inputs to verify the related results.

Assumption is that next arrival and departure times are resampled in the case of a change of system behavior.

'''

import numpy as np
import matplotlib.pyplot as plt

# Inputs (rates)

time_for_transient_pmf = 100
arrival_good=30
arrival_bad=30
service_good=20
service_bad=10
fail_rate=0.00002
repair_rate=0.00001

sim_time=100
initial_cust=0 # Number of initial customers in system
total_runs=1000
step_size=0.01 # Difference between times
initial_cond = 0 # 0 for good, 1 for bad


# Organizing parameters

total_mean=[]
arrival_time=1/arrival_good
mean_serv_time=1/service_good
arrival_time_bad=1/arrival_bad
mean_serv_time_bad=1/service_bad
run=total_runs
list_cust_per_run=[]
condition = initial_cond
var=time_for_transient_pmf

# Main

while run>0:

	print("Runs left: ", run)
	num_cust=[]
	num_cust.append(initial_cust)
	t=0
	t_dep=[sim_time*10000] #support upper bound

	# Creating departure times for initial customers

	if initial_cust>0:
	  for item in range(0, initial_cust):
	    if initial_cond==0:
	      t_dep.append(t + np.random.exponential(mean_serv_time))
	    else:
	      t_dep.append(t + np.random.exponential(mean_serv_time_bad))

	current_cust = initial_cust
	next_dep = [min(t_dep), t_dep.index(min(t_dep))]
	next_arrival = 0
	time_since_last_change_in_sytem = 0

	while t < sim_time: #verifying end of simulation

# Checking current situation of the system (0 for good, 1 for bad)

		if condition == 0:
			aux= 1 - np.exp(- fail_rate * (t - time_since_last_change_in_sytem))
			s = np.random.uniform(0,1)
			if s <= aux: # Checking whether system is changing
				condition = 1
				next_arrival = t + np.random.exponential(arrival_time_bad)
				t_dep_aux = [(t + np.random.exponential(mean_serv_time_bad)) for item in t_dep[:-1]] # Rescheduling all except the support upper bound one
				t_dep = t_dep_aux
				t_dep.append(sim_time*10000)
				next_dep = [min(t_dep), t_dep.index(min(t_dep))]
				time_since_last_change_in_sytem = mean_serv_time

		elif condition == 1:
			aux= 1 - np.exp(-repair_rate * (t - time_since_last_change_in_sytem))
			s = np.random.uniform(0,1) 
			if s <= aux: # Checking whether system is changing
				condition = 0
				next_arrival = t + np.random.exponential(arrival_time)
				t_dep_aux = [(t + np.random.exponential(mean_serv_time_bad)) for item in t_dep[:-1]] # Rescheduling all except the support upper bound one
				t_dep = t_dep_aux
				t_dep.append(sim_time*10000)
				next_dep = [min(t_dep), t_dep.index(min(t_dep))]
				time_since_last_change_in_sytem = t


# When the system is in good condition

		if condition == 0:
			while next_arrival <= t:
				current_cust += 1
				t_dep.append(t + np.random.exponential(mean_serv_time))
				next_dep = [min(t_dep), t_dep.index(min(t_dep))]
				next_arrival = t + np.random.exponential(arrival_time)

			while next_dep[0] <= t:
				current_cust -= 1
				t_dep.pop(int(next_dep[1]))
				next_dep = [min(t_dep), t_dep.index(min(t_dep))]

# When the system is in bad condition

		if condition == 1:
			while next_arrival <= t:
				current_cust += 1
				t_dep.append(t + np.random.exponential(mean_serv_time_bad))
				next_dep = [min(t_dep), t_dep.index(min(t_dep))]
				next_arrival = t + np.random.exponential(arrival_time_bad)

			while next_dep[0] <= t:
				current_cust -= 1
				t_dep.pop(int(next_dep[1]))
				next_dep = [min(t_dep), t_dep.index(min(t_dep))]

# Continue the simulation

		t += step_size
		num_cust.append(current_cust)

# Going to next run

	total_mean.append(np.mean(num_cust))
	list_cust_per_run.append(num_cust)
	run -= 1

list_transient = np.mean(list_cust_per_run,axis=0) # List with expected number of customers per t
print("mean of steady_state ", total_runs, ": ", np.mean(total_mean))

#######################################################
# PLOTS

# Plotting transient expected values for different t's

x_ax=np.arange(0, t, step_size)
plt.xlabel('t')
plt.ylabel('Expected Number of Customers')
plt.title('Expected Number of Customers for %d initial customers at time t' %(initial_cust))
plt.plot(x_ax, list_transient, 'tab:red', ms=0.1)
plt.show()


aux_list = [item[var] for item in list_cust_per_run]
trans_prob=[]

for i in range(0, max(aux_list)+1):
  count = [item for item in aux_list if item == i]
  trans_prob.append(len(count))

trans_prob = [item/total_runs for item in trans_prob]
print(trans_prob)

# Plotting transient probabilities for specific t

x_ax=np.arange(0, t, step_size)
plt.xlabel('Number of Customers')
plt.ylabel('Probability')
plt.title(' PMF at time %f for %d initial customers' %(var, initial_cust))
plt.bar(list(range(0,len(trans_prob))) , trans_prob, align='center', alpha=0.5)
plt.show()
