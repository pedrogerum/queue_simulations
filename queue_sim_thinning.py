
# Density - Queue Simulation

'''
This code simulates a random queue subject to randomly happening deterioration. The system has infinite servers.

Please change inputs to verify the related results.

Assumption is that next arrival and departure times are resampled in the case of a change of system behavior.

'''

import numpy as np
import matplotlib.pyplot as plt


def thinning(upper_bound_arr, arrival_rate, upper_bound_serv, service_rate, T, step_size):

	t = 0

	thin_prob_arr = np.array(arrival_rate)/upper_bound_arr
	thin_prob_serv = np.array(service_rate)/upper_bound_serv
	count_arr = 0
	count_dep = 0
	time_event_arr = []
	time_event_dep = []


	while t<T:
		# thinning process for the arrival
		u_1 = np.random.uniform()
		t = t - np.log(u_1)/upper_bound_arr
		if t>T:
			break
		u_2 = np.random.uniform()
		elem_to_check = int(t/step_size)-1
		if u_2 < arrival_rate[elem_to_check]:
			count_arr+=1
			time_event_arr.append(t)

			# for each arrival run a thinning algorithm for the correspondind departure time
			t_dep = t
			while len(time_event_dep) < len(time_event_arr):
				u_3 = np.random.uniform()
				t_dep = t_dep - np.log(u_3)/upper_bound_serv
				u_4 = np.random.uniform()
				if t_dep<T:
					elem_to_check = int(t_dep/step_size)-1
				else:
					elem_to_check = -1
				if u_4 < service_rate[elem_to_check]:
					count_dep+=1
					time_event_dep.append(t_dep)
		


	return time_event_arr, count_arr, time_event_dep, count_dep


# Inputs (rates)

time_for_transient_pmf = 100
arrival_good=30
arrival_bad=30
service_good=20
service_bad=10
fail_rate=0.02
repair_rate=0.01

sim_time=100
initial_cust=0 # Number of initial customers in system
total_runs=1000
step_size=1 # Difference between times
initial_cond = 0 # 0 for good, 1 for bad

# Main
#scheduling change points
t = 0
t_change = [0]
arrival_rate = []
service_rate = []
condition_step = []
condition = 0



while t < sim_time:
		if condition==0:
			t = t + np.random.exponential(1/fail_rate)
			t_change.append(t)
			condition = 1
		else:
			t = t + np.random.exponential(t + np.random.exponential(1/repair_rate))
			t_change.append(t)
			condition = 0 

print(t_change)
t = 0 


for step in range(int(sim_time/step_size)):
	item = step * step_size
	
	for i, time in enumerate(t_change[1:]):
		if (i+1)%2 == 0:
			condition = initial_cond
		else:
			condition = abs(initial_cond-1)

		if item <= time and item > t_change[i]:

			if condition == 0 :
				arrival_rate.append(arrival_good)
				condition_step.append(0)
				service_rate.append(service_good)
			else:
				arrival_rate.append(arrival_bad)
				condition_step.append(1)
				service_rate.append(service_bad)



time_event_arr, count_arr, time_event_dep, count_dep = thinning(arrival_good, arrival_rate, service_good, service_rate, sim_time, step_size)

print(time_event_arr[0])

customers = [10000]
count = 0
number_of_customer = 0
next_arrival = time_event_arr[count]
next_dep = time_event_dep[count]

total_customers = []

for step in range(int(sim_time/step_size)):
	time = step_size*step
	print(time)
	print(next_arrival)

	if time > next_arrival:
		customers.append(next_dep)
		number_of_customer += 1
		count += 1
		next_arrival = time_event_arr[count]
		next_dep = time_event_dep[count]
		print(count)

	if time > min(customers):
		customers.pop(customers.index(min(customers)))
		number_of_customer -= 1

	total_customers.append(number_of_customer)
	

print(total_customers)






# list_transient = np.mean(list_cust_per_run,axis=0) # List with expected number of customers per t
# print("mean of steady_state ", total_runs, ": ", np.mean(total_mean))

# #######################################################
# # PLOTS

# # Plotting transient expected values for different t's

# x_ax=np.arange(0, t, step_size)
# plt.xlabel('t')
# plt.ylabel('Expected Number of Customers')
# plt.title('Expected Number of Customers for %d initial customers at time t' %(initial_cust))
# plt.plot(x_ax, list_transient, 'tab:red', ms=0.1)
# plt.show()


# aux_list = [item[var] for item in list_cust_per_run]
# trans_prob=[]

# for i in range(0, max(aux_list)+1):
#   count = [item for item in aux_list if item == i]
#   trans_prob.append(len(count))

# trans_prob = [item/total_runs for item in trans_prob]
# print(trans_prob)

# # Plotting transient probabilities for specific t

# x_ax=np.arange(0, t, step_size)
# plt.xlabel('Number of Customers')
# plt.ylabel('Probability')
# plt.title(' PMF at time %f for %d initial customers' %(var, initial_cust))
# plt.bar(list(range(0,len(trans_prob))) , trans_prob, align='center', alpha=0.5)
# plt.show()


# # Organizing parameters

# total_mean=[]
# arrival_time=1/arrival_good
# mean_serv_time=1/service_good
# arrival_time_bad=1/arrival_bad
# mean_serv_time_bad=1/service_bad
# run=total_runs
# list_cust_per_run=[]
# condition = initial_cond
# var=time_for_transient_pmf

# # Main



