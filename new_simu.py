import numpy as np
import matplotlib.pyplot as plt
from math import ceil


# Constants
ALPHA = 2
BOND = 4
INITIAL_STAKE = 10000
INITIAL_BUFFER = 1000
TARGET = INITIAL_BUFFER
SIMULATION_DAYS = 10000
DAILY_ACTIONS = 24  # Simulate hourly actions
healthy_buffer = 0.8
linear_health_function = 0
normal_percentile = 0.1 # we want the buffer to keep within +/- 0.1 of the buffer

class BufferSystem:
    def __init__(self, initial_buffer, initial_target, initial_stake, alpha):
        self.buffer = initial_buffer
        self.target = initial_target
        self.total_amount = initial_stake + initial_buffer
        self.buffer_history = []
        self.enqueued_requests = 0
        self.alpha = alpha

    def get_buffer(self):
        return self.buffer

    def get_target(self):
        return self.target

    def set_target(self, new_target):
        self.target = new_target

    def get_enqueued_requests(self):
        return self.enqueued_requests
    
            
    def record_buffer(self):
        self.buffer_history.append(self.buffer)
    def buffer_health(self, buffer_amount):
        """Calculate buffer health as a ratio of buffer to target."""
        return min(1.0, buffer_amount / self.target) if self.target > 0 else 0
      
    # def buffer_health(self, b):
    #     if b <= 0 or self.target <= 0:
    #         return 0  # Avoid division by zero
    #     ratio = b / self.target
    #     if ratio >= 1:
    #         return ratio  # If b >= target, just return the ratio
    #     if linear_health_function:
    #         return ratio
    #     else:
    #         # Handle potential overflow by checking the exponent first
    #         log_ratio = np.log(ratio)
    #         if self.alpha * log_ratio > 709:  # np.log(np.finfo(np.float64).max) ≈ 709
    #             return ratio
    #         else:
    #             return min(np.exp(self.alpha * log_ratio), ratio)
    
    def deposit(self, amount):
      # when a new deposit is made, should this go to the buffer
      # or to the staked amount?
      # we put it in the buffer for now
        self.total_amount += amount
        if self.buffer + amount > (1+ normal_percentile) * self.target:
            self.trigger_stake(amount)
        else:
            self.buffer += amount

    def trigger_stake(self, amount):
        if self.buffer + amount > (1+ normal_percentile) * self.target:
            # stake anything above threshold
            staked_amount = self.buffer + amount - (1+ normal_percentile) * self.target
            self.buffer = (1+ normal_percentile) * self.target
            #return staked_amount
       # return 0
    def withdraw(self, amount):
        if amount < self.total_amount: # stop withdrawal if not enough stake
            health = self.buffer_health(self.buffer - amount)
            if health > healthy_buffer:
                actual_withdrawal = min(amount, self.buffer)  # Ensure we don't withdraw more than available
            else:
                actual_withdrawal = min(amount * (0.2 + health), self.buffer)
            if amount > self.buffer: # not enough in buffer to fulfil the withdrawal, the request is queued
                self.enqueued_requests += amount - actual_withdrawal
                #print("queued requests")
            self.buffer -= actual_withdrawal
            self.total_amount -= actual_withdrawal
            return actual_withdrawal
        else:
            return 0 # nothing was withdraw

    def update_target(self):
        #avg_health = np.mean(self.health_history[-24:]) if len(self.health_history) >= 24 else np.mean(self.health_history)
        # check if buffer falls within +/- stable_percentage of target

        if self.buffer < self.target*(1-normal_percentile):
            # if the buffer is below this limit -> we are too low and the
            # target was set too low, we increase it
            #self.target += self.target * (1 - avg_health)
            self.target = (self.target - self.buffer) / normal_percentile
        elif self.buffer > self.target*(1+normal_percentile):
            # the buffer is too high we can stake more and decrease the target 
            self.target = (self.buffer - self.target) / (1+normal_percentile)
        # Ensure target doesn't exceed total_amount
        self.target = min(self.target, self.total_amount)

        return self.target

    def stake(self):
        if self.buffer > self.target:
            staked_amount = self.buffer - self.target
            self.buffer = self.target
            return staked_amount
        return 0

    def top_up_buffer(self):
        if self.buffer < self.target:
            needed_amount = self.target - self.buffer
            if needed_amount <= self.total_amount - self.buffer:
                validators_to_exit = ceil(needed_amount / (32 - BOND))
                unstaked_amount = validators_to_exit * (32 - BOND)
                self.buffer += unstaked_amount
                return unstaked_amount
        return 0

def run_simulation(buffer_system):
    buffer_history = []
    target_history = []
    withdrawal_history = []
    deposit_history = []
    stake_history = []
    unstake_history = []
    total_amount_history = []


    for day in range(SIMULATION_DAYS):
        for _ in range(DAILY_ACTIONS):
            # Record current state
            buffer_history.append(buffer_system.get_buffer())
            target_history.append(buffer_system.get_target())

            # Random deposit or withdrawal
            action = np.random.choice(['deposit', 'withdraw'], p=[0.5, 0.5])
            #amount = max(0, np.random.normal(32, 20))  # Ensure non-negative amount
            amount = max(0,np.random.lognormal(mean=3, sigma=1))  # mu=3 would give median around 20



            if action == 'deposit':
                buffer_system.deposit(amount)
                deposit_history.append(amount)
                withdrawal_history.append(0)
            else:
                withdrawn = buffer_system.withdraw(amount)
                withdrawal_history.append(withdrawn)
                deposit_history.append(0)

            # Record buffer health
            #buffer_system.health_history.append(buffer_system.buffer_health(buffer_system.get_buffer()))
            buffer_system.record_buffer()
            total_amount_history.append(buffer_system.total_amount)  # record total stake


        # Update target at the end of the day
        new_target = buffer_system.update_target()

        # New target updated -> now update buffer
        # Stake or unstake based on new target -> should this be here or in the daily actions loop?
        # We probably don't want to do it too aften so let's leave it here
        if buffer_system.get_buffer() > new_target:
            staked = buffer_system.stake() # this update the value of the buffer
            stake_history.append(staked)
            unstake_history.append(0)
        elif buffer_system.get_buffer() < new_target:
            unstaked = buffer_system.top_up_buffer()
            unstake_history.append(unstaked)
            stake_history.append(0)
        else:
            stake_history.append(0)
            unstake_history.append(0)

    total_enqueued = buffer_system.get_enqueued_requests()

    return buffer_history, target_history, withdrawal_history, deposit_history, stake_history, unstake_history, total_amount_history, total_enqueued


############ Run  simulations for different values of alpha #############
def analyze_alpha_range(alpha_range):

    ## We define lists for our efficiency metrics
    staking_efficiencies = []
    withdrawal_efficiencies = []

    for alpha in alpha_range:
        buffer_system = BufferSystem(INITIAL_BUFFER, TARGET, INITIAL_STAKE, alpha)
        buffer_history, target_history, withdrawal_history, deposit_history, stake_history, unstake_history, total_amount_history, total_enqueued = run_simulation(buffer_system)

        withdrawals = sum(withdrawal_history)
        queued_withdrawals = total_enqueued
        avg_buffer = np.mean(buffer_history)
        avg_total = np.mean(total_amount_history)

        withdrawal_efficiency = withdrawals / (withdrawals + queued_withdrawals)
        staking_efficiency = (avg_total - avg_buffer) / avg_total
        # vec3 = np.empty_like(total_amount_history)
        # vec3 = np.divide(np.subtract(total_amount_history, buffer_history, out=vec3), total_amount_history, out=vec3)
        # #staking_efficiency = np.mean((total_amount_history - buffer_history)/total_amount_history)
        # staking_efficiency = np.mean(vec3)
        # buffer_array = np.array(buffer_history)
        # target_array = np.array(target_history)
        # total_array = np.array(total_amount_history)
    
        # # 1. Calculate what percentage of total funds is staked
        # staked_ratio = (total_array - buffer_array) / total_array
   

        staking_efficiencies.append(staking_efficiency)
        withdrawal_efficiencies.append(withdrawal_efficiency)

    return staking_efficiencies, withdrawal_efficiencies

# Run analysis for different alpha values
alpha_range = np.arange(0.5, 5, 0.5)
staking_efficiencies, withdrawal_efficiencies = analyze_alpha_range(alpha_range)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(alpha_range, staking_efficiencies, label='Staking Efficiency', marker='o')
plt.plot(alpha_range, withdrawal_efficiencies, label='Withdrawal Efficiency', marker='s')
plt.xlabel('Alpha')
plt.ylabel('Efficiency')
plt.title('Staking and Withdrawal Efficiency vs Alpha')
plt.legend()
plt.grid(True)
plt.show()

# Print results
print("Alpha\tStaking Efficiency\tWithdrawal Efficiency")
for alpha, staking_eff, withdrawal_eff in zip(alpha_range, staking_efficiencies, withdrawal_efficiencies):
    print(f"{alpha:.1f}\t{staking_eff:.4f}\t\t{withdrawal_eff:.4f}")