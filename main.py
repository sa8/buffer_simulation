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

class BufferSystem:
    def __init__(self, initial_buffer, initial_target, initial_stake, alpha):
        self.buffer = initial_buffer
        self.target = initial_target
        self.total_amount = initial_stake + initial_buffer
        self.health_history = []
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

    def buffer_health(self, b):
        if b <= 0 or self.target <= 0:
            return 0  # Avoid division by zero
        ratio = b / self.target
        if ratio >= 1:
            return ratio  # If b >= target, just return the ratio
        if linear_health_function:
            return ratio
        else:
            # Handle potential overflow by checking the exponent first
            log_ratio = np.log(ratio)
            if self.alpha * log_ratio > 709:  # np.log(np.finfo(np.float64).max) ≈ 709
                return ratio
            else:
                return min(np.exp(self.alpha * log_ratio), ratio)
    
            
    def record_health(self):
        current_health = self.buffer_health(self.buffer)
        self.health_history.append(current_health)
        return current_health

    def deposit(self, amount):
      # when a new deposit is made, should this go to the buffer
      # or to the staked amount?
        self.buffer += amount
        self.total_amount += amount
        # check if we are above a threshold (we need 32 eth + target before staking)
        # if self.buffer > TARGET + 32:
        #     self.stake()
    def withdraw(self, amount):
            if amount <= self.total_amount:  # Check if we have enough total funds
                current_health = self.buffer_health(self.buffer)
                
                # Calculate maximum withdrawal based on health
                if current_health > healthy_buffer:
                    max_withdrawal = self.buffer
                else:
                    # Apply slippage when health is low
                    slippage_factor = 0.2 + current_health
                    max_withdrawal = min(amount * slippage_factor, self.buffer)
                
                # If requested amount exceeds what we can fulfill
                if amount > max_withdrawal:
                    # Queue the unfulfilled portion
                    unfulfilled = amount - max_withdrawal
                    self.enqueued_requests += unfulfilled
                    
                    # Process the withdrawal we can fulfill
                    self.buffer -= max_withdrawal
                    self.total_amount -= max_withdrawal
                    return max_withdrawal
                else:
                    # Can fulfill entire request
                    self.buffer -= amount
                    self.total_amount -= amount
                    return amount
                    
            return 0  # Return 0 if we don't have enough total funds

    def update_target(self):
        # Get average health over last 24 periods (or all if less than 24)
        lookback = min(24, len(self.health_history))
        if lookback == 0:
            return self.target
            
        recent_health = self.health_history[-lookback:]
        avg_health = np.mean(recent_health)
        
        if avg_health < 1:
            # Buffer is consistently below target - increase target proportionally
            # The lower the health, the larger the increase
            increase_factor = 1 + (1 - avg_health)
            self.target *= increase_factor
        elif avg_health > 1:
            # Buffer is consistently above target - decrease target
            # The higher the health, the larger the decrease
            decrease_factor = 1 / avg_health
            self.target *= decrease_factor
            
        # Additional constraints
        # 1. Target shouldn't exceed total funds
        self.target = min(self.target, self.total_amount)
        
        # 2. Target shouldn't fall below minimum safety threshold
        min_target = INITIAL_BUFFER * 0.5  # Example minimum threshold
        self.target = max(self.target, min_target)
        
        # 3. Smooth large changes (optional)
        max_change = 0.2  # Maximum 20% change per update
        old_target = self.target
        if self.target > old_target * (1 + max_change):
            self.target = old_target * (1 + max_change)
        elif self.target < old_target * (1 - max_change):
            self.target = old_target * (1 - max_change)
            
        return self.target
    # def withdraw(self, amount):
    #     if amount < self.total_amount: # stop withdrawal if not enough stake
    #         health = self.buffer_health(self.buffer - amount)
    #         if health > healthy_buffer:
    #             actual_withdrawal = min(amount, self.buffer)  # Ensure we don't withdraw more than available
    #         else:
    #             actual_withdrawal = min(amount * (0.2 + health), self.buffer)
    #         if amount > self.buffer: # not enough in buffer to fulfil the withdrawal, the request is queued
    #             self.enqueued_requests += amount - actual_withdrawal
    #             #print("queued requests")
    #         self.buffer -= actual_withdrawal
    #         self.total_amount -= actual_withdrawal
    #         return actual_withdrawal
    #     else:
    #         return 0 # nothing was withdraw

    # def update_target(self):
    #     avg_health = np.mean(self.health_history[-24:]) if len(self.health_history) >= 24 else np.mean(self.health_history)
    #     if avg_health < 1:
    #         self.target += self.target * (1 - avg_health)
    #     elif avg_health > 1:
    #         self.target /= avg_health
    #     # Ensure target doesn't exceed total_amount
    #     self.target = min(self.target, self.total_amount)

    #     return self.target

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
            buffer_system.record_health()
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

        staking_efficiencies.append(staking_efficiency)
        withdrawal_efficiencies.append(withdrawal_efficiency)

    return staking_efficiencies, withdrawal_efficiencies

# Run analysis for different alpha values
alpha_range = np.arange(1, 3, 0.2)
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