import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Constants remain the same
ALPHA = 2
BOND = 4
INITIAL_STAKE = 10000
INITIAL_BUFFER = 5
INITIAL_TARGET = 5
SIMULATION_DAYS = 10000
DAILY_ACTIONS = 24  # Simulate hourly actions
healthy_buffer = 0.8
linear_health_function = 0
normal_percentile = 0.5
BASE_TARGET_PERCENTAGE = 20  # New constant: base target as 20% of total


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
      # this depends on the value of the buffer
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
        if amount <= self.total_amount:  # stop withdrawal if not enough total stake
            health = self.buffer_health(self.buffer)  # Check current buffer health
            
            if health > healthy_buffer:
                max_withdrawal = self.buffer  # Maximum we can withdraw from buffer
                if amount <= max_withdrawal:
                    # Can fulfill completely from buffer
                    self.buffer -= amount
                    self.total_amount -= amount
                    return amount
                else:
                    # Can only partially fulfill from buffer
                    fulfilled = max_withdrawal
                    remaining = amount - fulfilled
                    self.enqueued_requests += remaining  # Queue the remainder
                    self.buffer = 0  # Buffer is emptied
                    self.total_amount -= fulfilled
                    return fulfilled
            else:
                # Apply slippage to requested amount
                slipped_amount = amount * (0.2 + health)
                max_withdrawal = self.buffer
                
                if slipped_amount <= max_withdrawal:
                    # Can fulfill slipped amount from buffer
                    self.buffer -= slipped_amount
                    self.total_amount -= slipped_amount
                    return slipped_amount
                else:
                    # Can only partially fulfill slipped amount
                    fulfilled = max_withdrawal
                    remaining = slipped_amount - fulfilled
                    self.enqueued_requests += remaining  # Queue the remainder
                    self.buffer = 0  # Buffer is emptied
                    self.total_amount -= fulfilled
                    return fulfilled
        return 0  # Nothing was withdrawn if total amount insufficient



    def update_target(self):
        base_target = (self.total_amount * BASE_TARGET_PERCENTAGE) / 100
        health = self.buffer_health(self.buffer)
        
        if health < 1:
            increase = (base_target * (1 - health) * 150) / 100
            self.target = base_target + increase
        elif health > 1:
            decrease = (base_target * (health - 1) * 50) / 100
            if decrease < base_target:
                self.target = base_target - decrease
            else:
                self.target = base_target / 2
        else:
            self.target = base_target
            
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

class BufferSystemWithUpdateFrequency(BufferSystem):
    def __init__(self, initial_buffer, initial_target, initial_stake, alpha, update_frequency):
        super().__init__(initial_buffer, initial_target, initial_stake, alpha)
        self.update_frequency = update_frequency  # How often to update target (in hours)
        self.hours_since_update = 0

    def should_update_target(self):
        self.hours_since_update += 1
        if self.hours_since_update >= self.update_frequency:
            self.hours_since_update = 0
            return True
        return False

def run_simulation_with_frequency(buffer_system):
    buffer_history = []
    target_history = []
    withdrawal_history = []
    deposit_history = []
    stake_history = []
    unstake_history = []
    total_amount_history = []
    target_update_counts = 0

    for day in range(SIMULATION_DAYS):
        for hour in range(DAILY_ACTIONS):
            buffer_history.append(buffer_system.get_buffer())
            target_history.append(buffer_system.get_target())

            # Random deposit or withdrawal
            action = np.random.choice(['deposit', 'withdraw'], p=[0.5, 0.5])
            amount = max(0, np.random.lognormal(mean=3, sigma=1))

            if action == 'deposit':
                buffer_system.deposit(amount)
                deposit_history.append(amount)
                withdrawal_history.append(0)
            else:
                withdrawn = buffer_system.withdraw(amount)
                withdrawal_history.append(withdrawn)
                deposit_history.append(0)

            buffer_system.record_buffer()
            total_amount_history.append(buffer_system.total_amount)

            # Check if it's time to update target
            if buffer_system.should_update_target():
                new_target = buffer_system.update_target()
                target_update_counts += 1

                if buffer_system.get_buffer() > new_target:
                    staked = buffer_system.stake()
                    stake_history.append(staked)
                    unstake_history.append(0)
                elif buffer_system.get_buffer() < new_target:
                    unstaked = buffer_system.top_up_buffer()
                    unstake_history.append(unstaked)
                    stake_history.append(0)
                else:
                    stake_history.append(0)
                    unstake_history.append(0)
            else:
                stake_history.append(0)
                unstake_history.append(0)

    total_enqueued = buffer_system.get_enqueued_requests()
    
    return {
        'buffer_history': buffer_history,
        'target_history': target_history,
        'withdrawal_history': withdrawal_history,
        'deposit_history': deposit_history,
        'stake_history': stake_history,
        'unstake_history': unstake_history,
        'total_amount_history': total_amount_history,
        'total_enqueued': total_enqueued,
        'target_updates': target_update_counts
    }

def analyze_update_frequencies(update_frequencies):
    results = {
        'staking_efficiencies': [],
        'withdrawal_efficiencies': [],
        'avg_buffers': [],
        'buffer_volatilities': [],
        'target_volatilities': [],
        'update_counts': [],
        'avg_total_stakes': []
    }
    
    for freq in update_frequencies:
        buffer_system = BufferSystemWithUpdateFrequency(
            INITIAL_BUFFER, INITIAL_TARGET, INITIAL_STAKE, ALPHA, freq
        )
        simulation_results = run_simulation_with_frequency(buffer_system)
        
        # Calculate metrics
        withdrawals = sum(simulation_results['withdrawal_history'])
        queued_withdrawals = simulation_results['total_enqueued']
        buffer_history = simulation_results['buffer_history']
        target_history = simulation_results['target_history']
        total_amount_history = simulation_results['total_amount_history']
        
        avg_buffer = np.mean(buffer_history)
        avg_total = np.mean(total_amount_history)
        buffer_volatility = np.std(buffer_history) / avg_buffer if avg_buffer > 0 else 0
        target_volatility = np.std(target_history) / np.mean(target_history)
        
        withdrawal_efficiency = (withdrawals / (withdrawals + queued_withdrawals) 
                               if (withdrawals + queued_withdrawals) > 0 else 0)
        staking_efficiency = (avg_total - avg_buffer) / avg_total if avg_total > 0 else 0
        
        # Store results
        results['staking_efficiencies'].append(staking_efficiency)
        results['withdrawal_efficiencies'].append(withdrawal_efficiency)
        results['avg_buffers'].append(avg_buffer)
        results['buffer_volatilities'].append(buffer_volatility)
        results['target_volatilities'].append(target_volatility)
        results['update_counts'].append(simulation_results['target_updates'])
        results['avg_total_stakes'].append(avg_total)
    
    return results

# Run analysis for different update frequencies
update_frequencies = [1, 2, 4, 6, 8, 12, 24, 48, 72, 168]  # hours between updates
results = analyze_update_frequencies(update_frequencies)

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Convert frequencies to more readable format for x-axis
freq_labels = [f'{freq}h' for freq in update_frequencies]

# Plot 1: Efficiencies
ax1.plot(freq_labels, results['staking_efficiencies'], 'b-o', label='Staking')
ax1.plot(freq_labels, results['withdrawal_efficiencies'], 'r-s', label='Withdrawal')
ax1.set_xlabel('Update Frequency')
ax1.set_ylabel('Efficiency')
ax1.set_title('System Efficiencies vs Update Frequency')
ax1.legend()
ax1.grid(True)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# Plot 2: Volatilities
ax2.plot(freq_labels, results['buffer_volatilities'], 'g-o', label='Buffer')
ax2.plot(freq_labels, results['target_volatilities'], 'm-s', label='Target')
ax2.set_xlabel('Update Frequency')
ax2.set_ylabel('Volatility (CV)')
ax2.set_title('System Volatility vs Update Frequency')
ax2.legend()
ax2.grid(True)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# Plot 3: Average Buffer and Total Stake
ax3.plot(freq_labels, results['avg_buffers'], 'c-o', label='Avg Buffer')
ax3.plot(freq_labels, results['avg_total_stakes'], 'y-s', label='Avg Total Stake')
ax3.set_xlabel('Update Frequency')
ax3.set_ylabel('Amount')
ax3.set_title('Average Amounts vs Update Frequency')
ax3.legend()
ax3.grid(True)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# Plot 4: Update Counts
ax4.bar(freq_labels, results['update_counts'], color='purple')
ax4.set_xlabel('Update Frequency')
ax4.set_ylabel('Number of Updates')
ax4.set_title('Total Target Updates vs Update Frequency')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
ax4.grid(True)

plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Results:")
print("Update Freq\tStaking Eff\tWithdrawal Eff\tBuffer Vol\tTarget Vol\tUpdates\tAvg Buffer\tAvg Total Stake")
for i, freq in enumerate(update_frequencies):
    print(f"{freq}h\t\t{results['staking_efficiencies'][i]:.4f}\t{results['withdrawal_efficiencies'][i]:.4f}\t" +
          f"{results['buffer_volatilities'][i]:.4f}\t{results['target_volatilities'][i]:.4f}\t" +
          f"{results['update_counts'][i]}\t{results['avg_buffers'][i]:.4f}\t{results['avg_total_stakes'][i]:.4f}")