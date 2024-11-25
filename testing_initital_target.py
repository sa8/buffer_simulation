import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# Constants
ALPHA = 2
BOND = 4
INITIAL_STAKE = 10000
INITIAL_BUFFER = 5
SIMULATION_DAYS = 10000
DAILY_ACTIONS = 24  # Simulate hourly actions
healthy_buffer = 0.8
linear_health_function = 0
normal_percentile = 0.5 # we want the buffer to keep within +/- 0.1 of the buffer
BASE_TARGET_PERCENTAGE = 20  # New constant: base target as 20% of total


class BufferSystem:
    def __init__(self, initial_buffer, initial_target, initial_stake, alpha):
        self.buffer = initial_buffer
        self.target = initial_target
        self.total_amount = initial_stake + initial_buffer
        self.buffer_history = []
        self.enqueued_requests = 0
        self.alpha = alpha

    # [Previous methods remain the same]
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
    
    def deposit(self, amount):
        self.total_amount += amount
        if self.buffer + amount > (1+ normal_percentile) * self.target:
            self.trigger_stake(amount)
        else:
            self.buffer += amount

    def trigger_stake(self, amount):
        if self.buffer + amount > (1+ normal_percentile) * self.target:
            staked_amount = self.buffer + amount - (1+ normal_percentile) * self.target
            self.buffer = (1+ normal_percentile) * self.target

    def withdraw(self, amount):
        if amount <= self.total_amount:
            health = self.buffer_health(self.buffer)
            
            if health > healthy_buffer:
                max_withdrawal = self.buffer
                if amount <= max_withdrawal:
                    self.buffer -= amount
                    self.total_amount -= amount
                    return amount
                else:
                    fulfilled = max_withdrawal
                    remaining = amount - fulfilled
                    self.enqueued_requests += remaining
                    self.buffer = 0
                    self.total_amount -= fulfilled
                    return fulfilled
            else:
                slipped_amount = amount * (0.2 + health)
                max_withdrawal = self.buffer
                
                if slipped_amount <= max_withdrawal:
                    self.buffer -= slipped_amount
                    self.total_amount -= slipped_amount
                    return slipped_amount
                else:
                    fulfilled = max_withdrawal
                    remaining = slipped_amount - fulfilled
                    self.enqueued_requests += remaining
                    self.buffer = 0
                    self.total_amount -= fulfilled
                    return fulfilled
        return 0

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
            buffer_history.append(buffer_system.get_buffer())
            target_history.append(buffer_system.get_target())

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

        new_target = buffer_system.update_target()

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

    total_enqueued = buffer_system.get_enqueued_requests()

    return buffer_history, target_history, withdrawal_history, deposit_history, stake_history, unstake_history, total_amount_history, total_enqueued

def analyze_target_range(target_range):
    staking_efficiencies = []
    withdrawal_efficiencies = []
    avg_buffers = []
    buffer_volatilities = []
    total_stakes = []

    for initial_target in target_range:
        buffer_system = BufferSystem(INITIAL_BUFFER, initial_target, INITIAL_STAKE, ALPHA)
        buffer_history, target_history, withdrawal_history, deposit_history, stake_history, unstake_history, total_amount_history, total_enqueued = run_simulation(buffer_system)

        # Calculate metrics
        withdrawals = sum(withdrawal_history)
        queued_withdrawals = total_enqueued
        avg_buffer = np.mean(buffer_history)
        avg_total = np.mean(total_amount_history)
        buffer_volatility = np.std(buffer_history) / avg_buffer  # Coefficient of variation

        withdrawal_efficiency = withdrawals / (withdrawals + queued_withdrawals) if (withdrawals + queued_withdrawals) > 0 else 0
        staking_efficiency = (avg_total - avg_buffer) / avg_total if avg_total > 0 else 0

        # Store results
        staking_efficiencies.append(staking_efficiency)
        withdrawal_efficiencies.append(withdrawal_efficiency)
        avg_buffers.append(avg_buffer)
        buffer_volatilities.append(buffer_volatility)
        total_stakes.append(avg_total)

    return staking_efficiencies, withdrawal_efficiencies, avg_buffers, buffer_volatilities, total_stakes

# Run analysis for different initial target values
target_range = np.linspace(10, 5000, 10)  # Test 10 different target values from 1 to 50
results = analyze_target_range(target_range)
staking_efficiencies, withdrawal_efficiencies, avg_buffers, buffer_volatilities, total_stakes = results

# Create subplots for different metrics
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot staking and withdrawal efficiencies
ax1.plot(target_range, staking_efficiencies, 'b-o', label='Staking Efficiency')
ax1.plot(target_range, withdrawal_efficiencies, 'r-s', label='Withdrawal Efficiency')
ax1.set_xlabel('Initial Target Value')
ax1.set_ylabel('Efficiency')
ax1.set_title('Efficiencies vs Initial Target')
ax1.legend()
ax1.grid(True)

# Plot average buffer sizes
ax2.plot(target_range, avg_buffers, 'g-o')
ax2.set_xlabel('Initial Target Value')
ax2.set_ylabel('Average Buffer Size')
ax2.set_title('Average Buffer Size vs Initial Target')
ax2.grid(True)

# Plot buffer volatility
ax3.plot(target_range, buffer_volatilities, 'm-o')
ax3.set_xlabel('Initial Target Value')
ax3.set_ylabel('Buffer Volatility (CV)')
ax3.set_title('Buffer Volatility vs Initial Target')
ax3.grid(True)

# Plot total stake
ax4.plot(target_range, total_stakes, 'c-o')
ax4.set_xlabel('Initial Target Value')
ax4.set_ylabel('Average Total Stake')
ax4.set_title('Average Total Stake vs Initial Target')
ax4.grid(True)

plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Results:")
print("Initial Target\tStaking Eff\tWithdrawal Eff\tAvg Buffer\tBuffer Volatility\tAvg Total Stake")
for i, target in enumerate(target_range):
    print(f"{target:.1f}\t\t{staking_efficiencies[i]:.4f}\t{withdrawal_efficiencies[i]:.4f}\t" +
          f"{avg_buffers[i]:.4f}\t{buffer_volatilities[i]:.4f}\t\t{total_stakes[i]:.4f}")