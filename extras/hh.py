"""
Test script implementing the classic Hodgkin-Huxley model for comparison.

This provides a reference implementation of the HH equations for a single compartment
that we can use to validate our mesh-based compartmental modeling approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp


class HodgkinHuxleyModel:
    """
    Classic Hodgkin-Huxley model implementation.

    Parameters based on the original 1952 paper, adjusted for 279.45K (6.3°C).
    """

    def __init__(self):
        # Membrane capacitance (µF/cm²)
        self.C_m = 1.0

        # Maximum conductances (mS/cm²)
        self.g_Na_max = 120.0  # Sodium
        self.g_K_max = 36.0  # Potassium
        self.g_L = 0.3  # Leak

        # Reversal potentials (mV)
        self.E_Na = 50.0  # Sodium
        self.E_K = -77.0  # Potassium
        self.E_L = -54.387  # Leak

        # Resting potential (mV)
        self.V_rest = -65.0

    def alpha_m(self, V):
        """Sodium activation rate constant."""
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def beta_m(self, V):
        """Sodium activation rate constant."""
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        """Sodium inactivation rate constant."""
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        """Sodium inactivation rate constant."""
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        """Potassium activation rate constant."""
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def beta_n(self, V):
        """Potassium activation rate constant."""
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def steady_state_gating(self, V):
        """Calculate steady-state gating variables."""
        m_inf = self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))
        h_inf = self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))
        n_inf = self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))
        return m_inf, h_inf, n_inf

    def ionic_currents(self, V, m, h, n):
        """Calculate ionic currents."""
        # Sodium current
        I_Na = self.g_Na_max * (m**3) * h * (V - self.E_Na)

        # Potassium current
        I_K = self.g_K_max * (n**4) * (V - self.E_K)

        # Leak current
        I_L = self.g_L * (V - self.E_L)

        return I_Na, I_K, I_L

    def derivatives(self, t, y, I_ext):
        """
        Calculate derivatives for the HH system.

        Args:
            t: Time
            y: State vector [V, m, h, n]
            I_ext: External current (µA/cm²)

        Returns:
            Derivatives [dV/dt, dm/dt, dh/dt, dn/dt]
        """
        V, m, h, n = y

        # Rate constants
        alpha_m = self.alpha_m(V)
        beta_m = self.beta_m(V)
        alpha_h = self.alpha_h(V)
        beta_h = self.beta_h(V)
        alpha_n = self.alpha_n(V)
        beta_n = self.beta_n(V)

        # Ionic currents
        I_Na, I_K, I_L = self.ionic_currents(V, m, h, n)

        # Derivatives
        dV_dt = (I_ext - I_Na - I_K - I_L) / self.C_m
        dm_dt = alpha_m * (1 - m) - beta_m * m
        dh_dt = alpha_h * (1 - h) - beta_h * h
        dn_dt = alpha_n * (1 - n) - beta_n * n

        return np.array([dV_dt, dm_dt, dh_dt, dn_dt])

    def simulate(self, t_span, I_ext_func, initial_conditions=None, method="RK45"):
        """
        Simulate the Hodgkin-Huxley model.

        Args:
            t_span: Time span [t_start, t_end] or array of time points
            I_ext_func: Function that returns external current as function of time
            initial_conditions: Initial [V, m, h, n]. If None, uses steady state at rest.
            method: Integration method for solve_ivp

        Returns:
            solution: scipy integration solution object
        """
        # Set initial conditions
        if initial_conditions is None:
            V0 = self.V_rest
            m0, h0, n0 = self.steady_state_gating(V0)
            y0 = np.array([V0, m0, h0, n0])
        else:
            y0 = np.array(initial_conditions)

        # Create wrapper function for solve_ivp
        def dy_dt(t, y):
            return self.derivatives(t, y, I_ext_func(t))

        # Solve the system
        if isinstance(t_span, (list, tuple)) and len(t_span) == 2:
            # Time span given, let solver choose points
            solution = solve_ivp(dy_dt, t_span, y0, method=method, dense_output=True)
        else:
            # Specific time points given
            t_eval = np.array(t_span)
            solution = solve_ivp(dy_dt, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method=method)

        return solution


def current_step(t, start_time=10, end_time=40, amplitude=10):
    """Step current stimulus."""
    if start_time <= t <= end_time:
        return amplitude
    return 0.0


def current_pulse_train(t, pulse_start=10, pulse_duration=2, pulse_interval=20, num_pulses=3, amplitude=15):
    """Pulse train stimulus."""
    for i in range(num_pulses):
        pulse_time = pulse_start + i * pulse_interval
        if pulse_time <= t <= pulse_time + pulse_duration:
            return amplitude
    return 0.0


def test_action_potential():
    """Test basic action potential generation."""
    print("🧪 Testing Hodgkin-Huxley Action Potential Generation")
    print("=" * 60)

    # Create model
    hh = HodgkinHuxleyModel()

    # Simulation parameters
    t_span = [0, 50]  # ms
    t_eval = np.linspace(0, 50, 1000)

    # Current stimulus - step current
    def I_ext(t):
        return current_step(t, start_time=10, end_time=15, amplitude=10)

    # Run simulation
    print(f"Simulating HH model from {t_span[0]} to {t_span[-1]} ms...")
    solution = hh.simulate(t_eval, I_ext)

    if not solution.success:
        print(f"❌ Simulation failed: {solution.message}")
        return False

    # Extract results
    t = solution.t
    V = solution.y[0]
    m = solution.y[1]
    h = solution.y[2]
    n = solution.y[3]

    # Analyze results
    V_max = np.max(V)
    V_min = np.min(V)
    spike_threshold = -20  # mV
    spike_times = t[np.where(np.diff(np.sign(V - spike_threshold)) > 0)[0]]

    print(f"\n📊 Simulation Results:")
    print(f"  Voltage range: {V_min:.1f} to {V_max:.1f} mV")
    print(f"  Action potentials detected: {len(spike_times)}")
    if len(spike_times) > 0:
        print(f"  First spike time: {spike_times[0]:.1f} ms")
        print(f"  Peak voltage: {V_max:.1f} mV")

    # Validate basic properties
    has_spike = V_max > 0  # Should reach positive voltages
    returns_to_rest = abs(V[-1] - hh.V_rest) < 5  # Should return close to rest

    print(f"\n🎯 Validation:")
    print(f"  Generates action potential: {'✅ PASS' if has_spike else '❌ FAIL'}")
    print(f"  Returns to rest: {'✅ PASS' if returns_to_rest else '❌ FAIL'}")

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Voltage trace
    ax1.plot(t, V, "b-", linewidth=2)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Membrane Voltage")
    ax1.grid(True, alpha=0.3)

    # Gating variables
    ax2.plot(t, m, "r-", label="m (Na activation)", linewidth=2)
    ax2.plot(t, h, "g-", label="h (Na inactivation)", linewidth=2)
    ax2.plot(t, n, "b-", label="n (K activation)", linewidth=2)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Gating Variable")
    ax2.set_title("Gating Variables")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Current stimulus
    I_stim = [I_ext(time) for time in t]
    ax3.plot(t, I_stim, "k-", linewidth=2)
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Current (µA/cm²)")
    ax3.set_title("Stimulus Current")
    ax3.grid(True, alpha=0.3)

    # Ionic currents
    I_Na_vals = []
    I_K_vals = []
    I_L_vals = []
    for i in range(len(t)):
        I_Na, I_K, I_L = hh.ionic_currents(V[i], m[i], h[i], n[i])
        I_Na_vals.append(I_Na)
        I_K_vals.append(I_K)
        I_L_vals.append(I_L)

    ax4.plot(t, I_Na_vals, "r-", label="I_Na", linewidth=2)
    ax4.plot(t, I_K_vals, "b-", label="I_K", linewidth=2)
    ax4.plot(t, I_L_vals, "g-", label="I_L", linewidth=2)
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Current (µA/cm²)")
    ax4.set_title("Ionic Currents")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tests/data/hodgkin_huxley_test.png", dpi=150, bbox_inches="tight")
    print(f"\n📊 Plot saved to: tests/data/hodgkin_huxley_test.png")

    return has_spike and returns_to_rest


def test_iv_curve():
    """Test current-voltage relationship."""
    print("\n🧪 Testing I-V Relationship")
    print("=" * 40)

    hh = HodgkinHuxleyModel()

    # Test different current amplitudes
    current_amplitudes = np.linspace(0, 20, 11)
    spike_counts = []

    for I_amp in current_amplitudes:
        # Long step current
        def I_ext(t):
            return current_step(t, start_time=10, end_time=60, amplitude=I_amp)

        t_eval = np.linspace(0, 70, 1400)
        solution = hh.simulate(t_eval, I_ext)

        if solution.success:
            V = solution.y[0]
            # Count spikes
            spike_threshold = -20
            spikes = np.where(np.diff(np.sign(V - spike_threshold)) > 0)[0]
            spike_counts.append(len(spikes))
        else:
            spike_counts.append(0)

    # Find rheobase (minimum current for spiking)
    rheobase_idx = next((i for i, count in enumerate(spike_counts) if count > 0), None)
    rheobase = current_amplitudes[rheobase_idx] if rheobase_idx is not None else None

    print(f"  Current range tested: {current_amplitudes[0]:.1f} to {current_amplitudes[-1]:.1f} µA/cm²")
    print(f"  Rheobase (threshold current): {rheobase:.1f} µA/cm²" if rheobase else "  No threshold found")
    print(f"  Max spikes at highest current: {max(spike_counts)}")

    return rheobase is not None


def run_all_tests():
    """Run all Hodgkin-Huxley tests."""
    print("🧪 Running Hodgkin-Huxley Model Tests")
    print("=" * 50)

    # Ensure output directory exists
    import os

    os.makedirs("tests/data", exist_ok=True)

    # Test 1: Basic action potential
    ap_test = test_action_potential()

    # Test 2: I-V relationship
    iv_test = test_iv_curve()

    # Summary
    print("\n" + "=" * 50)
    print("🏆 TEST SUMMARY")
    print("=" * 50)
    print(f"Action potential generation: {'✅ PASS' if ap_test else '❌ FAIL'}")
    print(f"I-V relationship: {'✅ PASS' if iv_test else '❌ FAIL'}")

    if ap_test and iv_test:
        print("🎉 ALL HODGKIN-HUXLEY TESTS PASSED!")
        return True
    else:
        print("💥 SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    # Run tests
    run_all_tests()

    # Show plot if matplotlib backend supports it
    try:
        plt.show()
    except:
        print("Note: Plot display not available in this environment")
