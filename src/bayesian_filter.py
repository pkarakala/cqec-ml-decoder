import numpy as np
from scipy.stats import norm
from src.operators import ERROR_SIGNATURES


# ─── Bayesian Filter (Wonham Filter / HMM) ────────────────────
# This is a principled baseline decoder that treats the error
# decoding problem as a hidden Markov model. At each timestep
# it maintains a belief distribution over the 4 possible error
# states and updates that belief using Bayes' rule given the
# new measurement.
#
# This is the "optimal" decoder under the model assumptions:
# - Errors follow a Markov chain (controlled by p_flip)
# - Measurements are Gaussian: r ~ N(meas_strength * syndrome, noise_std^2)
#
# It should beat the threshold decoder but may lose to the GRU
# if the GRU learns patterns in the data that violate these
# assumptions (e.g., time-dependent dynamics from Phase 2).
# ───────────────────────────────────────────────────────────────


class BayesianFilter:
    def __init__(
        self,
        p_flip: float = 0.01,
        meas_strength: float = 1.0,
        noise_std: float = 1.0,
        n_states: int = 4
    ):
        """
        Initialize the Bayesian filter.

        Parameters:
            p_flip: probability of any single qubit flipping per timestep
            meas_strength: expected strength of the syndrome signal
            noise_std: standard deviation of measurement noise
            n_states: number of error states (always 4 for the 3-qubit code)
        """
        self.p_flip = p_flip
        self.meas_strength = meas_strength
        self.noise_std = noise_std
        self.n_states = n_states

        # ── Build the transition matrix ────────────────────────
        # P[i, j] = probability of transitioning from state i to state j
        # This models the error dynamics: with probability p_flip, a
        # qubit flips; otherwise the state stays the same.
        self.transition_matrix = self._build_transition_matrix()

        # ── Build the observation model ────────────────────────
        # For each state, what (S1, S2) syndrome pair do we expect?
        self.syndromes = [ERROR_SIGNATURES[i] for i in range(n_states)]

    def _build_transition_matrix(self) -> np.ndarray:
        """
        Build the transition probability matrix for the error process.

        Transition rules:
        - No error (0) can transition to any single-qubit error (1,2,3)
          with probability p_flip/3 each, or stay at 0 with prob 1-p_flip
        - Any single-qubit error (1,2,3) can:
            * Flip the same qubit again → back to 0 (prob p_flip/3)
            * Flip a different qubit → transition to that error (prob p_flip/3 × 2)
            * No flip → stay in current error (prob 1-p_flip)

        This is a simplified model that assumes only single-qubit errors.
        """
        P = np.zeros((self.n_states, self.n_states))

        # From state 0 (no error)
        P[0, 0] = 1 - self.p_flip           # stay at no error
        P[0, 1] = self.p_flip / 3.0         # flip qubit 1
        P[0, 2] = self.p_flip / 3.0         # flip qubit 2
        P[0, 3] = self.p_flip / 3.0         # flip qubit 3

        # From state 1 (qubit 1 flipped)
        P[1, 0] = self.p_flip / 3.0         # flip qubit 1 again → cancel
        P[1, 1] = 1 - self.p_flip           # no flip → stay
        P[1, 2] = self.p_flip / 3.0         # flip qubit 2 → now error 2
        P[1, 3] = self.p_flip / 3.0         # flip qubit 3 → now error 3

        # From state 2 (qubit 2 flipped)
        P[2, 0] = self.p_flip / 3.0         # flip qubit 2 again → cancel
        P[2, 1] = self.p_flip / 3.0         # flip qubit 1 → now error 1
        P[2, 2] = 1 - self.p_flip           # no flip → stay
        P[2, 3] = self.p_flip / 3.0         # flip qubit 3 → now error 3

        # From state 3 (qubit 3 flipped)
        P[3, 0] = self.p_flip / 3.0         # flip qubit 3 again → cancel
        P[3, 1] = self.p_flip / 3.0         # flip qubit 1 → now error 1
        P[3, 2] = self.p_flip / 3.0         # flip qubit 2 → now error 2
        P[3, 3] = 1 - self.p_flip           # no flip → stay

        return P

    def observation_likelihood(self, r1: float, r2: float, state: int) -> float:
        """
        Compute P(measurement | state).

        Given that we're in error state `state`, what's the probability
        of observing measurement (r1, r2)?

        Model: r1 ~ N(meas_strength * s1, noise_std^2)
               r2 ~ N(meas_strength * s2, noise_std^2)

        where (s1, s2) is the syndrome signature for `state`.
        """
        s1, s2 = self.syndromes[state]

        # Expected measurement given this state
        expected_r1 = self.meas_strength * s1
        expected_r2 = self.meas_strength * s2

        # Gaussian likelihood
        prob_r1 = norm.pdf(r1, loc=expected_r1, scale=self.noise_std)
        prob_r2 = norm.pdf(r2, loc=expected_r2, scale=self.noise_std)

        # Independent measurements → multiply likelihoods
        return prob_r1 * prob_r2

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run the Bayesian filter on a batch of windowed sequences.

        X: array of shape (n_samples, window_size, 2)
           where X[:, :, 0] = r1 windows, X[:, :, 1] = r2 windows

        Returns: array of shape (n_samples,) with predicted error labels

        Algorithm (standard forward filtering):
        For each sample:
            1. Initialize belief: uniform over all states
            2. For each timestep in the window:
                a. Predict: belief_new = transition_matrix @ belief_old
                b. Update: belief_new *= likelihood(measurement | state)
                c. Normalize: belief_new /= sum(belief_new)
            3. After processing the full window, predict the most likely state
        """
        n_samples = X.shape[0]
        window_size = X.shape[1]
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # ── Initialize belief: uniform prior ────────────────
            belief = np.ones(self.n_states) / self.n_states

            # ── Run filter over the window ──────────────────────
            for t in range(window_size):
                r1_t = X[i, t, 0]
                r2_t = X[i, t, 1]

                # Predict step: propagate belief forward using transition model
                belief_predict = self.transition_matrix.T @ belief

                # Update step: incorporate measurement likelihood
                likelihood = np.array([
                    self.observation_likelihood(r1_t, r2_t, state)
                    for state in range(self.n_states)
                ])
                belief_update = belief_predict * likelihood

                # Normalize to get posterior belief
                belief_sum = belief_update.sum()
                if belief_sum > 0:
                    belief = belief_update / belief_sum
                else:
                    # Numerical underflow → reset to uniform
                    belief = np.ones(self.n_states) / self.n_states

            # ── After processing window, predict most likely state ──
            predictions[i] = np.argmax(belief)

        return predictions


# ─── Quick Test ───────────────────────────────────────────────
if __name__ == "__main__":
    from src.sim_measurement import generate_trajectory
    from src.datasets import create_windows

    print("\n=== Testing BayesianFilter ===\n")

    # Generate a test trajectory
    traj = generate_trajectory(T=200, p_flip=0.02, meas_strength=1.0, noise_std=1.0, seed=42)
    windowed = create_windows(traj, window_size=20)

    # Create filter and make predictions
    bf = BayesianFilter(p_flip=0.02, meas_strength=1.0, noise_std=1.0)
    preds = bf.predict(windowed["X"])

    # Compute accuracy
    accuracy = (preds == windowed["y"]).mean()

    print(f"Test trajectory: T=200, {len(preds)} windows")
    print(f"Bayesian filter accuracy: {accuracy:.4f}")
    print(f"Label distribution (true):      {np.bincount(windowed['y'], minlength=4)}")
    print(f"Label distribution (predicted): {np.bincount(preds, minlength=4)}")
    print("\n✓ BayesianFilter basic test passed\n")