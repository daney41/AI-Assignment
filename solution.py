import random
import sys
import time
from collections import deque

from constants import *
from environment import *
from state import State

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

"""


class Solver:

    def __init__(self, environment: Environment):
        self.trans_cache = {}
        self.environment = environment
        self.states = []

        self.V = {}
        self.policy = {}

        self.gamma = environment.gamma  # Discount factor
        self.epsilon = environment.epsilon  # Convergence threshold
        self.is_converged = False  # Convergence flag

        # generate the initial state and store in self.states
        self.initial_state = State(
            self.environment,
            self.environment.BEE_init_posit,
            self.environment.BEE_init_orient,
            self.environment.widget_init_posits,
            self.environment.widget_init_orients)
        self.states.append(self.initial_state)

        queue = deque([self.initial_state])
        visited = set()  # store the visited states
        visited.add(self.initial_state)

        # generate the state list of environment using bfs
        while queue:
            current_state = queue.popleft()

            # iterate all possible movement (FORWARD, REVERSE, SPIN_LEFT, SPIN_RIGHT)
            for action in BEE_ACTIONS:
                reward, new_state = self.environment.apply_dynamics(current_state, action)

                # if the new state is not visited, add to queue and visited list and save it to self.states.
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)
                    self.states.append(new_state)


    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5, 6]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.V = {state: 0 for state in self.states}  # Store initial value for each state
        self.policy = {state: BEE_ACTIONS[0] for state in self.states}  # Store initial policy for each state

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """

        return self.is_converged

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """

        new_values = {}
        new_policy = {}

        for state in self.V:
            # check if the state is solved or not
            if self.environment.is_solved(state):
                # if the state is solved assign it a reward value and no new movement
                new_values[state] = 23
                new_policy[state] = None
                continue
            best_q_value = -float('inf')
            best_action = None

            for action in BEE_ACTIONS:
                total_value = 0

                # Get the possible outcomes (transitions) for this state-action pair
                for prob, reward, next_state in self.get_transition_outcomes(state, action):
                    next_state_value = self.V.get(next_state, 0)
                    total_value += prob * (reward + (self.gamma * next_state_value))

                # Update the best action if the Q-value is higher
                if total_value > best_q_value:
                    best_q_value = total_value
                    best_action = action
            # Update the value and policy for this state
            new_values[state] = best_q_value
            new_policy[state] = best_action

        # Check convergence by comparing value changes
        differences = [abs(self.V[state] - new_values[state]) for state in self.states]
        max_diff = max(differences)

        if max_diff < self.epsilon:
            self.is_converged = True

        # Update values and policy
        self.V = new_values
        self.policy = new_policy

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(state) for the given state.
        :param state: the current state
        :return: V(state)
        """
        return self.V.get(state, 0)

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        self.V = {state: 0 for state in self.states}  # Store initial value for each state
        self.policy = {state: BEE_ACTIONS[0] for state in self.states}  # Store policy for each state

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.is_converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        # Perform a single policy evaluation
        values = self.policy_evaluation()
        # Perform a single policy improvement
        self.policy_improvement(values)

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[state]

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #
    def get_transition_outcomes(self, state, action):
        """
        Return a list of (probability, next_state, reward) tuples representing each possible outcome of performing the
        given action from the given state.
        :param state: a BeeBot State instance
        :param action: an element of BEE_ACTIONS
        :return: list of (probability, next_state, reward) tuples
        """
        # Caching to avoid redundant calculations
        if (state, action) in self.trans_cache:
            return self.trans_cache[(state, action)]
        outcomes = []

        # If the state is solved, return the terminal state
        if self.environment.is_solved(state):
            return [(1.0, 0.0, state)]

        if action in [SPIN_LEFT, SPIN_RIGHT]:
            # Handle spin actions separately since they have a deterministic outcome
            reward, next_state = self.environment.apply_dynamics(state, action)
            outcomes.append([1.0, reward, next_state])
        else:
            # Retrieve drift and double-move probabilities from the environment
            P_drift_CW = self.environment.drift_cw_probs[action]
            P_drift_CCW = self.environment.drift_ccw_probs[action]
            P_double_move = self.environment.double_move_probs[action]

            # No drift and no double move case
            prob_no_drift_no_double_move = (1 - P_drift_CW - P_drift_CCW) * (1 - P_double_move)
            reward_1, state_1 = self.apply_movements(state, [action])
            outcomes.append([prob_no_drift_no_double_move, reward_1, state_1])

            # No drift but double move case
            prob_no_drift_double_move = (1 - P_drift_CW - P_drift_CCW) * P_double_move
            reward_2, state_2 = self.apply_movements(state, [action, action])
            outcomes.append([prob_no_drift_double_move, reward_2, state_2])

            # Clockwise drift, no double move
            prob_drift_CW_no_double_move = P_drift_CW * (1 - P_double_move)
            reward_3, state_3 = self.apply_movements(state, [SPIN_RIGHT, action])
            outcomes.append([prob_drift_CW_no_double_move, reward_3, state_3])

            # Clockwise drift and double move
            prob_drift_CW_double_move = P_drift_CW * P_double_move
            reward_4, state_4 = self.apply_movements(state, [SPIN_RIGHT, action, action])
            outcomes.append([prob_drift_CW_double_move, reward_4, state_4])

            # Counter-clockwise drift, no double move
            prob_drift_CCW_no_double_move = P_drift_CCW * (1 - P_double_move)
            reward_5, state_5 = self.apply_movements(state, [SPIN_LEFT, action])
            outcomes.append([prob_drift_CCW_no_double_move, reward_5, state_5])

            # Counter-clockwise drift and double move
            prob_drift_CCW_double_move = P_drift_CCW * P_double_move
            reward_6, state_6 = self.apply_movements(state, [SPIN_LEFT, action, action])
            outcomes.append([prob_drift_CCW_double_move, reward_6, state_6])

        # Cache the results to avoid recalculating for the same state-action pair
        self.trans_cache[(state, action)] = outcomes
        return outcomes

    def apply_movements(self, state, movements: list):
        """
        根据给定的状态和动作，计算可能的最小 reward 和最终的状态。
        使用 apply_dynamic 处理漂移、双重移动情况，返回最小 reward 和最终状态。
        """
        # Initialize the state and reward
        new_state = state
        min_reward = 0
        for m in movements:
            # For each movement, apply dynamics and update state
            reward, new_state = self.environment.apply_dynamics(new_state, m)
            # Keep track of the minimum reward across all movements
            if reward < min_reward:
                min_reward = reward

        return min_reward, new_state

    def policy_evaluation(self):
        """
        Evaluate the current policy to convergence.
        """
        values_converged = False
        iteration = 0
        while not values_converged and iteration < 150:
            new_values = {}
            for state in self.states:
                # If the environment has been solved, set a predefined value
                if self.environment.is_solved(state):
                    new_values[state] = 23
                    continue
                # Calculate Q-value using the current policy's action
                action = self.policy[state]
                q_value = 0
                # Loop through all possible state transitions
                for prob, reward, next_state in self.get_transition_outcomes(state, action):
                    q_value += prob * (reward + self.gamma * self.V.get(next_state, 0))

                # Update state value with the computed Q-value
                new_values[state] = q_value

            # Check for convergence by comparing old and new values
            differences = [abs(self.V[state] - new_values[state]) for state in self.states]
            if max(differences) < self.epsilon:
                values_converged = True

            # Update values for the next iteration
            self.V = new_values
            iteration += 1

        return self.V

    def policy_improvement(self, values):
        """
        Improve the current policy based on 1-step lookahead.
        """
        policy_changed = False

        # Loop through all states to find the best action
        for state in self.states:
            if self.environment.is_solved(state):
                continue
            best_q = -float('inf')
            best_a = None
            # Loop through all actions and find the one with the highest reward
            for action in BEE_ACTIONS:
                total = 0
                # Calculate Q-value for the action
                for prob, reward, next_state in self.get_transition_outcomes(state, action):
                    total += prob * (reward + self.gamma * values.get(next_state, 0))

                # If this action has a higher Q-value, update the best action
                if total > best_q:
                    best_q = total
                    best_a = action

            # Update the policy with the best action
            if self.policy[state] != best_a:
                policy_changed = True
                self.policy[state] = best_a

        # If the policy didn't change, it has converged
        self.is_converged = not policy_changed
