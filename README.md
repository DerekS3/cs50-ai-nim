# CS50 AI Nim

Nim game with AI using Q-learning and the epsilon-greedy algorithm. The AI refines its strategy through trial and error, balancing exploration and exploitation by occasionally choosing random moves (with probability epsilon) and otherwise selecting the optimal move. Over time, the AI develops a winning strategy for this two-player game.

## Contributions

`nim.py`:

`get_q_value`: Returns the Q-value for a given state and action, retrieving it from the dictionary self.q. If the state/action pair is not in the dictionary, it returns 0.

`update_q_value`: Updates the Q-value for a state/action pair using the Q-learning formula:
Q(s, a) <- old_q + alpha * (reward + future_rewards - old_q).

`best_future_reward`: Returns the highest possible reward for any available action in a given state, based on existing Q-values in self.q. Assumes a Q-value of 0 for actions not in self.q.

`choose_action`: Chooses an action for a given state. If epsilon is False, it selects the best action based on the highest Q-value. If epsilon is True, it follows the epsilon-greedy algorithm, selecting a random action with a probability of self.epsilon and otherwise the best action.

### Testing

A test script (`test_nim.py`) has been developed to verify the correct operation of all listed functions.

### Technologies Used

- `Unittest`

### Usage

- main: `python3 play.py`
- test: `python3 test_nim.py`