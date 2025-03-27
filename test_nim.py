import unittest
from nim import *


class TestGetQValue(unittest.TestCase):
    def setUp(self):
        self.ai = NimAI()
        self.ai.state = tuple([1, 3, 5, 7])
        self.ai.action = tuple([0, 1])
        self.ai.q = {(self.ai.state, self.ai.action): 3}
        
    def test_get_q_value(self):
        actual_q_value = self.ai.get_q_value(self.ai.state, self.ai.action)
        expected_q_value = 3
        self.assertEqual(actual_q_value, expected_q_value)


class TestUpdateQValue(unittest.TestCase):
    def setUp(self):
        self.ai = NimAI()
        self.ai.state = tuple([1, 3, 5, 7])
        self.ai.action = tuple([0, 1])
        self.ai.q = {(self.ai.state, self.ai.action): 3}
        
    def test_update_q_value(self):
        reward = 1
        future_rewards = 3
        self.ai.update_q_value(
            self.ai.state, self.ai.action, 
            self.ai.q[(self.ai.state, self.ai.action)], 
            reward, future_rewards
        )
        updated_q_value = self.ai.q[(self.ai.state, self.ai.action)]
        expected_q_value = 3.5
        self.assertEqual(updated_q_value, expected_q_value)


class TestBestFutureReward(unittest.TestCase):
    def setUp(self):
        self.ai = NimAI()
        self.ai.state = tuple([0, 1, 1, 0])
        self.ai.q.update({(self.ai.state, (1, 1)): 3}) 
        self.ai.q.update({(self.ai.state, (2, 1)): 2})
        
    def test_best_future_reward(self):
        actual_result = self.ai.best_future_reward(self.ai.state)
        expected_result = 3
        self.assertEqual(actual_result, expected_result)


class TestChooseAction(unittest.TestCase):
    def setUp(self):
        self.ai = NimAI()
        self.ai.state = tuple([0, 1, 1, 2])
        self.ai.q.update({(self.ai.state, (1, 1)): 3}) 
        self.ai.q.update({(self.ai.state, (2, 1)): 2})
        self.ai.q.update({(self.ai.state, (3, 1)): 0}) 
        self.ai.q.update({(self.ai.state, (3, 2)): 5})
        
    def test_choose_action_epsilon_0(self):
        self.ai.epsilon = 0
        actual_result = self.ai.choose_action(self.ai.state, self.ai.epsilon)
        expected_result = (3, 2)
        self.assertEqual(actual_result, expected_result)

    def test_choose_action_epsilon_1(self):
        self.ai.epsilon = 1
        available_actions = list(Nim.available_actions(self.ai.state))
        actions = [
            self.ai.choose_action(self.ai.state, self.ai.epsilon) for _ in range(20)
        ]
        self.assertTrue(all(action in available_actions for action in actions))


if __name__ == '__main__':
    unittest.main()