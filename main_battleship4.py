"""
Main file to run alternative dqn for lunar lander
"""
import matplotlib.pyplot as plt
from dqn_battleship_4 import Agent
from battleship_wrapper4 import BattleshipEnvironment
import numpy as np
from tqdm import tqdm


def plotLearning(x, scores_hist, epsilons, lines=None):
    """
    Plots the learning curve
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores_hist)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores_hist[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    # plt.savefig(filename)
    plt.show()


def renderRun():
    """
    Renders a run with the current model
    """
    env = BattleshipEnvironment()
    score = 0
    state = env.reset(0)
    while True:
        # env.render()
        action = agent.chooseAction(state)
        next_state, reward, done, info = env.step(action)
        agent.storeTransition(state, action, reward, new_observation, done)
        state = next_state
        score += reward
        if done:
            break


if __name__ == "__main__":
    env = BattleshipEnvironment()
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=100,
                  eps_end=0.01, input_dims=[100], learning_rate=0.003)
    scores, eps_history = [], []
    NUM_GAMES = 500

    loop = tqdm(total=NUM_GAMES, position=0, leave=False)
    for game in range(NUM_GAMES):
        score = 0
        done = False
        observation = env.reset(0)
        # while not done:
        for _ in range(100):
            action = agent.chooseAction(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.storeTransition(observation, action, reward, new_observation, done)
            agent.learn()
            observation = new_observation
            if done:
                break
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        loop.set_description('game:{}, avg_reward:{:.5f}'.format(game, avg_score))
        loop.update(1)

    x = [i + 1 for i in range(NUM_GAMES)]
    plotLearning(x, scores, eps_history)
