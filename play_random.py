import random
from snake_rl.env import SnakeGameAI, Direction

game = SnakeGameAI()

while True:
    n = random.randint(0, 2)
    action = [0, 0, 0]
    action[n] = 1

    reward, done, score = game.play_step(action)

    if done:
        print("GAME_OVER! Score: ", score)
        game.reset()