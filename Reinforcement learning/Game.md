## Задание
Предстоит обучить агента взаимодействовать и побеждать в условиях не самой дружелюбной среды. Классически задача звучит следующим образом: агент с друзьями ранней весной играли в фрисби (летающая тарелка) на берегу прекрасного озера. В результате неудачного броска тарелка приземлилась на поверхности частично замерзшего озера. На озере есть участки с прочным льдом (белые квадраты) и участки с проталинами (черные квадраты). Задача агента — пройти от точки старта до точки падения тарелки (финиш), минуя проталины. \
Агенту доступны четыре действия: влево, вверх, вправо и вниз. Игра заканчивается, если исчерпано количество шагов (в нашем случае это 100), агент проваливается под воду (черная клетка) или все же доходит до тарелки фрисби (финиш). В последнем случае он получает награду +1, во всех остальных случаях — награда равна нулю. \
Выполните моделирование взаимодействия агента и среды. Для обучения используйте алгоритм Q-обучения. В качестве стратегии используйте -жадную стратегию с параметром epsilon = 0.1. Коэффициент дисконтирования gamma = 0.8. Установите параметр random_seed = 3.

## Выполнение
#### Импортируем библиотеки
```python
!pip install -q --upgrade gym==0.18.0
import gym
import numpy as np
import time
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
```
#### Зададим параметры гипермодели
```python
epsilon = 0.1 # Параметр эпсилон при использовании эпсилон жадной стратегии
gamma = 0.8 # Коэффциент дисконтирования гамма
random_seed = 100 #Random seed
time_delay = 1 # Задержка времени при отрисовке процесса игры после обучения (секунды)
lr_rate = 0.9 #Коэффициент скорости обучения альфа
```

#### Создадим среду размера 6х6. S обозначает точку старта. F -- лед безопасен, H -- проталина, G -- цель. Была также скопирована из библиотки GYM и слегка модифицировали функцию ```generate_random_map ```, для того, чтобы генерировать произвольные карты на основе ```random_seed ```. Создаём функцию проверки пути
```python
# Создание карты
def generate_random_map(size, p, sd):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False
    np.random.seed(sd)

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] not in '#H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]

```
#### Генерируем карту
```python
# Генерация карты
random_map = generate_random_map(size=6, p=0.8, sd=random_seed)  # Создаем свою карту
env = gym.make("FrozenLake-v0", desc=random_map, is_slippery=False)  # Инициализируем среду
print("Ваша карта")
env.render()  # Выводим карту на экран
```
#### Задание алгоритма Q - обучения
```python
# Задание алгоритма Q - Обучения
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0,env.action_space.n) #***
    else:
        action = np.random.choice(np.array(np.argwhere(Q[state, :] == np.amax(Q[state, :])).flatten().tolist()))
    return action
# argwhere() ищет индекс элемента по условию, flatten() - возвращает 1D ряд, tolist() - обратная array([])
# amax() - максимальное значение вдоль строки
def learn(state, state2, reward, action, done):
    if done:
        Q[state, action] = Q[state, action] + lr_rate * (reward - Q[state, action]) # терминальное состояние
    else:
        Q[state, action] = Q[state, action] + lr_rate * (reward + gamma * np.max(Q[state2,:]) - Q[state, action])
```
#### Узнать количество побед и номер игры, в которой агент впервые одержал пятую победу подряд
```python
from tqdm import tqdm
# Inititalization
np.random.seed(random_seed)
wins = []
total_games = 10000
max_steps = 100
Q = np.zeros((env.observation_space.n, env.action_space.n))
episode = 0
#Main cycle
for game in tqdm(range(total_games)):
    state = env.reset()
    t = 0
    while t < max_steps:
        if game > 5 and wins[game-5] == 1 and wins[game-4] == 1 and wins[game-3] == 1 and wins[game-2] == 1 and wins[game-1] == 1 and episode == 0:
          episode = game
        t += 1

        action = choose_action(state)

        state2, reward, done, info = env.step(action)

        if t == max_steps:
            done = True  

        learn(state, state2, reward, action, done)

        state = state2

        if done and reward == 1:
            wins.append(1)
            break
        if done:
            wins.append(0)
            break        


print("Количество побед в серии из 10 000 игр: ", np.sum(wins))
print("Пять побед подряд впервые было одержано в игре ", episode)
```
#### Проверка работоспособности с условием, что модель полностью обучена
```python
states=[] # Массив для сохранения состояний агента в течение игры
t = 0
state = env.reset()
wn = 0
while(t<100):
  env.render()
  time.sleep(time_delay)
  clear_output(wait=True)
  action = choose_action_one_game(state)  
  state2, reward, done, info = env.step(action)  
  states.append(state)
  state = state2
  t += 1
  if done and reward == 1:
    wn=1
  if done:
    break
if wn == 1:
  print("!!!Победа!!!")
```

#### Отобразим карту
```python
import matplotlib.pyplot as plt

def make_maze_pic(maze):
  maze_pic=[]
  for i in range(len(maze)):
    row = []
    for j in range(len(maze[i])):
      if maze[i][j] == 'S':
        row.append(0)
      if maze[i][j] == 'F':
        row.append(0)
      if maze[i][j] == 'H':
        row.append(1)
      if maze[i][j] == 'G':
        row.append(0)
    maze_pic.append(row)
  maze_pic = np.array(maze_pic)
  return maze_pic
  

#Make maze fit to plot
maze_pic = make_maze_pic(random_map)
nrows, ncols = maze_pic.shape

#Arrays of picture elements
rw = np.remainder(states,nrows)
cl = np.floor_divide(states,nrows)
if wn == 1:
  rw = np.append(rw, [nrows-1])
  cl = np.append(cl,[ncols-1])

#Picture plotting
fig, ax1 = plt.subplots(1, 1, tight_layout=True)
ax1.clear()
ax1.set_xticks(np.arange(0.5, nrows, step=1))
ax1.set_xticklabels([])
ax1.set_yticks(np.arange(0.5, ncols, step=1))
ax1.set_yticklabels([])
ax1.grid(True)
ax1.plot([0],[0], "gs", markersize=40)  # start is a big green square
ax1.text(0, 0.2,"Start", ha="center", va="center", color="white", fontsize=12) #Start text
ax1.plot([nrows-1],[ncols-1], "rs", markersize=40)  # exit is a big red square
ax1.text(nrows-1, ncols-1+0.2,"Finish", ha="center", va="center", color="white", fontsize=12) #Exit text
ax1.plot(rw,cl, ls = '-', color = 'blue') #Blue lines path
ax1.plot(rw,cl, "bo")  # Blue dots visited cells
ax1.imshow(maze_pic, cmap="binary")
```