import numpy as np
import random



class Config:
    H = 6
    W = 10
    RADIUS = 1
    HP = 40
    ATK = 6
    MAX_SCORE = 5

    POS = [[[1,1], [2,3], [1,5]],
           [[8,1], [7,3], [8,5]]]
    MASS = [[1, 1, 1], 
            [1, 1, 1]]

    MU_GROUND = 0.1
    E = 0.7
    MU_BALL = 0

class Seal:
    def __init__(self, name: int, team: int, pos: np.ndarray, radius=Config.RADIUS, hp=Config.HP, atk=Config.ATK, mass=1):
        self.name = name
        self.team = team
        self.pos = pos
        self.mass = mass
        self.radius = radius
        self.hp = hp
        self.atk = atk
        self.velocity = np.array([0, 0])
        self.moveable = True

    def draw(self):
        pass


class Game:
    def __init__(self, H=6, W=10, num=3):
        self.H = H
        self.W = W
        self.num = num
        self.A_members = {}
        self.A_score = 0
        self.B_members = {}
        self.B_score = 0
        self.first = None
        self.reset()


    def reset(self):
        for i in range(num):
            self.A_members[i] = spawn(team=0, name=i)
            self.B_members[i] = spawn(team=1, name=i)
        self.A_score = 0
        self.B_score = 0
        self.first = 0 # 0: A first, 1: B first 

    def step(self, team, velocity, theta):
        
        # lst = [all check_collsion_seal, check_collision_wall]
        # while lst not empty:
        #     a = lst.pop(min time)
        #     update seal in a
        #     update lst
        # if seal dead:
        #     respawn

        state_A =  [[self.A_members[i].pos, self.A_members[i].hp] for i in range(num)]
        state_B =  [[self.B_members[i].pos, self.B_members[i].hp] for i in range(num)]
        observation = [state_A,  state_B, self.A_score, self.B_score]
        terminated = self.check_win()
        truncated, info = None, None
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


    def spawn(self, team:int, name:int):
        return Seal(team=team, name=name, pos=Config.POS[team][name])

    def check_collision_seal(self, s1:Seal, s2:Seal):
        return {'seals':[s1,s2], 'time':None, 'pos':None}

    def check_collision_wall(self, seal:Seal):
        return {'seals':[seal], 'time':None, 'pos':None}

    def collision_wall(self, seal:Seal):
        pass

    def collision_seal(self, s1:Seal, s2:Seal):
        pass

    def check_win(self):
        if self.A_score == Config.MAX_SCORE:
            return 0
        if self.B_score == Config.MAX_SCORE:
            return 1
        return None


if __name__ == '__main__':
    print('adsf')
