import pygame
import random
import math

pygame.font.init() # you have to call this at the start,
                   # if you want to use this module.
my_font = pygame.font.SysFont(pygame.font.get_default_font(), 30)

display = pygame.display.set_mode((500,500))

class Obstacle:
    def __init__(self, x) :
        self.x = x
        self.height = random.randint(175,350)

    def reset(self):
        self.x = 810
        self.height = random.randint(125,300)
    def update(self):
        self.x -= 5
        if self.x <= -60:
            self.reset()
        pygame.draw.rect(display,(0,255,0),pygame.Rect(self.x,self.height,50,500-self.height))
        pygame.draw.rect(display,(0,255,255),pygame.Rect(self.x,0,50,self.height-150))


    


class FlappyBirdAI:
    def __init__(self):
        pygame.display.set_caption('FlappyBird')
        self.clock = pygame.time.Clock()
        self.reset()
        
    
    def reset(self):
        self.x = 50
        self.y = 300
        self.gravity = 1
        self.acc = 0
        self.obstacle = [Obstacle(570),Obstacle(860),Obstacle(1070)]
        self.point = 0
        self.timer = 0

    def rbody(self):
        if self.acc < 30:    
            self.acc += self.gravity
        self.y += self.acc
        
    def jump(self):
        self.acc = -12

    def play(self , action):
        self.clock.tick(40)
        self.rbody()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        if action == [0,1] :
            self.lockout_timer = 0
            self.jump()

        done,reward = self.out_of_bound()
        done,reward = self.collision()
        reward += self.pointup()
        self._update()
        if self.timer == 0:
            reward += -abs(self.y-self.obstacle[self.point%3].height+62.5)/10 + 7.5
        else:
            self.timer -= 1
        return reward, done, self.point

    def out_of_bound(self):
        if self.y > 500 or self.y < -35:
            return True,-10
        else:
            return False,0

    def collision(self):
        for ob in self.obstacle:
            if self.x < ob.x + 50 and self.x + 35 > ob.x:
                if (self.y < ob.height-150) or self.y + 35 > ob.height:
                    return True,-30
                return False,5
        return False,0
    
    def _update(self):
        display.fill((0,0,0))
        for ob in self.obstacle:
            ob.update()
        pygame.draw.rect(display, (255,255,0) , pygame.Rect(self.x,self.y,35,35))
        text = my_font.render('Score : '+str(self.point), True, (255, 255, 255))
        display.blit(text, (0,0))
        pygame.display.flip()
    
    def pointup(self):
        for ob in  self.obstacle:
            if self.x > ob.x + 50 and self.x < ob.x + 60:
                self.point += 1
                self.timer = 5
                return 50
        return 0



# if __name__ == '__main__':
    
#     bird = FlappyBird()
#     while True:
#         bird.play()