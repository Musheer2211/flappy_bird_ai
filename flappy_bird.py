import pygame, sys
import random

pygame.font.init() # you have to call this at the start,
                   # if you want to use this module.
my_font = pygame.font.SysFont(pygame.font.get_default_font(), 30)
font_end = pygame.font.SysFont(pygame.font.get_default_font(), 40)
display = pygame.display.set_mode((500,500))

class SpriteSheet:
    def __init__(self, image):
        self.sheet = image

    def get_image(self, frame, width, height, scale, colour):
        image = pygame.Surface((width, height)).convert_alpha()
        image.blit(self.sheet, (0, 0), ((frame * width), 0, width, height))
        image = pygame.transform.scale(image, (width * scale, height * scale))
        image.set_colorkey(colour)  
        return image
    
spritesheet = SpriteSheet(pygame.image.load('./img/Bird1-1.png').convert_alpha())

bgimg = pygame.transform.scale(pygame.image.load('./img/Background2.png'),(500,500))

pipe = pygame.image.load('./img/pipe.png')

frames = [spritesheet.get_image(x,16,16,3,(0,0,0)) for x in range(4)]

run = True


class Obstacle:
    def __init__(self, x) :
        self.x = x
        self.height = random.randint(125,300)

    def reset(self):
        self.x = 720
        self.height = random.randint(125,300)
    def update(self):
        self.x -= 3.5
        if self.x <= -60:
            self.reset()
        
        display.blit(pygame.transform.scale(pipe,(50,self.height-125+50)),(self.x,-50))
        display.blit(pygame.transform.scale(pipe,(50,500-self.height+100)),(self.x,self.height))


    


class FlappyBird:
    def __init__(self):
        pygame.display.set_caption('FlappyBird')
        self.clock = pygame.time.Clock()
        self.reset()
        self.record = 0
    
    def reset(self):
        self.x = 50
        self.y = 250     
        self.gravity = 0.25
        self.acc =   0
        self.obstacle = [Obstacle(570),Obstacle(830),Obstacle(1040)]
        self.point = 0
        self.itr = 0

    def rbody(self):
        if self.acc < 10:    
            self.acc += self.gravity
        self.y += self.acc
        
    def jump(self):
        self.acc = -5.5

    def play(self):
        self.clock.tick(60)
        self.rbody()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.jump()
        self.pointup()
        self.collision()
        self._update()

    def collision(self):
        for ob in self.obstacle:
            if self.x < ob.x + 50 and self.x + 35 > ob.x:
                if (self.y < ob.height-125) or self.y + 35 > ob.height:
                    self.game_over()
                    self.reset()
                    self.play()
    
    def _update(self):
        display.blit(bgimg,(0,0))
        self.score()
        for ob in self.obstacle:
            ob.update()
        display.blit(frames[(self.itr//7)%4],(self.x-4,self.y-9))
        self.itr += 1
        pygame.display.update()
    
    def pointup(self):
        for ob in  self.obstacle:
            if self.x > ob.x + 50 and self.x < ob.x + 54:
                self.point += 1

    def score(self):
        text = my_font.render('Score : '+str(self.point), True, (255, 255, 255))
        display.blit(text, (0,0))       

    def game_over(self):
        if self.point > self.record:
            self.record = self.point
        while True:
            display.fill((0,0,0))
            text = font_end.render('High Score : '+str(self.record) , False , (255,255,255))
            display.blit(text,(20,120)) 
            text = font_end.render('Score : '+str(self.point) , False , (255,255,255))
            display.blit(text,(20,160)) 
            text = font_end.render('Press Space To Restart' , False , (255,255,255))
            display.blit(text,(20,220))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        return




if __name__ == '__main__':
    
    bird = FlappyBird()
    while run:
        bird.play()