"""

The game was given as an assignment in our DS course towards the end of the python section.
A reference to a website with the game code was provided, however this code was written in very bad python.
So the work was writing the code from scratch, putting what we learned into practice.

To write the game in 'good' pyhton, I did the following:
- Using modularity - arranging the code in functions and connecting between them
- Following DRY  (don't repeat yourself)
- Detailing the game constants and variables, so these could be used per need easily
- Simplifying the code
- Adding comments
- Many other changes (for instance, including the arrow and badger's rectangles within the lists, adding a dictionary
  for the arrows etc.)

I separated the program into sections:
- Game initiation and downloads
- Game constants/parameters
- Initiation variables per topic
- Game functions
- Running the program

The core functions of the game are:
- game_course
- set_surface
The other functions are sub function of these two, and appear above each of them respectively (i.e. sub functions
of game_course apper above it)
"""


import pygame
import math
import random
import copy


#Game initiation and downloads
pygame.init()
pygame.mixer.init()


## Downloading images
dude = pygame.image.load("dude.png")
grass = pygame.image.load("grass.png")
castle = pygame.image.load("castle.png")
arrowimg = pygame.image.load("bullet.png")
badgerimg = pygame.image.load("badguy.png")
healthbarimg = pygame.image.load("healthbar.png")
healthimg = pygame.image.load("health.png")
youwinimg = pygame.image.load("youwin.png")
gameoverimg = pygame.image.load("gameover.png")
## Downloading sounds
hit_sound = pygame.mixer.Sound("explode.wav")
enemy_sound = pygame.mixer.Sound("enemy.wav")
shoot_sound = pygame.mixer.Sound("shoot.wav")
pygame.mixer.music.load("moonlight.wav")


# Game constants/parameters

## Screen size and display parameters
SCREEN_WIDTH = 640 # Screen width
SCREEN_HEIGHT = 480 # Screen height
CLOCK_DIST_FROM_RIGHT = 50 # Clock distance from the right end
CLOCK_FROM_TOP = 5 # Clock distance from top
N_CASTLES = 4 # Setting the number of castles. The number can be changed and layout will adapt accordingly
BADGER_SPEED = 5 # Distance a badger moves each iteration
ARROW_SPEED = 5 # Distance an arrow moves each iteration
BUNNY_JUMP = 20 # The distance the dude (bunny) moves every time we hit relevant key
INITIAL_DUDEPOS = [200,100]# Initial position of bunny

## Time parameters
GAME_TIME_MINUTES = 1 # Number of minute for the game timer
END_GAME_PASUE_SEC = 5 # The number of seconds the 'game over' of 'you win' stays before quiting
BADGER_FIRST_TIMER_SEC = 5 # The time from game initiation till first badger appears
TIME_REDUCTION_LIMIT_SEC = 4 # Reduce time between badgers (must be lower than badger_timer, else no badgers at some point)
if TIME_REDUCTION_LIMIT_SEC >= BADGER_FIRST_TIMER_SEC:
    print("TIME_REDUCTION_LIMIT_SEC must be lower than BADGER_FIRST_TIMER_SEC.")
REDUCTION_INCREMENT_MLSEC = 50 # The time in hundredths of second
HEALTHBAR_FROM_TOP = 5 # Healthbar from top
HEALTHBAR_FROM_LEFT = 5 # Healthbar from left

## Health parameters
HEALTH_VALUE = 194
HEALTH_RADINT_MAX = 20 # Maximum reduction in random function every time a badger hits the castle
HEALTH_RADINT_MIN = 10 # Minimum reduction in random function every time a badger hits the castle
HEALTH_BORDER_THICKNESS = 3

## Keyboard dictionary
KEYDOWN_DICT = {pygame.K_UP: (1, -1), pygame.K_DOWN: (1, 1), pygame.K_LEFT: (0, -1), pygame.K_RIGHT: (0, 1)}

# Initiation variables per topic

## General variables
my_screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
dudepos = [200,100] # Position of bunny
health_value = HEALTH_VALUE
## Castles variables
healthbar_total = healthbarimg.get_height() + HEALTHBAR_FROM_TOP # The height of the healthbar bottom
reduced_screen_height = SCREEN_HEIGHT - healthbar_total # Heaight reduced by the bar and scace above it
distance_between_castles = reduced_screen_height  / N_CASTLES
## The line below - the distance betwen the screen top and the upper castle + the healthbar and a scace of 5 from the top screen
upper_coord_top_castle = (distance_between_castles - castle.get_height())/2 +  healthbar_total
middle_of_upper_castle = distance_between_castles/2 + healthbar_total
## Arrow parameters
arrow_length = arrowimg.get_width() # Arrow length - calculated to fallow full visibility of the arrow when it leaves the screan from top and left (see tail when head is out) 
arrows_list =[]
## Badger variables
badger_list = []
middle_of_upper_castle_with_half_badger = middle_of_upper_castle - badgerimg.get_height()/2 # To be in the middle, compensating half badger
badger_heights = [middle_of_upper_castle_with_half_badger + i*distance_between_castles for i in range(N_CASTLES)] # Badgers appear in castle height
time_reduction = 50 # At the beginning no time reduction between badgers
time_reduction_limit = TIME_REDUCTION_LIMIT_SEC * 100
badger_first_timer = BADGER_FIRST_TIMER_SEC * 100
badger_timer = copy.copy(badger_first_timer)

## Music and sounds
hit_sound.set_volume(0.15)
enemy_sound.set_volume(0.15)
shoot_sound.set_volume(0.05)
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.set_volume(0.15)


# Game functions

def dude_angle_position():
    """
    The function takes the position of the mouse against the image 'dude'(bunny), and returns:
    angle - The rotation angle
    duderot - New surface, shaped by the 'dude', but rotated
    dudepos1 - Position of the rotated 'dude'
    """
    
    mouse_position = pygame.mouse.get_pos() # Mouse position, for getting the angle between the dude and the mouse
    angle = math.atan2(mouse_position[1]-(dudepos[1]+32),mouse_position[0]-(dudepos[0]+26)) # The angle
    duderot = pygame.transform.rotate(dude, 360-angle*57.29) # Gets the dude image rotate by the angle 
    dudepos1 = (dudepos[0]-duderot.get_rect().width/2, dudepos[1]-duderot.get_rect().height/2) # Position of rotated dude
    return angle, duderot, dudepos1 


def badger_hit_castle(position):
    """
    The fuction operates the following operations when a badger hits a castle:
    - pops the hitting badger
    - operates a hit sound
    - calculates a random health reduction
    The funciton returns the updated health_value
    """

    badger_list.pop(badger_list.index(position)) # Takes the badger out of the list
    hit_sound.play() # Hit sound
    global health_value
    health_value -= random.randint(HEALTH_RADINT_MIN,HEALTH_RADINT_MAX) # Reducing the health level
    return health_value


def creating_badger():
    """
    The function creates new badger once the badger timer reaches 0.
    It creates a nested list with the badger's position within badger_list, blits the badger, and reset the timer.
    """

    global badger_timer
    global time_reduction
    badger_rect =pygame.Rect(badgerimg.get_rect())
    badger_list.append([SCREEN_WIDTH, random.choice(badger_heights), badger_rect]) # Adds the new badger to the list with its position (random height by castle)
                                                                            # Also I decided to include the rect inside the list
    my_screen.blit(badgerimg, (badger_list[-1][0], badger_list[-1][1])) # Blit the badger according to its location
    time_reduction += REDUCTION_INCREMENT_MLSEC
    if time_reduction >= time_reduction_limit: 
        time_reduction = time_reduction_limit # This limit is to allow a minimum of timer (badger_timer - time_reduction)
    badger_timer = badger_first_timer - time_reduction # The actual timer


def set_badger_surface():
    """
    Setting the surface for the badgers. For every badger, the function does the following:
    - moves the badger's position to the left
    - blit new badger
    - update badger's rectangle
    - calls the function badger_hit_castle
    - pops badger and arrow if they coolide
    """

    for badger_pos in badger_list:
        badger_pos[0] -= BADGER_SPEED # Taking the badger to the left
        my_screen.blit(badgerimg, (badger_pos[0], badger_pos[1])) # Show the badegr in the new position
        badger_pos[2].top = badger_pos[0]
        badger_pos[2].left = badger_pos[1]
        if badger_pos[0] < castle.get_width()-20: # When the badger reaches the right side of the castle
            badger_hit_castle(badger_pos)
        # The following lines handle the collision of an arrow with a badger
        for bullet in arrows_list:
            if badger_pos[2].colliderect(bullet[4]): # if the two rectangles collide
                badger_list.pop(badger_list.index(badger_pos)) # Pop badger
                arrows_list.pop(arrows_list.index(bullet)) # Pop arrow
                enemy_sound.play()


def set_surface_arrow():
    """
    Setting the surface for the arrow. For every arrow, the function do the following:
    - rotates it according to its angle
    - blits the arrow
    - updates its rectangle
    - pops the arrow when it reaches any screen end
    """

    for bullet in arrows_list:
        bullet[1] += math.cos(bullet[0])*ARROW_SPEED  # updated position x axis according to the original angle and speed (multiplyer)
        bullet[2] += math.sin(bullet[0])*ARROW_SPEED  # updated position y axis according to the original angle and speed (multiplyer)
        my_screen.blit(bullet[3], (bullet[1], bullet[2])) # PLace the arrow in a new positon pygame.Rect(bullet[1], bullet[2], 0, 0))
        bullet[4].top = bullet[1] # Updating the position of the arrow rectangle
        bullet[4].left = bullet[2] # Updating the position of the arrow rectangle
        # The condition below pops an arrow if it get out of the screan
        if any ([bullet[1]< -arrow_length, bullet[1]>SCREEN_WIDTH, bullet[2]< -arrow_length, bullet[2]>SCREEN_HEIGHT]):
            arrows_list.pop(arrows_list.index(bullet))


def clock_update(game_time_min = GAME_TIME_MINUTES, screen_width = SCREEN_WIDTH, clock_dist_from_right = CLOCK_DIST_FROM_RIGHT): #GAME_TIME_MINUTES
    """
    This function updates the display timer and the 'real timer' according to which the game ends.
    The function returns the real_time value
    """
    myfont = pygame.font.Font(None, 30) # Determines the font
    ticks = pygame.time.get_ticks() # Variable that counts ticks from initiation
    game_time_mlsec = game_time_min * 60000  # Game time in ml seconds
    clock_left = screen_width - clock_dist_from_right
    textsurface = myfont.render(str((game_time_mlsec - ticks)//60000)+":"+str((game_time_mlsec - ticks)//1000%60).zfill(2), True, (0, 0, 0)) # How to display the clock
    my_screen.blit(textsurface,(clock_left,CLOCK_FROM_TOP)) # Blit the clock
    real_timer = game_time_mlsec - ticks # This is the real clock for knowing when the time ends
    return (real_timer)

def bar_update():
    """
    This function updates the healthbar. It blits the healthbar image (the red bar) and blits the green stacks according
    to health value
    """

    my_screen.blit(healthbarimg, (HEALTHBAR_FROM_LEFT,HEALTHBAR_FROM_TOP))
    for health1 in range(health_value): # The loop reducing the health bar  ..badger_hit_castle(dude_angle_position()[3])
        my_screen.blit(healthimg, (health1+HEALTHBAR_FROM_LEFT+HEALTH_BORDER_THICKNESS,HEALTHBAR_FROM_TOP+HEALTH_BORDER_THICKNESS))

    

def set_surface():
    """
    The function resets the screen in every iteration of the game (see end of game_course() function).
    It resets the background, the castle, the bunny and the arrows.
    The function returns nothing
    """

    my_screen.fill(0)  # Starts the screen from scratch (black)
    # Next 3 lines place the grass background
    for x in range(SCREEN_WIDTH//grass.get_width() + 1): # WIDTH//grass.get_width() + 1 is the number of tiles to cover the width (round up)
        for y in range(SCREEN_HEIGHT//grass.get_height() + 1): # Height//grass.get_width() + 1 is the number of tiles to cover the height (round up)
            my_screen.blit(grass, (x * grass.get_width(), y* grass.get_height())) # Location of each tile
    # Now placing the castles
    for i in range(N_CASTLES):
        my_screen.blit(castle, (0, upper_coord_top_castle + i*distance_between_castles))
    # Placing and updating the bunny
    my_screen.blit(dude_angle_position()[1], dude_angle_position()[2]) # get position from dude_angle_positio function
    # Updating the arrows
    set_surface_arrow() # Calls the function that updates the surface with the arrow motion
    # Updeting badgers
    global badger_timer
    if badger_timer==0: # This is for creating new badgers when timer reaches 0
        creating_badger()
    set_badger_surface() # Calls the function that updates the surface with the badgers motion and arrow-badger collision 
    clock_update()
    bar_update()
    pygame.display.flip()


def game_end(is_win, end_game_pause_sec = END_GAME_PASUE_SEC):
    """
    The function defines the actions for a case where the healthbar reaches 0. In this case the player looses
    """
    end_game_pause_mlsec = end_game_pause_sec * 1000
    if is_win:
        my_screen.blit(youwinimg, (0, 0))
    else:
        my_screen.blit(gameoverimg, (0,0))
    pygame.display.update()
    pygame.time.delay(end_game_pause_mlsec) # Pausing with the game over message
    pygame.quit()




def game_course():
    """
    The function is the main function of the game, collects events and handles them
    The function returns nothing
    """

    is_quit = False
    global badger_timer
    while is_quit == False and clock_update(GAME_TIME_MINUTES) > 0: # Game continues a s long as the is_quit variable is True abd clock (timer) positive
        global badger_timer
        badger_timer -= 1
        for event in pygame.event.get():
            # The arrow keys move the bunny
            if event.type == pygame.KEYDOWN:
                dudepos[KEYDOWN_DICT[event.key][0]] += BUNNY_JUMP * KEYDOWN_DICT[event.key][1]
                my_screen.blit(dude, dudepos) # blit the bunny with the new location
                pygame.display.update()
            # The condition below create an arrow with evevry mouse click
            elif event.type==pygame.MOUSEBUTTONDOWN:
                rotated_arrow = pygame.transform.rotate(arrowimg, 360-dude_angle_position()[0]*57.29) # A rotated arrow by the dude's angle
                rotated_arrow_rect =pygame.Rect(rotated_arrow.get_rect()) # creating rectangle with the arrow shape and angle (rotated)
                arrows_list.append([dude_angle_position()[0], dude_angle_position()[2][0]+32, dude_angle_position()[2][1]+32, rotated_arrow, rotated_arrow_rect])
                my_screen.blit(rotated_arrow, (arrows_list[0][1], arrows_list[0][2])) # Blitting the arrow
                shoot_sound.play() # Shoot sound
                pygame.display.update()
            # The condition below quits the program
            elif event.type == pygame.QUIT:
                pygame.quit()
                is_quit = True
                # Originally there was a return here since I got an error
                # that I need to let the program end in the same way a normal python program ends
        set_surface() # Calling set_surface function to update screen dynamics
        if clock_update(GAME_TIME_MINUTES) == 0 or health_value <= 0: # If timer reaches 0 or health_value reaches 0 loop ends
            break
    if health_value <= 0:
        is_win = False
    else:
        is_win = True
    game_end(is_win, END_GAME_PASUE_SEC)# Calls the game_end

# Running the program
set_surface() # Calls the set_surface function, to update the screen The first time
game_course() # Calls the game_course function