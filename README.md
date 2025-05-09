# Chess_engine

I'll do it as log since I think it would be more efficent and effective in tacking; But here are some basic things. 

# log 1 

So the first try was using only 4 Neural nets with the basic parameters like adam optimizer, i also have a RELu activation function. Testing has been so-so, it will definitely need more epochs since 6000 just won't do. I also forgot the save the the forth layer it seems but ill have a different struture and more layers so it will be fine. 

# log 2

So,  still have the adam optimizer and the RElu activation function, but now I have Gemma decay for exploration rate  so overtime it becomes "smarter" and tries to predict a few moves ahead. I also have a rewards function for exloration to ensure that it exactly explores. The reward function has different rewards stages and it gets rewards via move comperation. So, I would compare the my bots move against another and whether it was a point gain or loss determines it. It also gets a lot more points if does a good move and get punished a lot if it does a bad move. I though this was the best way to do it since calculating after the game would result in worse learning. 
