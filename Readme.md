### Large Language Model Behaviours through Seeded Warming of the model and KV cache generation. 

The idea here is how we precive thinking in human standards, we are not reactive thinkers, but a proactive one. 
The proposal is how we can handle the proactive elements within the LLMs instead of them being cold when asked question. 
We are gonna intailised a seed like, we generate 256 random numbers assigned to between 0 and length of the unique token characters of the trained model. 
From this we create random seed initializer. We pass this as input and create a random cache of KV cache and then ask to solve different problems and see the performance of the LLM between warm and cold. 

The thinking aspect here, we take the KV cache from the first generation as distrubution, we choose the tokens form this cache and pass it to the second pass. This creates an internal noise that emulates the human thinking noise.
Now we pass the question and check it performance against the cold start. 

We also have determinstic seed, in terms of domain specific tasks like coding agents, we seed the random noise with elements of the specific domain to warm the LLM to solve the task. 

We are using the local open weights models from the 2-8b models to see if the internal que would reach performance equal to bigger models or beat their orignal performance by any standards.

#### Different Types of seeding. 

We can intialised seeding, where the chain of kv cache generates and continue the noise from the seed it generated from the first. 

We intialised new seed at every pass like either from the generation output and move to next set of new seed increasing random nosie. 