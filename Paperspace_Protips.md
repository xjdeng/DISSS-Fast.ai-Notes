# Paperspace Protips

In order to speed up GPU computations, I've paid for a [Paperspace](http://www.paperspace.com) instance to run my programs.  This is just a random collection of tactics and shortcuts I've discovered that made my life a lot easier in my [Fast.ai](http://course.fast.ai) journey which were not officially taught in the course.

## Run your Jupyter Notebook server in the background

This ensures that your computations will continue. If you don't do this and your terminal crashes, you might need to start all over again in running your Jupyter Notebook.

- Step 1: First, run jupyter notebook in the background using nohup and output it to a log: ```nohup jupyter notebook > jupyter.log &```
- Step 2: Next, read the log and copy the token: ```cat jupyter.log```
- Step 3: Then, open a browser to http://<your Paperspace instance's ip>:8888
- Step 4: You'll see a field asking for a token. Paste the token from step 2 there.

## Get Audio Reminders when your cell finish running

- Add the following line to your import statements at the beginning: ```from IPython.display import Audio```
- Now add the following line after another line that you anticipate will take a LONG time: ```Audio(url="sound.wav", autoplay=True)```
- You can play .wav or .mp3 files, that example assumes you have a sound named "sound.wav" in the same directory.
- Example, suppose ```model.fit(X,Y)``` will take a LONG time.  You'll need to invoke the following if you'd like to grab some coffee or watch some TV and immediately come back once execution is finished:
```
model.fit(X,Y)
Audio(url="sound.wav", autoplay=True)
```
For more information on the [Audio module, see here](https://musicinformationretrieval.com/ipython_audio.html).
