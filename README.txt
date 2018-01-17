===================================
YouTube Personality Detection
===================================

Brief Intro
-------------------------------

This project mainly aims to build a model to predict the Big Five Personality scores of an individual according to a video clip. Specifically, there are three features sets (i.e., all features, text features, and non-text features), and three different model (i.e., Lasso, SVM, and Random Forest).

Files
-------------------------------

The files include "final_project.py", "README.txt", "sample_output.txt", and a folder named "dataset".

Technical Requirements
-------------------------------

Note that the "sklearn", "pandas", "numpy", "matplotlib", and "nltk" modules need to be pip installed in order to run the program successfully. Also, the code should be run in python 2.

Run Guideline
-------------------------------

The only file that need to run is "final_project.py" and there is no need to manually enter anything to run the code. However, by uncommenting the line of "plot_hist(rf)" (line 89, line 121, line 152), you could draw a hist graph of the first ten feature importances.

Dataset
-------------------------------

The dataset includes the non-text features as well as the corresponding transcripts from 404 video clips. See in dataset/README.txt for more details.

Model Application
-------------------------------

This project also introduce a model application using Donald Trump's victory speech after president election. The video transcript is in the "trump_speech" folder and the corresponding video can be accessed in [2].

Reference
-------------------------------

[1] Joan-Isaac Biel, Vagia Tsiminaki, John Dines, and Daniel Gatica-Perez. 2013. Hi youtube!: Personality impressions and verbal content in social video. In Proceedings of the 15th ACM on International Conference on Multimodal Interaction, ICMI ’13, pages 119–126, New York, NY, USA. ACM.

[2] Donald Trump's full victory speech. Retrieved December 5, 2017, from https://www.youtube.com/watch?v=pMgjwBgCZIw&t=26s
