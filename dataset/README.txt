

The YouTube Personality Dataset
-------------------------------

The YouTube personality dataset consists of  a collection of behavorial features, speech transcriptions,  and personality impression scores for a set of 404 YouTube vloggers that explicitly show themselves in front of the a webcam talking about a variety of topics including personal issues, politics, movies, books, etc.  There is no content-related restriction and the language used in the videos is natural and diverse. 

Detailed description
--------------------

Behavioral features 

Nonverbal cues were automatically extracted from the conversational excerpts of vlogs and aggregated at the video level.

- Nonverbal Features from Audio:

Audio cues were computed from the audio channel the toolbox developed by the Human Dynamics group at MIT Media Lab (http://groupmedia.media.mit.edu/data.php) and include both speaking activity and prosody cues. This is a total of 21 features.

- Nonverbal Features from Video:

The overall motion of the vlogger is an indicator of vloggers’ excitement and kinetic expressiveness.  We computed the overall visual activity of the vlogger with a modiﬁed version of motion energy images called ”Weighted Motion Energy Images” (wMEI)]. The normalized wMEI describes the motion throughout a video as a gray-scale image, where the intensity of each pixel indicates the visual activity in it. From the normalized wMEIs, we extract statistical features as descriptors of the vlogger’s body activity such as the entropy, median, and center of gravity (in horizontal and vertical dimensions). This is a total of 4 features.

Manual transcripts

We asked a professional company to manually transcribe the audio from vlogs. The transcription corresponds to the full video duration. In total, around 28h of video were annotated. 
We manually anonymized the data by substituting with a ‘XXXX’ any name entities referenced by the speaker that could be use to identify the video or the vlogger, as well as references to other social media accounts. The transcriptions are provided in raw text and contain a total of ~10K unique words and ~240K word tokens.

Personality Impression Scores

The personality impressions consist of Big-Five personality scores that were collected using Amazon Mechanical Turk (https://www.mturk.com/mturk/) and the Ten-Item Personality Inventory (http://homepage.psy.utexas.edu/HomePage/Faculty/Gosling/tipi%20site/tipi.htm). MTurk annotators watched one-minute slices of each vlog, and rated impressions using a personality questionnaire. The aggregated Big-Five scores are reliable with the following intra-class correlations (ICC(1,k), k=5): Extraversion (ICC = .76), Agreeableness (ICC = .64), Conscientiousness (ICC = .45), Emotional Stability (ICC = .42), Openness to Experience (ICC = .47), all significant with p < 10^{-3}.


Gender

The collection is mostly mostly balanced in gender, with 194 males (48%%) and 210 females (52%). The gender labels are also provided.


Acknowledgments

If you use this database, please cite the following publications:

- when using the NVB feautures:

@article{biel2013youtube,
  title={The youtube lens: Crowdsourced personality impressions and audiovisual analysis of vlogs},
  author={Biel, Joan-Isaac and Gatica-Perez, Daniel},
  journal={Multimedia, IEEE Transactions on},
  volume={15},
  number={1},
  pages={41--55},
  year={2013},
  publisher={IEEE}
}

- when using the text features:

@inproceedings{biel2013hi,
  title={Hi YouTube!: personality impressions and verbal content in social video},
  author={Biel, Joan-Isaac and Tsiminaki, Vagia and Dines, John and Gatica-Perez, Daniel},
  booktitle={Proceedings of the 15th ACM on International conference on multimodal interaction},
  pages={119--126},
  year={2013},
  organization={ACM}
}




Related literature 

J.-I. Biel and D. Gatica-Perez, “The YouTube Lens: Crowdsourced Personality Impressions
and Audiovisual Analysis of Vlogs" in IEEE Transactions on Multimedia , Vol. 15, No. 1,
pp. 41-55, Jan. 2013.

J.-I. Biel and D. Gatica-Perez. “VlogSense: Conversational Behavior and Social Attention
in YouTube" in ACM Transactions on Multimedia Computing, Communications, and Applications,
Special Issue on Social Media, Oct 2011.

 J.-I. Biel, V. Vtsminaki, V., J. Dines and D. Gatica-Perez “Hi YouTube! What verbal content
reveals in social video" in Proceedings International Conference on Multimodal Interaction
(ICMI) , Sydney, Dec. 2013.

J.-I. Biel, Teijeiro-Mosquera, L. and D. Gatica-Perez “FaceTube: Predicting Personality from
Facial Expressions of Emotion in Online Conversational Video" in Proceedings International
Conference on Multimodal Interaction (ICMI) , Santa Monica, Oct. 2012.

J.-I. Biel and D. Gatica-Perez “The Good, the Bad, and the Angry: Analyzing Crowdsourced
Impressions of Vloggers" in Proceedings of AAAI International Conference on Weblogs and
Social Media (ICWSM) , Dublin, June 2012

J-I. Biel, O. Aran, and D. Gatica-Perez, “You Are Known by How You Vlog: Personality
Impressions and Nonverbal Behavior in YouTube” In Proc. AAAI Int. Conf. . on Weblogs
and Social Media (ICWSM) , Barcelona, July 2011.

J.-I. Biel and D. Gatica-Perez, “Vlogcast Yourself: Nonverbal Behavior and Attention in
Social

