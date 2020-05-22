---
layout: post
title: Recap from MIT Covid-19 Challenge
category: 🧐
excerpt: Recap and potential future directions from the MIT Covid-19 Challenge.
image: https://community.intersystems.com/sites/default/files/inline/images/mit_covid_challenge.jpg
---

*Update - April 6th. It seems that [Facebook][zuck-facebook] is actually deploying a survey by CMU world-wide. This is pretty neat. It will be cool to see what kind of data emerges from this! And it's a good lesson in moving fast.*

This weekend I participated in the [MIT Covid-19 Challenge][challenge-link]. I thought it would be useful to write a blog post about some of the ideas seemed like they could be potentially fruitful moving forward.

## Notes

The particular area we focused on was at-home triaging. The team that I worked with most closely was interested in building a tool to detect Covid-19 from coughs ([here][github-link] is our Github repo). Another team that I worked with was interested in using speech recognition and text-to-speech to build a questionnaire assistant to call people in at-risk populations who might not have access to a smartphone, so that they can triage themself at home.

There are already a number of questionnaire-based apps out there, including:

- [The CDC's official bot][cdc-bot], which implements their own decision tree for identifying patients which should go into the hospital.
- A [screening tool][emory-checker] from my alma mater, Emory, which was built by the [Office of Critical Event Preparedness and Response][emory-critical-prep]. Dr. Isakov, the executive director, led the team that developed SORT, which played an important role during the Swine Flu (by [some ambitious estimates][sort-paper] preventing 100,000 unnecessary ED visits).
- Apple, for some reason, built [their own screening tool][apple-bot], which looks quite polished and is probably just a reimplementation of the CDC bot.

Additionally, there have been a number of projects attempting to classify coronavirus patients from their symptoms, including:

- A [project][cmu-checker] developed by CMU, led by [Ben Striner][ben-striner], who I've vaguely interacted with on Github and is a really solid machine learning engineer.
- Another [project][voca-ai] by Voca.ai and CMU, which may or may not be associated with the project above.
- A [hackathon project][cough-detect-hack] which attempted this same problem (and got beaten up on [/r/MachineLearning][cough-detect-hack-reddit] because of lack of evidence that machine learning could be used for this kind of problem).

For me personally, there were some useful technical learnings:

- Most core AWS services are HIPAA-compliant, including S3 and Elastic Beanstalk.
- Flask and Django are very well-supported within the AWS ecosystem (although maybe this makes me show my age). [Here][aws-flask] is the tutorial I followed.
- GCE has some really good AI APIs, including an awesome speech recognition API that has great Python bindings.

## Ideas and Future Directions

From thinking about this project a bit more, the most important ideas here seem to be as follows:

- The most important features that seem to be lacking in the existing questionnaire approaches are:
  - *Unique user identification*. These are largely anonymous, presumably because of fears of harvesting medical data which could pose a liability. My understanding of HIPAA is that our architecture should satisfy the legal requirements about data protection and anonymization. However, many Americans probably still don't trust anything remotely medical being stored somewhere centrally, for important historical reasons. I think this might be something which could be circumvented in other countries, where laws perspectives about data sharing and collection are different from in the United States.
  - *Follow-up*. I think from a personal perspective, it is very important to check in on people over the course of multiple days. For example, if someone wanted to subscribe to regular, automatic check-ins, this would provide a useful psychological service (anecdotally, my girlfriend's mom seriously suggested installing a video camera in her room - I'm sure she would feel more at ease knowing that her daughter would be checked on by the government).
  - *Reach*. Perhaps this is the most obvious, but many people just aren't aware that these bots exist, which really stunts their usefulness. The dream of something like this is large-scale symptom reporting, to be able to identify potential patients before their symptoms become too severe. Right now, the healthcare ecosystem is too fractured to answer even simple questions, like whether or not certain drugs are effective. Expanding the reach of these data collection efforts could potentially be invaluable for epidemiological work. As a potential bonus, because of the strict laws around medical data, it seems unlikely that this kind of program would easily lead to privacy overreaches.
- Potentially, coughing (or other kinds of at-home physical diagnosis) can be used as a lever to make people feel engaged with and cared for "remotely". I think there's a lot of psychological power in engaging beyond just a questionnaire, even if it just means coughing into your phone.

If you're interested in fleshing out some of these ideas more, feel free to shoot me an email.

[challenge-link]: https://covid19challenge.mit.edu/
[github-link]: https://github.com/codekansas/covid-cough-prediction
[aws-flask]: https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html
[emory-checker]: https://c19check.com/
[voca-ai]: https://voca.ai/corona-virus/
[cmu-checker]: https://futurism.com/neoscope/new-app-detects-covid19-voice
[cdc-bot]: https://covid19healthbot.cdc.gov/
[apple-bot]: https://www.apple.com/covid19/
[emory-critical-prep]: https://emergency.emory.edu/about/team/index.html
[sort-paper]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7115325/
[ben-striner]: https://github.com/bstriner
[cough-detect-hack]: https://devpost.com/software/detect-now
[cough-detect-hack-reddit]: https://www.reddit.com/r/MachineLearning/comments/frf02x/p_covid19_cough_detection_model_deep_learning/
[zuck-facebook]: https://www.facebook.com/zuck
