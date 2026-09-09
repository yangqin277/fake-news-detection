# Error Analysis


## False Positive

真实新闻被预测为虚假新闻。


Example:

Text:

苹果发布最新人工智能手机


Prediction:

Fake News


Analysis:

The text is short and lacks detailed source information.
The model may associate concise announcement-style texts
with unreliable information.


---


## False Negative

虚假新闻被预测为真实新闻。


Example:

某专家表示某食品可以治疗疾病


Prediction:

Real News


Analysis:

The sentence contains formal expressions such as
"专家表示", which may make the text appear credible.