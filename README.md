# Intro to Machine Learning

Summary: Today we will help you with basic tasks involved in machine learning in
Python.

💡 [Tap here](https://new.oprosso.net/p/4cb31ec3f47a4596bc758ea1861fb624) **to leave your feedback on the project**. It's anonymous and will help our team make your educational experience better. We recommend completing the survey immediately after the project.

## Contents

1. [Chapter I](#chapter-i) \
    1.1. [Foreword](#foreword)
2. [Chapter II](#chapter-ii) \
    2.1. [Instructions](#instructions)
3. [Chapter III](#chapter-iii) \
    3.1. [Specific instructions for the day](#specific-instructions-for-the-day)
4. [Chapter IV](#chapter-iv) \
    4.1. [Exercise 00: Binary classifier](#exercise-00-binary-classifier)
5. [Chapter V](#chapter-v) \
    5.1. [Exercise 01: Decision boundaries](#exercise-01-decision-boundaries)
6. [Chapter VI](#chapter-vi) \
    6.1. [Exercise 02: Multiclass](#exercise-02-multiclass)
7. [Chapter VII](#chapter-vii) \
    7.1. [Exercise 03: Overfitting](#exercise-03-overfitting)
8. [Chapter VIII](#chapter-viii) \
    8.1. [Exercise 04: Regression](#exercise-04-regression)
9. [Chapter IX](#chapter-ix) \
    9.1. [Exercise 05: Clustering](#exercise-05-clustering)
    
   
## Chapter I

### Foreword

- There are many terms connected to the field of data, such as artificial intelligence, machine learning, neural networks, and deep learning. Do you know the difference between them?
- Each item on the list is a subset of the previous one. The broadest term, artificial intelligence, includes any techniques that mimic human cognitive behavior. This can be done using machine learning algorithms or other techniques, such as writing a program with many "if-then-else" rules.
- Machine learning includes statistical algorithms that automate the process of creating rules. Machines can find correlations and use them for different tasks by "looking" at the data.
- Neural networks are a subset of machine learning algorithms. They were inspired by how the human brain works, but they are still far from it. Deep learning algorithms are a subset of neural nets. They usually have many layers. That is why they are called "deep".
- Remember that none of these algorithms are limitless. They can only help you if you have the data. Simple algorithms can be satisfied with small amounts of data, while deep learning algorithms require large amounts. At the same time, if your data is poor, the knowledge you get from the algorithms will be poor, too. No surprise, huh?

## Chapter II

### Instructions

* Use this page as your only reference. Do not pay attention to rumors or speculation about how to prepare your solution.
* Here and throughout, we use Python 3 as the only correct version of Python.
* The python files for python exercises (module01, module02, module03) must have the following block at the end: `if __name__ == ‘__main__’`.
* Pay attention to the permissions of your files and directories.
* To be assessed your solution must be in your GIT repository.
* Your solutions will be evaluated by your peers in the bootcamp.
* You should not leave any other files in your directory other than those explicitly specified in the exercise instructions. It is recommended that you modify your .gitignore to avoid any accidents.
* Your solution must be in your GIT repository for evaluation. Always push only to the develop branch! The master branch will be ignored. Work in the src directory.
* When you need to get precise output in your programs, it is forbidden to display a precalculated output instead of performing the exercise correctly.
* Have a question? Ask your neighbor on the right. If that fails, try your neighbor on the left.
* Your reference materials are your peers, the internet, and Google.
* Read the examples carefully. They may require information that is not specified elsewhere in the subject.
* May the Force be with you!

## Chapter III

### Specific instructions for the day

* Use the Jupyter notebook to work with your code.
* For each major subtask in the list of exercises, your ipynb file should have an H2 heading to help your peers easily navigate your code.
* No imports are allowed except those mentioned in the "Authorized Functions" section of the title block of each exercise.
* You can use any built-in function as long as it is not prohibited by the exercise.
* Save and load all the required data to the subfolder data/.
* `scikit-learn` (0.23.1) is the library that you need for all the machine learning tasks.

## Chapter IV

### Exercise 00: Binary classifier

- Turn-in directory: `ex00/`.
- Files to turn in: `00_binary_classifier_logreg.ipynb`.
- Allowed functions: no restrictions.

Today and the following day, you will work with machine learning. Today, we will cover the basics to help you avoid feeling overwhelmed by too much information. The next day, we will cover more sophisticated techniques. Stay tuned!

First, machine learning can be divided into supervised and unsupervised learning. In supervised learning, you want to make predictions. To do so, you provide the machine with examples, including a set of features and a target variable. For example, imagine that you want to predict whether a user would like or dislike a given movie. The features (X) could include genre, year of release, budget, cast, and director. The target variable (y) would be whether the user likes or dislikes the movie. This type of task is called a classification task. In classification problems, y is always categorical. If your target variable is continuous, such as a movie rating, it is called a regression problem.

Unsupervised learning does not require a target variable. It does not make forecasts. It usually helps you understand your data better. Clustering algorithms, for example, help you identify homogeneous groups of observations. You may find that you can divide your users into six groups. Then, you can create a special offer or recommendation for each group. Do not confuse this with classification. You are not trying to make predictions. You are simply examining your data.

In addition to clustering algorithms, unsupervised learning includes dimensionality reduction algorithms. These algorithms help reduce the number of features or observations. This may be helpful if you have a lot of features or observations but insufficient resources. They may also help you find latent features and improve the quality of your supervised learning algorithms.


But enough theory! Let's try training our first classifier. It will be a binary classifier, meaning the target variable has only two unique values. You will work with the dataset from the previous day. That's perfectly normal. In fact, that's how data science works. First, you perform descriptive analysis, then you analyze some basic statistics. Then, you perform exploratory analysis by creating different plots to gain a better understanding of your data. Only then do you move on to predictive analysis to make forecasts.

For this exercise, imagine the following common situation (which is, by the way, quite common). At some point, you realized that the more data you collect, the better.

You started saving more fields in the logs. Before, you only collected the time of the commit. But from that point on, you started collecting the date as well.

Now, you need to train a classifier that can predict whether a commit was made during the week or on the weekend. Then, you can use the classifier to label commits from the past, when you weren't collecting data.

Every supervised machine learning algorithm requires at least two arguments: X, the list of features, and y, the target column. But what are the features?

As you may recall, we only have logs such as 2020-04-17 05:19:02.744528. How can we use these to predict the day of the week? That is the creative part of the work. It's called feature engineering. You need to extract these features from the logs. What could they be? Remember, the observation in this task is a day. So, we need to extract something that characterizes days. What could that be? One possibility is the number of commits during the day. It could also be the number of commits before and after midday. It could also be the percentage of commits made before midday. You are only limited by your imagination!

For this exercise, we will use a simple approach with two features: the number of commits before and after midday.

* What you need to do is fully explained in the [notebook](https://drive.google.com/file/d/10X-v79bBGWFZ-Qo2W_59WR-yE6pqI-jD/view).

## Chapter V

### Exercise 01: Decision boundaries

- Turn-in directory: `ex01/`.
- Files to turn in: `01_binary_classifier_svm_tree.ipynb`.
- Allowed functions: no restrictions.

Okay, you've trained your first classifier! It probably seems like magic to you now. Let's take a look under the hood, try to improve your classifier's quality, and experiment with other algorithms.

In this exercise, you will learn how logistic regression works. You will also try two more machine learning algorithms: SVM and the decision tree. You will also visualize them.

* What you need to do is fully explained in the [notebook](https://drive.google.com/file/d/1DIaOKYrSUlFRv05dZi4B71t4YEKS3ISd/view).

## Chapter VI

### Exercise 02: Multiclass

- Turn-in directory: `ex02/`.
- Files to turn in: `02_multiclassi_one-hot.ipynb`.
- Allowed functions: no restrictions.

Okay, now you have a general understanding of how different algorithms make classifications. Now, it's time to get closer to a real-life scenario. In real life, your target column may contain more than two values. In such cases, you will need to train a multiclass classifier, not a binary classifier (do not confuse the second one; we're talking about situations in which your observations can belong to several classes at the same time).

Also, you may have categorical features as well as continuous ones. You will need to deal with them somehow. Algorithms understand numbers; they don't know what to do with text.

In this exercise, you will address both issues. You will also determine which features are most important to different algorithms. You will also try one more algorithm: random forest.

* What you need to do is described in full detail in the [notebook](https://drive.google.com/file/d/1R0nISlChrzwuca1OZzQIrOvlV1ZPFehq/view).

Isn’t it cool that we can predict the weekday of any commit with high accuracy, knowing who made it, when, for which lab, and how many tries were made?

## Chapter VII

### Exercise 03: Overfitting

- Turn-in directory: `ex03/`.
- Files to turn in: `03_split_crossval.ipynb`.
- Allowed functions: no restrictions.

We are confident that you achieved a high level of accuracy. However, those numbers are somewhat unfair. Why? You made predictions using the same data you used for training. Your models could have simply memorized all the observations. In theory, you could achieve 100% accuracy, but would your model be useful if it tried to predict data it hadn't seen? We highly doubt it. This is called overfitting.

One technique to prevent overfitting is to create a train/test split. You use one portion of the data for training and another to check the final quality of your model. Another technique is cross-validation. With this technique, we don't make a constant split; rather, we try different splits and see what quality of predictions we get.

* What you need to do is described in full detail in the [notebook](https://drive.google.com/file/d/190B3pI6DukYeWBHpfBuW1epTCzUCRA5L/view).

## Chapter VIII

### Exercise 04: Regression

- Turn-in directory: `ex04/`.
- Files to turn in: `04_regression.ipynb`.
- Allowed functions: no restrictions.

Now that you know a thing or two about classification tasks, let's work on a regression problem. In this exercise, let's work on a regression problem. Using data about their views of the newsfeed and the number of commits they made during the program, you will need to predict the average delta between the deadlines and the first commit for every user.

* What you need to do is described in full detail in the [notebook](https://drive.google.com/file/d/1Dog5RO7ViNlLSowPbUAd8LiTIUOh05CZ/view).

## Chapter IX

### Exercise 05: Clustering

- Turn-in directory: `ex05/`.
- Files to turn in: `05_clustering.ipynb`.
- Allowed functions: no restrictions.

It is time to try using unsupervised machine learning. This time, we will work with clustering algorithms. Our goal is to determine if we can categorize our users into homogeneous groups for future analysis. Perhaps we can add more triggers or engaging mechanics for them. However, the triggers and mechanics may differ depending on the users' existing behavior.

* What you need to do is described in full detail in the [notebook](https://drive.google.com/file/d/1SgbuTFWjJykYYV9HFRFd_iLrOZi0kUQy/view).
