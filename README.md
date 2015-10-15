# rkproject
RK Dat9 project files

I originally looked at using data from the DC Government, but was unable to find a complete data set that could train and test a model, or multiple data sets that I thought I could combine to achieve that goal for the project.  I looked into the Kaggle-like competitions listed on the project readme, and thought that the Data Driven projects were more interesting. (There is also a third competition about well water quality in Tanzania that could also be really cool, and make for nice vizualizations <http://www.drivendata.org/competitions/7/page/23/>.  I think I need advice on what you think could best fit the objectives for the course project)

## Option A - Blood Donation Data
* What is the question you hope to answer?
The goal of this competition is to develop a model to determine which donors donated in March 2007 based on their historic donation data.

* What data are you planning to use to answer that question?
The provided series include volume donated, months since first donation, month since prior donation, number of donations, and then the target series.  **The dataset only includes 576 rows of data.**  Is that anywhere close to sufficient?
See file named "Driven Data Blood Donation.csv"

* What do you know about the data so far?
The data comes from a mobile blood donation center in Taipei, Taiwan.  There are only 4 dependent variable series, and then the 1 target series.

* Why did you choose this topic?
I've been a blood donor for 10 years, and if there was a sandwich punch card for donations, I'd probably have received at least 4 sandwiches by now.  In all seriousness, I think the variables are intuitive and easily understood.  When looking at some other data sets over the past two weeks I think that some of the variable definitions were opaque and modeling based on them may not necessarily be meaningful.


## Option B - World Bank Millenium Development Goals
* What is the question you hope to answer?
The goal of this Driven Data competition is to develop a model that will predict the value of Millenium Development goals 1 year and 5 years into the future.

* What data are you planning to use to answer that question?
The provided data set is time-series from 1972-2007, with up to 1200 variables per country (214 countries) although there is a lot of redudancy in these variables and many could be dropped.
See the file named 'MDGTrainingSet.csv' in the 'MDG data' folder. 

* What do you know about the data so far?
The data is incredibly varied - the data include economic data (labor rate participation, unemployment, tariffs), environmental (water pollution), public health (life expectancy), all from the UN.  There is a lot of missing data, and dealing with missing values in a meaningful way may be difficult (e.g. data for Croatia prior to 1991 either doesn't exist, or was for Yugoslavia).

* Why did you choose this topic?
I was an econ major in college and my courses focused on macro and international development.  There are a variety of topics within the dataset provided by the World Bank.  I think that I could probably whittle down the data to examine fewer Millenium Development goals.
