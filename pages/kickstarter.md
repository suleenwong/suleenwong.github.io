<span style="font-weight: bold; color: black; font-size:180%; line-height: 32px;"> Predicting Kickstarter project success using machine learning  </span>  <br>
<span style="color:darkgrey;">June 2022 &nbsp;&ndash;&nbsp; Su Leen Wong</span>


**Overview:**   
Kickstarter, founded in 2009, is a crowdfunding platform where project creators can raise money from the public, circumventing traditional avenues of investment. It has an all-or-nothing funding model, whereby a project is only funded if it meets its goal amount; otherwise no money is given by backers to a project.

A huge variety of factors contribute to the success or failure of a project on Kickstarter. Some of these factors are able to be quantified or categorized, which allows machine learning models to predict whether a project will succeed or not.

The goal of this project is to predict if a Kickstart project will succeed or fail through using Exploratory Data Analysis and supervised Machine Learning models.

More generally, the aim is to help potential project creators as well as potential investors assess what their chances of success on Kickstarter will be.


Please follow the link to the github repo for a more in-depth explanation, data and resource citations.

**Authors:** 
Su Leen Wong and Erick Cantu

**Tools:**  
<span style="color:grey">Numpy:</span> matrix operations  
<span style="color:grey">Pandas:</span> data handling

*Language used: Python*

See the <span style="color:#steelblue;font-weight:bold;">[repo](https://github.com/vivienneprince/LoLNBClassifierDemo) </span>here. 

View the final presentation for this project here.

<br>  


Here's a preview of the code:   

```python
# ===================================================================
# Learn from train set
# ===================================================================

# Y = (y1,y2,..,yk) classes
# X = (x1,x2,...,xn) features

P_Y = []
distributions_XGivenY = []

for classtype in classes:
    class_data = train[train['blueWins'] == classtype]  # grab data for yj
    P_Y.append(len(class_data) / len(train))  # get P(yj)

    class_feature_distributions = []  # temp variable to store feature distributions for yj

    for feature in features:
        feature_data = class_data[feature]  # grab data for xi|yj

        class_feature_distributions.append([np.mean(feature_data), np.std(feature_data)])

    # each class gets their own array for feature distributions
    distributions_XGivenY.append(class_feature_distributions)

# convert feature|class distribution data to pd df
# this is in form: columns=classes, rows=features
distributions_XGivenY = pd.DataFrame(distributions_XGivenY, columns=[features]).transpose()

# ===================================================================
# model validation using test set
# ===================================================================

validation_array = []  # store if model was correct or incorrect
validation_score = 0  # mean of validation array


def likelihood(value, mu, sigma):
    # probability density function
    # image: https://wikimedia.org/api/rest_v1/media/math/render/svg/c9167a4f19898b676d4d1831530a8ff1246d33ab
    a = 2 * np.pi * sigma ** 2
    b = (value - mu) / sigma
    return 1 / np.sqrt(a) * np.exp(-0.5 * b ** 2)


def validate_result(guess, index):
    # compares guess with actual value from data
    if guess == test['blueWins'][index]: return 1
    else: return 0


for ind in test.index:

    # initiate
    P_yjGivenX = np.NINF  # P(yj|X)
    prediction = 0  # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]

    for classtype in classes:

        class_distribution_data = distributions_XGivenY[classtype]  # grab feature distribution data for yj

        P_xiGivenyj_array = []  # temp array to store log(P(xi|yj)) values

        for feature in features:
            feature_data = class_distribution_data[feature]  # grab xi distribution data
            feature_mu, feature_sigma = np.concatenate(feature_data.to_numpy()).ravel()

            feature_value = test[feature][ind]  # grab instance xi data

            # calculate log(P(xi|yj))
            P_xiGivenyj_array.append(np.log(likelihood(feature_value, feature_mu, feature_sigma)))

        # calculate log( P(yj)*PROD(i=1,n)(P(xi|yj) )
        temp_P = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)

        if temp_P > P_yjGivenX:
            # argmax(j=1,k) [ P(yj) * PROD(i=1,n)(P(xi|yj) ]
            P_yjGivenX = np.log(P_Y[classtype]) + np.sum(P_xiGivenyj_array)
            prediction = classtype

    validation_array.append(validate_result(prediction, ind))  # model validation

validation_score = np.mean(validation_array)  # prediction success rate. with np.rand seed = 0, success rate is ~75%
print(validation_score)
```
