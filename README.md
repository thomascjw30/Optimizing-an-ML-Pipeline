# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**This dataset contains customer data, each record is a unique customer/account. We are tasked to predict (using the metrics provided in the data) whether the client will sign up for the bank."**

**The best performing model was the HyperDrive model with run ID: AutoML_c2a68fe6-5e22-4232-acc7-69b485c5dd76
This particular model had an accuracy of 91.6% accuracy rate. It's algorithmn was VotingEnsemble.
StackEnsemble came up second however, it took this 2nd best model to run ~20 more seconds. (VotingEnsemble finished training in 50 seconds)**

## Scikit-learn Pipeline
**This is the Parameter Sampler I chose:

ps = RandomParameterSampling(
    {
        "--C": choice(10, 20, 50, 100, 200, 500),
        '--max_iter': choice(1,10)
    }
)

I utilized RandomParameterSampling as it is the one of the better choices to start with because it is faster and supports early termination. 
As for the parameter choices, I kept it simple and only selected the 6 values: (10,20,50,100,200,500)

Although the parameters chosen and the accuracy was not as good as the AutoML run, it was still respectable. Best model had an accuracy 91.1%.
This particular model finished in 42 seconds with max iterations of 10 and regularization strength of 100. {"--C": 100, "--max_iter": 10}

This is the early stoppage policy I chose:
BanditPolicy(evaluation_interval=10, slack_factor=0.1)


**


## AutoML & Pipeline Comparison
** AutoML selected over 38 algorithmns to run on the data. It produced the best model in using the VotingEnsemble algorithmn with accuracy rate of 91.6%.
Although it produced the best model at 91.6% vs HyperDrive's model at 91.1% it did take quite awhile to train on all 38 algorithmns.

However, with that said, if one has the time and the computational resources, I think AutoML would be the more advantageous option as it 
- achieved a better accuracy
- has less steps involved to get the better results (hyperdrive requires more coding)
- AutoML utilizes more complex algorithmns without the need to code them separately 
- perfect for less-experienced ML/DS 
**


## Future work
**First and foremost, the data itself is imbalanced, so I think for future experiments, we should under-sample or over-sample the imbalanced classes or try to obtain more data in general.
Second, I think it would be worthwhile to do hyperparameter-tuning on the best algorithmn produced by AutoML as a starting point to see if one can improve upon the results.**

