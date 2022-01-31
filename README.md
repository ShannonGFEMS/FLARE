# FLARE
Forced Labor Automated Risk Estimator, a supply chain classifier

This classifer depends on data that I can share with others with the implementation of a Data Sharing Agreement (it is generally permissive, but my organization requires it so we don't get sued). For a copy of the underlying data, email me at shannon at gfems dot org.

This model reflects the following key decisions, which could have reasonably been made a different way, and I invite you to use your own judgement to adapt these principles to your own needs.

1. This model aims to quantify risk at the import/export step. The underlying data is likely to be biased towards companies that have international trade relationships, as NGOs and journalists focus their efforts there for maximum impact. I would expect the tool to be less accurate in evaluating firms that serve only a domestic market.
2. The challenge of using open-source data is that entities will appear in slight variations in each list. For the purposes of this model, an 'entity' includes all businesses registered by the same (or overlapping subsets of) directors, operating in the same industry, located at the same address or contactable through the same email address. This would include a subsidiary of a garment company that specializes in cotton elastic tape, even if the name is quite different from the parent company. If you implement a similar tool, you might make a different choice, but I felt that labor conditions at the two businesses were likely to be substantially similar.
3. Training data uses two possible criteria: either an entity has been found guilty of a crime analogous to forced labor by a labor tribunal, or there are at least two independent reports by journalists or NGOs that find forced labor conditions at a firm within the five years before data collection began. I never second-guess the first because I assume they have received due process, and the second is the standard used by the financial industry database Refinitiv.
4. We know from prevalence estimates in high-risk industries that the rate of identifiable forced labor is around 5-10% at the worker level, but it is generally detected at a rate of about .01% at the firm level. Therefore, it becomes difficult to truly assess the accuracy of a tool like this. Its accuracy is now measured by internal comparison, but in general, it would be better to undertake an expensive but thorough experiment to train the tool adaptively.

This project could be divided into three major phases: 1) landscaping and data collection 2) technical execution and 3) stakeholder engagement and socialization. 

<b>Landscaping and data collection</b>

From the outset, we knew that data collection would take at least several months to get from identification to data transfers, so we conducted a thorough landscaping in parallel. 

The landscaping exercise turned out to be critically important to our success. We learned from this exactly what the state of the art was in the field and that some solutions existed, but they were often based on closely-guarded audit data, or they used completely uninformative training data. As a result, potential clients of the tool were sophisticated consumers who are rightly skeptical of promises made by each new tool and approach. This experience still informs our pitches today.
