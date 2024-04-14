# Deep-Learning-Physics-as-Inverse
This repository contains a replication of the paper "Physics as Inverse Graphics: Unsupervised Physical Parameter Estimation from Video". 

**Contributors (Group 74)**: Felipe Bononi Bello, Michael van Breukelen, Áron Dömötör Kohlhéb, Vanessa Timmer

**Original paper**: https://arxiv.org/pdf/1905.11169.pdf

**Original repository**: https://github.com/seuqaj114/paig

\===============================================================

In this blog, unsupervised physical parameter estimation from video is explained. In 2020, M. Jaques, M. Burke and T. Hospedales \[1] published an article that expanded the concept of physical parameter estimation from video to use differentiable physics engines. As part of the Deep Learning CS4240 course of the Delft University of Technology, we have attempted to reproduce this paper. The hope of this blog post  is to provide the reader with a detailed understanding of the findings of this paper and to discuss the process and result of the reproduction.

The core challenge addressed in the paper, Physics-as-Inverse-Graphics: Unsupervised Physical Parameter Estimation From Video, revolves around the concept of physical parameter estimation from video data, a task traditionally reliant on heavily supervised methods with large amounts of labelled data \[1]. This research breaks new ground by proposing a model that can perform unsupervised physical parameter estimation. It does this by integrating vision-as-inverse-graphics with differentiable physics engines, thereby allowing the model to learn about objects and their dynamics without the need for explicit labels. 

![](https://lh7-us.googleusercontent.com/1o_QV4WHgl_qaCYqk5utYuHDmLEtzkiuTfKRLpK1OfgNFkn-AC4aLPEYyorPv2o4pNNXgeA4DgB4znlHXBjY7v9F33JoPHzkq1PrSzTsWgxyYe1_2EoRva6AlINjVnLYuxEmEkZT1EA5XjrJHPt1GF8)

At the heart of the paper's methodology is the new "physics-as-inverse-graphics" approach. This technique uses the power of spatial transformers and a differentiable physics engine to extract and learn the underlying physical parameters and states of objects captured in video.This model is special as it has the ability to determine the explicit trajectory coordinates and velocities of objects, enabling understanding of physical dynamics purely from visual data. The differentiable physics engine can be found in Equation 1. Where the position and velocity of timestep t-1 can be used to find the position at t. The velocity at time t uses the velocity at time t-1 and the governing force equation F. This governing physics equation depends on the system being analysed.

![](https://lh7-us.googleusercontent.com/WXU_gQqgjpAv7drPVFsaKmLH2nFNg95g4UarsO6D14-ROSwu9iE94NkiuiMDpdcdAv3ZTm9Ssvuzw7RHVZZHAhtSyWx9dX1aonzbs_eEfwnACIIbnZpFHhZQilFHkj0BYqmkzEjYssSJwlNlZ9AG8kI)

By enabling models to learn and predict physical states and parameters directly from pixels, the approach paves the way for more intuitive and efficient machine learning systems. Whether for predicting the motion of celestial bodies or designing robots that can navigate and interact with the physical world.

![](https://lh7-us.googleusercontent.com/6i1S-R3aaHtStYSCbxghkHGYtG-P2LDlq0D0jzxJuqzjw8IzimmxQfeEHOqj7BxY-E4H8FYwvbxmdCHK1ztBMinIXQVAjyUUfKONObtzpeKmOUBTkP4keHAMNzv23o42DLC945wPl_G5vlM6OfW4lbY)


### _Challenges and Opportunities_<a id="challenges-and-opportunities"></a>

While the results are promising, the paper also candidly addresses limitations and areas for further research. The challenge of representing objects and scenes in a manner that mirrors real-world complexity remains, as does the need for models that can adapt to dynamic backgrounds and variable object counts. Yet, these challenges also present opportunities for future innovation, pushing the boundaries of what's possible in unsupervised learning and physical parameter estimation.


### _Reproduction_<a id="reproduction"></a>

To reproduce the paper the main strategy was to replicate and update the [original code base](https://github.com/seuqaj114/paig.git) given that it used multiple deprecated libraries such as TensorFlow 1.12 and Pycharm 2.2. We ran into several issues during the reproduction that ultimately led to a product that is not working yet. The subparts of the network such as the generators and iterators, the encoder and decoder, the cells and the blocks, and the runners are, to the best of our knowledge, correctly replicated. However, the physics models script which integrates them remains unreplicated at the moment. 

During the reproduction we found that the methods used often interlink strongly, making the understanding and the rewriting of the code more difficult. The shape of the transformed data is often changed to comply with the layers, but it has led to issues. The paper or their source code didn’t provide enough information on the shapes required by each layer, therefore we couldn’t get the velocity encoder to work. We tried to train with respect to only the autoencoder loss as well, but we couldn’t separate the training from the prediction and the feedforward due to their integral role in the source code. 

A main problem that is commonly encountered in deep learning research is that software updates can render previously functional codebases unusable \[2]. This is a common issue for Python projects \[3], which often have many dependencies. This was particularly applicable in our case, as the original codebase could not even be replicated using the appropriate library versions because they were no longer supported. Consequently, replicating each component of the neural network was difficult to validate since we could not run the original code or access the output values

Another issue that significantly hindered our replication was the discrepancies between the implementation in the codebase and the model description in the actual research paper. Many components described in the paper were, in reality, implemented with slight changes or different parameters that were left unexplained. Deciding whether to follow the codebase or the paper during implementation introduced ambiguity and uncertainty regarding how exactly the model worked and which method was correct.

Converting from TensorFlow to PyTorch also presented some difficulties. The first issue was the difference in the image data format. While TensorFlow uses NHWC (samples, height, width, channels) to define the image input format, PyTorch uses NCHW. This required us to carefully examine the codebase to detect for what inputs the convention was being used and how to account for that in our replication. Despite some initial issues with this difference, it was eventually resolved. 

Another interesting difference in the original codebase was the conventions used to define classes and their methods in a neural network. In Pytorch, typically a network class will be defined, consisting of several separate layer classes from the used library. A method within the network class is then defined for performing a forward pass over all the layers. Once a network object is instantiated, the forward(x) method can be called on the instance to perform a forward pass. This typical structure was not used in the original codebase, which is written in Tensorflow; instead, both the network class, layers, and forward pass were all conducted within a single function. The rest of the codebase also followed this method when performing forward passes. For clarity and sticking to best practices as defined for Pytorch, our reproduction was written with standard practices in mind, which also required significant refactoring of the rest of the code and how the network class and its methods were used.


### _Conclusion_<a id="conclusion"></a>

In conclusion, public access to the code and dataset go a long way in promoting reproducibility but are not guaranteed to be sufficient. Ensuring the original code persists, whether by automated or manual methods, is also crucial. Extensive descriptions of the experiment that include expected input and output shapes at different stages could also facilitate reproductions, even in the event that the original code is no longer executable. 

Nevertheless, "Physics-as-Inverse-Graphics" stands as a testament to the potential of merging the worlds of computer vision and physical simulation. By enabling machines to learn the unseen laws that govern our universe, this research not only advances our technological capabilities but also deepens our understanding of the fundamental principles that shape our reality.


### _Task Division_<a id="task-division"></a>

|                       |                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Áron Dömötör Kohlhéb  | Decoder, Autoencoder loss, Blogpost: reproduction                                                            |
| Felipe Bononi Bello   | Encoder, Decoder, Blogpost: paper summary                                                                    |
| Michael van Breukelen | Encoder, Network Classes, Poster,Blogpost: reproduction                                                      |
| Vanessa Timmer        | UNet, Velocity estimator, def conv\_feedforward, run\_physics.py, general debugging, initial work on Encoder |


### _References_<a id="references"></a>

\[1] M. Jaques, M. Burke, and T. Hospedales, “PHYSICS-AS-INVERSE-GRAPHICS: UNSUPERVISED PHYSICAL PARAMETER ESTIMATION FROM VIDEO,” 2020. Accessed: Apr. 14, 2024. [Online]. Available: https://arxiv.org/pdf/1905.11169.pdf

\[2] J. M. González-Barahona and G. Robles, “On the reproducibility of empirical software engineering studies based on data retrieved from development repositories,” _Empirical Software Engineering_, vol. 17, no. 1–2, pp. 75–89, Oct. 2011, doi: https\://doi.org/10.1007/s10664-011-9181-9.

\[3] S. Mukherjee, A. Almanza, and C. Rubio-González, “Fixing dependency errors for Python build reproducibility,” _eScholarship University of California (University of California)_, Jul. 2021, doi: <https://doi.org/10.1145/3460319.3464797>.
