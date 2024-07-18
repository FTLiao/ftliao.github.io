---
title: "Wanna dance like a pro? A neural network can help you"
disqus: hackmd
date: 2019-10-18T21:40:07+02:00
drafts: false
params:
    math: true
---

<!--

(1) A brief introduction: How to achieve photorealistic video translation with GANs?
===
(2) xxxxxx: an approach throguh GAN
===

-->
<!-- <header>
    <h1>
Wanna dance like a pro? A neural network can help you
       <p><font size="+1">
       <i>
       A brief introduction to photorealistic video translation with GANs
       </i>
       </font></p>
       <p><font size="+0">
       Feng-Ting Liao, Ru-Han Wu
       </font></p>
    </h1>
</header>
 -->


**A brief introduction to photorealistic video translation with GANs**

Feng-Ting Liao, Ru-Han Wu



*The original post apears [here](https://medium.com/@scarecrow/a-brief-introduction-to-photorealistic-video-translation-with-gans-e1ffcc989a12)*

<!--
TBC:
1. make reference numbering consistent
2. Check the abstract symbols. Data symbols should not be italic (only the variables are).
3. Check the consistency of math symbols
## Table of Contents
[TOC]
-->

Imagine you see your favourite star, perhaps it's Bruno Mars, Beyonce, BTS, or TWICE, dancing amazingly in music videos. You then wonder if you could dance just as good as they do and maybe replicate their moves and reproduce what they did in the videos. Same posture, same movement and even the same tricks of yourself match at the exact timestamp to the stars in their videos. To have a video of such, you start practising very hard or perhaps, finding ways to synthesise frames of images of yourself into making the moves. Such a video was not something achievable in the past without a team of post-effect experts or can be generated with conventional computer vision techniques. However, thanks to the research in deep learning and generative models, the creation of it can potentially be done with ease and without you spending months and years perfecting your dancing skills.



<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/Mqw9Pr1.gif" alt="drawing" width="400"/>
</div> -->

![image alt](https://i.imgur.com/Mqw9Pr1.gif#center)
Figure 1: Replicating the dance moves of Bruno Mars in That's What I Like. Screenshot taken from [Everybody Dance Now demo video](https://www.youtube.com/watch?v=PCBTZh41Ris)

The technology behind this kind of video generation is called video-to-video translation. It has been an active area of research in deep learning and has great potential in not just creating cool dancing videos but also in other applications such as image inpainting or post-effect editing. In this article, we would like to introduce how the translation works and what are the key ingredients for creating a photorealistic translated video.



## Introduction to video-to-video translation
<!--
So what's video-to-video translation and how to make it 
photorealistic? 
-->

Technically speaking, the translation generates a new video resembling a target video by transforming a piece of contextual information, e.g. a mask representing a person, from a source video. It is achieved with two neural networks: a feature extraction network; and a generative network. The feature extraction network will extract contextual information from the source video; the generative network, learnt from the target video, takes the information and generate translated video. For example, in Figure 2., a feature extraction network, e.g. Densepose[[6](^densepose)], takes frames of a source video of a dancing professional and extract related semantic information, i.e. contextual masks. Then, such information goes into a generative network which synthesises it with a target video of a non-dancing person. The resulting synthesised video will look as if the non-dancing person in the target video is dancing just as good as that in the source video.


<!-- <div style="text-align:center">
    <img src="https://tcwang0509.github.io/vid2vid/paper_gifs/pose.mp4" alt="drawing" width="800"/>
</div> -->

<!-- <div style="text-align:center">
    <video src="https://tcwang0509.github.io/vid2vid/paper_gifs/pose.mp4" alt="drawing" width="800"/>
</div> -->

![video alt](https://tcwang0509.github.io/vid2vid/paper_gifs/pose.mp4#center)

Figure 2: Examples of synthesising dance $\rightarrow$ pose $\rightarrow$ dance videos. For each set, contexual masks (bottom middle) are extracted from source video (left) for generating the synthesised video (right). Taken from [vid2vid](https://tcwang0509.github.io/vid2vid/paper_gifs/pose.mp4). 

Past state-of-the-art algorithms for the video translation task [[1](^vid2vid)][[2](^ebdn)][[3](^pcloning)] are based on generative adversarial network (GAN) [[4](^gan)]. These GANs, with appropriate designs of the prior conditioning, network architecture, objective function, and temporal smoothing, can generate photorealistic videos. 

In this blog post, we will explain critical ingredients in designing a generative network with GAN for photorealistic video translation. We will first introduce the framework of the video-to-video translation task and the technology, i.e. conditional generative adversarial network [[12](^cgan)], behind it. Then, we will discuss other factors such as network architecture and loss functions in improving the resolution of image frames of the videos. Last but not least, we will introduce temporal influence from adjacent frames and optical flows of frames which are crucial for the temporal smoothness of the generated videos.



## How to do the translation with a framework of neural networks?

The goal of video-to-video translation is to learn a mapping function from a sequence of contextual masks of a source video to an output photorealistic video that captures precisely the content of the source video. The basis of this consists of translations at the image-to-image level and the temporal coherent level on frames at different times. Both tasks can be incorporated into a general framework commonly used by [[1](^vid2vid)][[2](^ebdn)][[3](^pcloning)][[9](^pix2pixhd)]. 

The framework has two parts, a transfer stage and a training stage. For the transfer stage, images $\mathbf{x}^\prime$ of a source video goes into a context extraction network, which, in the example of Figure 3 (top) from [[2](^ebdn)], is a pose detection network $P$. The output pose joints $\mathbf s^\prime$ of $P$ are transformed by a normalization process and then passed into a generator $G$ that generates target images $G(\mathbf s^{\prime\prime})$ resembling the context information from source images $\mathbf x^{\prime}$. The success of such resemblance depends on how $G$ is trained in the training stage (see Figure 3 (bottom) for example). In this stage, $G$ is learnt alongside an adversarial discriminator $D$ which tries to differentiate a real pair of image and context information $(\mathbf x_t, \mathbf{s}_t)$ from a corresponding fake pair $(G(\mathbf s_t), \mathbf s_t)$ at frame $t$. The learnt $G$ would then be able to generate images resembling the target images with given context information $\mathbf{s}$.

<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/4IIPUIy.png" alt="drawing" width="800"/>
</div> -->

![image alt](https://i.imgur.com/4IIPUIy.png#center)


Figure 3: Framework of the video translation task. (Top): Pose movement of the source frame $x_t^\prime$ can be translated into the target frame $G(s^{\prime\prime})$ with the use of a pose extraction network $P$ and a mapping generator $G$. (Bottom): Adversarial training of $G$ with target video frames and the discriminator $D$. The loss functions for training are described in detail in a following section. (Taken from [EBDN](https://arxiv.org/abs/1808.07371) with minor modification)


## Conditional GAN: the backbone of the framework

The stages of the framework for generating synthesised video stems from the idea of conditional generative adversarial networks (cGANs). It allows the networks to be trained with given conditional information, e.g. a semantic map or pose joints introduced in the previous section.

Before we dive deep into explaining how cGANs work for the translation task, we will give a brief recap on the generative adversarial network. This recap should hopefully provide a better intuition on how "conditional" works. 

**Generative Adversarial Networks**

Generative Adversarial Networks (GANs) [[4](^gan)] are deep neural network architectures that can learn to estimate the underlying probability distribution of a given set of training data with an end-to-end fashion. For example (Figure 4), with a well trained GAN, we can have a generator network taking random noise as input to generate fake images of digits. Therefore, we can now generate samples from the learnt probability distribution that may not be present in the original training set. 


<!-- <div style="text-align:center">
    <img src="https://cdn-images-1.medium.com/max/1600/1*XKanAdkjQbg1eDDMF2-4ow.png" alt="drawing" width="800"/>
</div> -->

![image alt](https://cdn-images-1.medium.com/max/1600/1*XKanAdkjQbg1eDDMF2-4ow.png#center)

Figure 4: Generative Adversarial Networks framework. Image credit: [Thalles Silva](https://www.freecodecamp.org/news/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394/)

GANs comprise two networks, a generator $G$ and a discriminator $D$, in competition with each other. Given a training set $X$, the generator $G$ takes a random vector input $z$ and tries to produce data instances similar to those in the training set. The discriminator network $D(x)$ is a binary classifier that attempts to distinguish between real data instances and their fake counterparts generated by $G$. 

The two networks play a min-max game with each other for obtaining global optimality. The loss function for optimising such a game is

$$
\begin{equation}
    \underset{G}{\text{min}}\
       \underset{D}{\text{max}}\
           V(D, G) = 
           \mathbb{E}_{x\sim p_{data}(x)}[\text{log} D(x)] + 
           \mathbb{E}_{z\sim p_{z}(z)}[\text{log}(1-D(G(z)))]\text{.}        
    \label{eq:GANloss}
    \tag{1}
\end{equation}
$$ 

The equation suggests that, on one hand, we would like to minimise the contribution to the cross entropy from $D$ and, on the other hand, to maximise the contribution to the entropy from $G$. By optimising the game, it is possible to generate fake digits (Figure 5) purely from the input of some random vectors.


<!-- <div style="text-align:center">
    <img src="https://tensorflow.org/images/gan/dcgan.gif" alt="drawing" width="300"/>
</div> -->

![image alt](https://tensorflow.org/images/gan/dcgan.gif#center)

Figure 5: Generated MNIST digits. Each digit is generated with a random vector input.  Taken from [Tensorflow: Deep Convolutional Generative Adversarial Network](https://www.tensorflow.org/beta/tutorials/generative/dcgan)


**Conditional Generative Adversarial Networks**

<!---
Bascially, quickly point out the use of the conditional information
--->
Conditional Generative Adversarial Networks (cGANS) is an extension of GAN's framework by substituting random vectors in the canonical GANs for feature vectors as its prior. Specifically, it incorporates the prior as some kind of auxiliar information $\mathbf s$ such as class labels or data from other modalities for the training of the generator and the discriminator. Also, importantly, the prior provides control over modes of the generated outcomes. As a result, an ideal cGAN can learn a multi-modal mapping from inputs to outputs by feeding it with different contextual information. The mapping makes cGANs suitable for image-to-image translation tasks, i.e. an output image can be generated with its corresponding contextual information.

The training of cGANs is the same as that of GANs, i.e. playing a min-max game. The objective function of such a game is

$$
\begin{equation}
    \underset{G}{\text{min}}\ 
    \underset{D}{\text{max}}\ V(D, G) = 
               \mathbb{E}_{x\sim p_{data}(x)}    [\text{log} D(x\mid s)] + 
               \mathbb{E}_{z\sim p_{z}(z)}[\text{log}(1-D(G(z\mid s)))]\text{,}
    \label{eq:CGAN}
    \tag{2}
\end{equation}
$$ 

where the samples $x$ and $z$ are conditioned on a  given information $s$. Interestingly, with such a small addition of the prior, the optimisation results in a trained generator capable of generating outputs based on different modes. For instance, as shown in Figure 6, at each row, a trained generator created fake images conditioned on one provided label (i.e. a digit, encoded as an one-hot vector). It no longer generates digits randomly but translate the conditioned information into particular digits.



<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/1bJohWG.png" alt="drawing" width="800"/>
</div> -->

![image alt](https://i.imgur.com/1bJohWG.png#center)
Figure 6: Generated MNIST digits, each row conditioned on one label. Therefore, by swapping the labels, a specific digit can be generated. [Taken from cGANs paper](^cgan).


## How to improve resolution of translated images?

Network architectures and the loss functions of the framework are other key ingredients in line to consider for generating images of high resolution for videos synthesis. Two challenges in the design of these are: mapping global contextual information, i.e. contextual masks, to RGB pixels; and refining locations and the sharpness of local pixels. 

In this section, we will discuss the network architectures and the loss functions for addressing these challenges.

**Architecture-wise** 

A generator for the image translation task should be able to generate global contextual information while enhancing resolution at local pixel-level. One such generator, proposed by [Wang et al.](^pix2pixhd), is a network architecture of two-stages. The first stage ($G_1$) network is for capturing global context and generating coarser images; the second stage ($G_2$) network is for enhancing pixel resolution and generating finer images. For example, a full generator, shown in Figure. 7, is expressed as $G=\{G_1, G_2\}$. The $G_1$, the global generator, operates at $\frac{1}{4}$ of the semantic input maps (or any contextual maps)  whereas $G_2$, the local enhancer, works at the original size of the maps. Both generators use similar downsampling/upsampling blocks and residual blocks to construct the feature maps. This design of the generator has proven to be effective in [[1](^vid2vid)][[9](^pix2pixhd)][[3](^ebdn)][[11](^ploss)] for generating images of quality resolution.


<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/dj58WIO.png" alt="drawing" width="800"/>
</div> -->

![image alt](https://i.imgur.com/dj58WIO.png#center)
Figure 7: The architecture of coarse-to-fine generator. The global network $G_1$ and the local enhancer network $G_2$ operates on semantic maps of different scales. Such design improves the resolution of generated images. Image taken from [pix2pixHD.](^pix2pixhd)


As for the network of the discriminator, it has to differentiate generated images from real images effectively. It also should be able to capture objects of various sizes in the source image. To achieve these, a discriminator with a large receptive field is desirable. However, this requires a network to be deep, have large kernels, or use dilation in convolution layers, all of which can increase the network capacity but are memory intensive for training. 


<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/8eg8oty.png" alt="drawing" width="450"/>
</div> -->

![image alt](https://i.imgur.com/8eg8oty.png#center)
Figure 8: Multi-scale image discriminator. Three identical discriminators are trained on the same set of input images and semantic maps at different scales. Taken from [Research at NVIDIA: Video to Video Synthesis](https://www.youtube.com/watch?v=GrP_aOSXt5U&feature=youtu.be).


An effective solution is a multi-scale discriminator consisting of multiple discriminators operating at different image scales but trained on an identical network architecture [[9](^pix2pixhd)]. The discriminator operating at smaller image size has a relatively large receptive field and has a larger global view of images. On the other hand, the discriminator operating at larger image size has smaller receptive field and sees finer details of images and thus encourages generators to produce finer details. Empirically, without the use of such discriminator, the generated images show repeated patterns as pointed out by [Wang et al.](^pix2pixhd).

**Design a good loss function**

Another key factor in generating images of high resolution is the design of the loss function for training the GAN. It is reported that loss functions such as the least-squares generative adversarial network (LSGAN) loss and perceptual loss are beneficial.


LSGAN proposed by [Mao. et al.](^lsgan) is shown to be capable of generating more realistic images than the canonical GAN loss of the Equation 1. The loss introduces an $a$-$b$-$c$ coding scheme to represent the data, where $a$ and $b$ are the labels of real data and fake data respectively, and $c$ denotes the value that generator wants discriminator to believe for fake data. The function is expressed as 

$$
\begin{align*}
    \underset{D}{\text{min}}\ V_{\text{LSGAN}}(D) &= \frac{1}{2}\mathbb{E}_{x\sim p_{data}(x)}[(D(x)-b)^2]+\frac{1}{2}\mathbb{E}_{z\sim p_z(z)}[(D(G(z))-a)^2]
    \\
    \underset{G}{\text{max}}\ V_{\text{LSGAN}}(G) &= \frac{1}{2}\mathbb{E}_{z\sim p_z(z)}[(D(G(z))-c)^2]\text{.}
    \\
    \label{eq:LSGAN}
    \tag{3}
\end{align*}
$$

If the conditions, $b-c=1$ and $b-a=2$, are satisified, it is found that minimising the Equation 3 yields minimising the Pearson $\chi^{2}$ divergence of a relation between the probability distributions of the generator and the discriminator.

So why can the LSGAN loss be useful for the translation task? The benefit is due to its capability in penalising correctly classified samples even though they lie in a long way from the decision boundary. $x$, GAN loss tends to fail to do so owing to its use of sigmoid function. The function, as a result, leads to the problem of vanishing gradients for samples far from the decision boundaries. Also, qualitatively, LSGAN loss constrain the data better than GAN loss does (Figure 9). The loss value of the LSGAN is flat at one point, whereas that of the GAN saturates as the $\mathcal{x}$ (the value of a sample, e.g. an image) grows.


<!-- <div style="text-align:center">
    <img src="http://1.bp.blogspot.com/-eN13aeENvVg/WODZ4P0KCqI/AAAAAAAABfw/vUAyBGI45WglkfNI0EB5ddkSEAIEeLlSQCK4B/s1600/lsgan_4.PNG" alt="drawing" width="600"/>
</div> -->

![image alt](http://1.bp.blogspot.com/-eN13aeENvVg/WODZ4P0KCqI/AAAAAAAABfw/vUAyBGI45WglkfNI0EB5ddkSEAIEeLlSQCK4B/s1600/lsgan_4.PNG#center)
Figure 9: Comparing the loss values between GAN loss and LSGAN loss. (a): Sigmoid cross entropy loss function of GAN (b) Least squares loss function of LSGAN. Taken from [LSGAN](https://arxiv.org/abs/1611.04076).

The other commonly used loss function for generating high resolution images is the perceptual loss. It simply is the Euclidean distance between feature representations of the outputs of the generated image and the original image at a layer of a pretrained loss network such as VGG [[5](^vgg)]. The loss can be denoted as 

$$
\begin{equation}
    \mathcal{L}_{VGG}(\hat{x}, x) = \frac{1}{C_jH_jW_j}     \|\phi_j(\hat{x})-\phi_j(x)\|^2_2\text{,}
    \label{eq:vggloss}
    \tag{4}
\end{equation}
$$ 

where $\hat{x}$ and $x$ are the generated and original images respectively and $\phi_j(x)$ is the feature map of the shape ${C_jH_jW_j}$ at $j$-th layer of a loss network.
The reason why $\mathcal{L}_{VGG}$ is useful is that, rather than enforcing the approximation of $\hat{x} \sim x$ directly, the loss measures the perceptual and semantic information of them through feature maps of another convolutional network. Such network, usually pretrained on image classification tasks, has alrealy learnt to encode the perceptual and semantic information within. 

<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/UKv5A4h.png" alt="drawing" width="250"/>
</div> -->

![image alt](https://i.imgur.com/UKv5A4h.png#center)
Figure 10: An qualitative example from [[3](^pcloning)] showing the resolution and perception improvement of the generated images due to the use of the perceptual loss. Inference results of a generator trained with: (Left): L1 loss and GAN loss; (Right): VGG loss and GAN loss.


## How to improve temporal smoothness of generated video?

To generate photorealistic videos, directly applying image translation approaches to video frames is not enough. It often leads to temporally incoherent videos of low visual quality [[1](^vid2vid)]. Such temporal inconsistency of a generated video could be solved with two approaches: conditioning $G$ and $D$ on consecutive frames; and pixel-wise optical flow prediction with a dedicated flow net.


**Network conditions on consecutive frames**

One intuitive way to improve temporal coherence of a generated video is to condition on information from previous frames. Previous work from *Everybody Dance Now* [[2](^ebdn)], illustrated in Figure 12, treats pairs of images and contextual pose joints of the current frame and its previous frame together in the training stage. This method, empirically, shows to generate videos of good temporal smoothness.

<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/3fpzF7G.png" alt="drawing" width="400"/>
</div> -->

![image alt](https://i.imgur.com/3fpzF7G.png#center)
Figure 11: Temporal smoothing setup in [*Everybody Dance Now*](^ebdd). When synthesizing the current frame $G(s_{t})$, the generator conditions on its corresponding pose $s_{t}$ and the previously generated frame $G(s_{t-1})$. Discriminator then attempts differentiate the real temporally coherence sequence $(s_{t-1}, s_{t}, x_{t-1}, x_{t})$ from the fake sequence $(s_{t-1}, s_{t}, G(s_{t-1}), G(s_{t}))$.


Formally speaking, we can express the conditioning on consecutive frames with Markov assumption by factorising the conditional distribution to a product form. It is 

$$
\begin{equation}
    \mathcal{p}(\tilde {\mathbf x}_{1}^{T}\mid {\mathbf s}_{1}^{T}) = 
    \prod_{t=1}^{T}
    \mathcal{p}(\tilde {\mathbf x}_{t} \mid \tilde {\mathbf x}_{t-L}^{t-1}, {\mathbf s}_{t-L}^{t}),
    \tag{5}
\end{equation}
$$ 


where ${\mathbf s}$ is a sequence of contextual frames, e.g. semantic segmentation maps or edge masks, of source video frames and $\tilde {\mathbf x}$ is a sequence of output video frames. The equation describes that generator network can learn a mapping from the past $L$ source frames \({\mathbf s}_{t-L}^{t}\) and the past $L-1$ generated frames $\tilde {\mathbf x}_{t-L}^{t-1}$ to a newly generated $t$-th output frame.



Furthermore, *Vid2vid*, expanding the idea of the Equation 4, introduces a sampling operator in selecting past $K$ consecutive frames for training a multi-scale video disciminator in the time domain leading to better video quality. It subsampes frames of past original images, generated images, contextual maps, and optical flow maps with a fine-to-coarse manner (Figure 12). In the finest scale, the discriminator takes in consecutive frames. In the temporally coarser scale, it subsamples the frames by a factor of $K$, meaning skipping $K-1$ intermediate frames for every sampling. Such mechanism ensures the short-term consistency kept at the finer scale while perserving the long-term coherence at the coarser scale. 



<div style="text-align:center">
    <img src="https://i.imgur.com/Qokxw9A.gif" alt="drawing" width="450"/>
</div>

![image alt](https://i.imgur.com/Qokxw9A.gif#center)

Figure 12: Subsampling process of a video discriminator that takes both adjacent frames and flow maps for ensuring temporal consistency. The clip demonstrates three scales of video downsampling by a factor of $K$ at $1$, $2$, and $4$, respectively. Frames not sampled are shown in dark grey. The intermediate frames between sampled frames are their corresponding optical flow maps. Taken from [Research at NVIDIA: Video to Video Synthesis](https://www.youtube.com/watch?v=GrP_aOSXt5U&feature=youtu.be)


**Use optical flow to predict next frame**

The other perspective to improve temporal smoothness is by considering optical flow. Optical flow describes the pattern of apparent motion of image objects, such as velocities and positions at the pixel level, between consecutive frames due to the movement of object or camera. In other words, it represents the temporal dynamic of pixels between frames. Such dynamics can be expressed in colour coding (Figure 13) where colours and intensities of the pixels represent velocities of them relative to the centre of an image. An example clip in Figure 14 visualises the dynamic changes. The foreground objects, e.g. moving cars or rapid moving items, shows more substantial speed variations than the background objects, e.g. buildings and street signs. 

<!-- <div style="text-align:center;">
    <img src="https://i.imgur.com/JLl34gR.png" alt="drawing" width="300"/>
</div> -->

![image alt](https://i.imgur.com/JLl34gR.png#center)
Figure 13: Colour-coding of the optical flow field. The colours and their intensities represent the velocity vectors of pixels relative to the centre of the square. Smaller vectors are faint, and different colours represent different moving directions. 

<!-- <div style="text-align:center">
    <img src="https://i.imgur.com/PGXHwZe.gif" alt="drawing"/>
</div> -->

![image alt](https://i.imgur.com/PGXHwZe.gif#center)
Figure 14: An example of flow maps of corresponding cityscapes. (Top): clip of a street view camera. (Bottom): the corresponding optical flow of the clip. The colours follow the coding scheme discussed in Figure 13. Taken from [FlowNet 2.0 demo.](https://www.youtube.com/watch?v=JSzUdVBmQP4)

[*Vid2vid*](^vid2vid) leverages the property of optical flow map and incorporates a flow net into the generator to better synthesis the foreground objects. The flow map is used to estimate the next frame by moving the pixels in the current frame to the location prescribed by the optical flow (also known as warping). The warping technique has two benefits. First, the optical flow serves as a practical background prior since the background area dominates a frame and varies little between consecutive frames. The flow estimation would be mostly accurate except for the occluded areas. Second, the flow lets the hallucination network in the generator to focus on synthesising the more difficult foreground objects. Such difficulty comes from their significant motion and a small fraction of occupancy within an image. The foreground-background-prior with the optical flow proves to be an ideal approach in improving the synthesis performance. It reduces flickering artefacts and improves visual quality (Figure 15). Also, the synthesised video with it is shown qualitatively by user studies to be preferable than videos without it [[1](^vid2vid)].



![image alt](https://tcwang0509.github.io/vid2vid/paper_gifs/cityscapes_comparison.gif#center)
Figure 15: Generating a photorealistic video from an input segmentation map video on Cityscapes. The reduction of the flicker of lines on the road and foreground objects can be easily observed in the bottom right image. Top left: semantic input map. Top right: [pix2pixHD](^pix2pixhd). Bottom left: COVST. Bottom right: [vid2vid](^vid2vid).


## TL;DR
We have introduced all the necessary ingredients for synthesising photorealistic videos. We foresee that technology may have a considerable benefit in assisting current image or video inpainting tools. It may also help in creating a new form of video entertainment. 

We hope that this article is informative enough for you to understand the topic of video-to-video translation. Last but not least, let's watch a cool demo video from one of these papers. Enjoy!

**Everybody Dance Now**
<!-- <iframe width="640" height="480"
src="https://www.youtube.com/embed/PCBTZh41Ris">
</iframe> -->


## References

[1] [Video-to-Video Synthesis](https://arxiv.org/abs/1808.06601) \
[2] [Everybody Dance Now](https://arxiv.org/abs/1808.07371) \
[3] [Performance cloning](https://arxiv.org/abs/1808.06847) \
[4] [Generative adversarial networks. In Advances in Neural Information Processing Systems (NIPS), 2014. 2, 3](https://arxiv.org/abs/1406.2661) \
[5] [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) \
[6] [Dense Human Pose Estimation In The Wild]
(http://densepose.org) \
[7] [FlowNet2.0](https://arxiv.org/abs/1612.01925) \
[8] [Pix2Pix](https://arxiv.org/abs/1611.07004) \
[9] [Pix2PixHD](https://arxiv.org/abs/1711.11585) \
[10] [LSGAN](https://arxiv.org/abs/1611.04076) \
[11] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution: Supplementary Material](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf) \
[12]: [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
