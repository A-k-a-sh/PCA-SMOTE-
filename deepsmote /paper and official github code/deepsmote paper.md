1 

# DeepSMOTE: Fusing Deep Learning and SMOTE for Imbalanced Data 

Damien Dablain, Bartosz Krawczyk<sup>_†_</sup> , _Member, IEEE,_ and Nitesh V. Chawla, _Senior Member, IEEE,_ 

**_Abstract_ —Despite over two decades of progress, imbalanced data is still considered a significant challenge for contemporary machine learning models. Modern advances in deep learning have magnified the importance of the imbalanced data problem. The two main approaches to address this issue are based on loss function modifications and instance resampling. Instance sampling is typically based on Generative Adversarial Networks (GANs), which may suffer from mode collapse. Therefore, there is a need for an oversampling method that is specifically tailored to deep learning models, can work on raw images while preserving their properties, and is capable of generating high quality, artificial images that can enhance minority classes and balance the training set. We propose DeepSMOTE - a novel oversampling algorithm for deep learning models. It is simple, yet effective in its design. It consists of three major components: (i) an encoder/decoder framework; (ii) SMOTEbased oversampling; and (iii) a dedicated loss function that is enhanced with a penalty term. An important advantage of DeepSMOTE over GAN-based oversampling is that DeepSMOTE does not require a discriminator, and it generates high-quality artificial images that are both information-rich and suitable for visual inspection. DeepSMOTE code is publicly available at: https://github.com/dd1github/DeepSMOTE.** 

**_Index Terms_ —machine learning, deep learning, class imbalance, SMOTE, oversampling** 

## I. INTRODUCTION 

EARNING from imbalanced data is among the crucial **L** problems faced by the machine learning community [1]. Imbalanced class distributions affect the training process of classifiers, leading to unfavourable bias towards the majority class(es). This may result in high error, or even complete omission, of the minority class(es). Such a situation cannot be accepted in most real-world applications (e.g., medicine or intrusion detection) and thus algorithms for countering the class imbalance problem have been a focus of intense research for over two decades [2]. Contemporary applications have extended our view of the problem of imbalanced data, confirming that disproportionate classes are not the sole source of learning problems. A skewed class imbalance ratio is often accompanied by additional factors, such as difficult and borderline instances, small disjuncts, small sample size, or the drifting nature of streaming data [2]. These continuously emerging challenges keep the field expanding, calling for novel and effective solutions that can analyze, understand, and 

> _†_ corresponding author 

D. Dablain and N.V. Chawla are with the Department of Computer Science and Engineering, and the Interdisciplinary Center for Network Science and Applications (iCeNSA), the University of Notre Dame, Notre Dame, IN 46556 e-mail: _{_ ddablain,nchawla _}_ @nd.edu 

B. Krawczyk is with Department of Computer Science, Virginia Commonwealth University, Richmond, VA 23284 e-mail: bkrawczyk@vcu.edu 

tackle these data-level difficulties. Deep learning is currently considered as the most promising branch of machine learning, capable of achieving outstanding cognitive and recognition potentials. However, despite its powerful capabilities, deep architectures are still very vulnerable to imbalanced data distributions [3], [4] and are affected by novel challenges such as complex data representations [5], the relationship between imbalanced data and extracted embeddings [6], and learning from an extremely large number of classes [7]. 

**Research goal.** We propose a novel oversampling method for imbalanced data that is specifically tailored to deep learning models and that leverages the advantages of SMOTE [8], while embedding it in a deep architecture capable of efficient operation on complex data representations, such as images. 

**Motivation.** Although the imbalanced data problem strongly affects both deep learning models [9] and their shallow counterparts, there has been limited research on how to counter this challenge in the deep learning realm. In the past, the two main directions that have been pursued to overcome this challenge have been loss function modifications and resampling approaches. The deep learning resampling solutions are either pixel-based or use GANs for artificial instance generation. Both of these approaches suffer from strong limitations. Pixelbased solutions often cannot capture complex data properties of images and are not capable of generating meaningful artificial images. GAN-based solutions require significant amounts of data, are difficult to tune, and may suffer from mode collapse [10], [11], [12], [13]. Therefore, there is a need for a novel oversampling method that is specifically tailored to the nature of deep learning models, can work on raw images while preserving their properties, and is capable of generating artificial images that are of both of high visual quality and enrich the discriminative capabilities of deep models. 

**Summary.** We propose DeepSMOTE - a novel oversampling algorithm for deep learning models based on the highly popular SMOTE method. Our method bridges the advantages of metric-based resampling approaches that use data characteristics to leverage their performance, with a deep architecture capable of working with complex and high-dimensional data. DeepSMOTE consists of three major components: (i) an encoder/decoder framework; (ii) SMOTE-based oversampling; and (iii) a dedicated loss function enhanced with a penalty term. This approach allows us to embed effective SMOTEbased artificial instance generation within a deep encoder / decoder model for a streamlined and end-to-end process, including low dimensional embeddings, artificial image generation, and multi-class classification. 

2 

**Main contributions.** In order for an oversampling method to be successfully applied to deep learning models, we believe that it should meet three essential criteria: (1) it should operate in an end-to-end manner, (2) it should learn a representation of the raw data and embed the data into a lower dimensional _feature space_ , and (3) it should readily generate output (e.g., images) that can be visually inspected. In this paper, we propose DeepSMOTE, which meets these three criteria, and also offer the following scientific contributions to the field of deep learning under class imbalance: 

- **Deep oversampling architecture.** We introduce DeepSMOTE - a self-contained deep architecture for oversampling and artificial instance generation that allows efficient handling of complex-imbalanced and high-dimensional data, such as images. 

- **Simple and effective solution to class imbalance.** Our framework is simple, yet effective in its design. It consists of only three major components responsible for low dimensional representations of raw data, resampling, and 

- **No need for a discriminator during training.** An important advantage of DeepSMOTE over GAN-based oversampling lies in the fact that DeepSMOTE does not require a discriminator during the artificial instance generation process. We propose a penalty function that ensures efficient usage of training data to prime our generator. 

- **High quality image generation.** DeepSMOTE generates high-quality artificial images that are both suitable for visual inspection (they are of identical quality as their real counterparts), and information-rich, which allows for efficient balancing of classes and alleviates the effects of imbalanced distributions. 

- **Extensive experimental study.** We propose a carefully designed and thorough experimental study that compares DeepSMOTE with state-of-the-art oversampling and GAN-based methods. Using five popular image benchmarks and three dedicated skew-insensitive metrics over two different testing protocols, we empirically prove the merits of DeepSMOTE over the reference algorithms. Furthermore, we show that DeepSMOTE displays an excellent robustness to increasing imbalance ratios, being able to efficiently handle even extremely skewed problems. 

**Paper outline.** In this paper, we first provide an overview of the imbalanced data problem and the traditional approaches that have been employed to overcome this issue. Next, we discuss how deep learning methods have been used to generate data and augment imbalanced datasets. We then introduce our approach to imbalanced learning, which combines deep learning with SMOTE. Finally, we discuss our extensive experimentation, which validates the benefits of DeepSMOTE. 

## II. LEARNING FROM IMBALANCED DATA 

The first works on imbalanced data came from binary classification problems. Here, the presence of majority and minority classes is assumed, with a specific imbalance ratio. 

Such skewed class distributions pose a challenge for machine learning models, as standard classifiers are driven by a 0-1 loss function that assumes a uniform penalty over both classes. Therefore, any learning procedure driven by such a function will lead to a bias towards the majority class. At the same time, the minority class is usually more important and thus cannot be poorly recognized. Therefore, methods dedicated to overcoming the imbalance problem aim at either alleviating the class skew or alternating the learning procedure. The three main approaches are: 

**Data-level approaches.** This solution should be viewed as a preprocessing phase that is classifier-independent. Here, we focus on balancing the dataset before applying any classifier training. This is usually achieved in one of three ways: (i) reducing the size of the majority class (undersampling); (ii) increasing the size of minority class (oversampling); or (iii) a combination of the two previous solutions (hybrid approach). Both under- and oversampling can be performed in a random manner, which has low complexity, but leads to potentially unstable behavior (e.g., removing important instances or enhancing noisy ones). Therefore, guided solutions have been proposed that try to smartly choose instances for preprocessing. While not many solutions have been proposed for guided undersampling [14], [15], [16], oversampling has gained much more attention due to the success of SMOTE [8], which led to the introduction of a plethora of variants [17], [18], [19]. However, recent works show that SMOTE-based methods cannot properly deal with multi-modal data and cases with high intra-class overlap or noise. Therefore, completely new approaches that do not rely on _k_ -nearest neighbors have been successfully developed [20], [21]. 

**Algorithm-level approaches.** Contrary to the previously discussed approaches, algorithm-level solutions work directly within the training procedure of the considered classifier. Therefore, they lack the flexibility offered by data-level approaches, but compensate with a more direct and powerful way of reducing the bias of the learning algorithm. They also require an in-depth understanding of how a given training procedure is conducted and what specific part of it may lead to bias towards the majority class. The most commonly addressed issues with the algorithmic approach are developing novel skew-insensitive split criteria for decision trees [22], [23], [24], using instance weighting for Support Vector Machines [25], [26], [27], or modifying the way different layers are trained in deep learning [28], [29], [30]. Furthermore, costsensitive solutions [31], [32], [33] and one-class classification [34], [35], [36] can also be considered as a form of algorithmlevel approaches. 

**Ensemble approaches.** The third way of managing imbalanced data is to use ensemble learning [37]. Here, one either combines a popular ensemble architecture (usually based on Bagging or Boosting) with one of the two previously discussed approaches, or develops a completely new ensemble architecture that is skew-insensitive on its own [38]. One of the most successful families of methods is the combination of Bagging with undersampling [39], [40], [41], Boosting with any resampling technique [42], [43], [44], or cost-sensitive learning with multiple classifiers [45], [46], [47]. Data-level 

3 

techniques can be used to manage the diversity of the ensemble [48], which is a crucial factor behind the predictive power of multiple classifier systems. Additionally, to manage the individual accuracy of classifiers and eliminate weaker learners, one may use dynamic classifier selection [49] and dynamic ensemble selection [50], which ensures that the final decision will be based only on the most competent classifiers from the pool [51]. 

## III. DEEP LEARNING FROM IMBALANCED DATA 

Since the imbalanced data problem has been attracting increasing attention from the deep learning community, let us discuss three main trends in this area. 

**Instance generation with deep neural networks.** Recent works that combine deep learning with shallow oversampoling methods do not give desirable results and traditional resampling approaches cannot efficiently augment the training set for deep models [2], [52]. This leads to an interest in generative models and adapting them to work in a similar manner to oversampling techniques [53]. An encoder / decoder combination can efficiently introduce artificial instances into a given embedding space [54]. Generative Adversarial Networks (GAN) [55], Variational Autoencoders (VAE) [56], and Wasserstein Autoencoders (WAE) [57] have been successfully used within computer vision [58], [59] and robotic control [60], [61] to learn the latent distribution of data. These techniques can also be extended to data generation for oversampling (e.g., medical imaging) [62]. 

VAEs operate by maximizing a variational lower bound of the data log-likelihood [63], [64]. The loss function in a VAE is typically implemented by combining a reconstruction loss with the Kullback-Leibler (KL) divergence. The KL divergence can be interpreted as an implicit penalty on the reconstruction loss. By penalizing the reconstruction loss, the model can learn to _vary_ its reconstruction of the data distribution and thus _generate_ output (e.g., images) based on a latent distribution of the input. 

WAEs also exhibit generative qualities. Similar to VAEs, the loss function of a WAE is often implemented by combining a reconstruction loss with a penalty term. In the case of a WAE, the penalty term is expressed as the output of a discriminator network. 

GANs have achieved impressive results in the computer vision arena [65], [66]. GANs formulate image generation as a min-max game between a generator and a discriminator network [67]. Despite their impressive results, GANs require the use of two networks, are sometimes difficult to train and are subject to mode collapse (i.e., the repetitive generation of similar examples) [10], [11], [12], [13]. 

**Loss function adaptation.** One of the most popular approaches for making neural networks skew-insensitive is to modify their loss function. This approach successfully carried over to deep architectures and can be seen as an algorithmlevel modification. The idea behind modifying the loss function is based on the assumption that instances should not be treated uniformly during training and that errors on minority classes should be penalized more strongly, making it parallel 

to cost-sensitive learning [33]. Mean False Error [68] and Focal Loss [69] are two of the most popular approaches based on this principle. The former simply balances the impact of instances from minority and majority classes, while the latter reduces the impact of easy instances on the loss function. More recently, multiple other loss functions were proposed, such as Log Bilinear Loss [70], Cross Entropy Loss [71], and ClassBalanced Loss [72]. 

**Long-tailed recognition.** This sub-field of deep learning evolved from problems where there is a high number of very rare classes that should nevertheless be properly recognized, despite their low sample size. Long-tailed recognition can be thus seen as an extreme case of the multi-class imbalanced problem, where we deal with a very high number of classes (hundreds) and an extremely high imbalance ratio. Due to very disproportionate class sizes, direct resampling is not advisable, as it will either significantly reduce the size of majority classes or require creation of too many artificial instances. Furthermore, classifiers need to handle the problem of small sample size, making learning from the tail classes very challenging. It is important to note that the majority of works in this domain assume that the test set is balanced. Very interesting solutions to this problem are based on adaptation of the loss function in deep neural networks, such as equalization loss [73], hubless loss [74], and range loss [75]. Recent works suggest looking closer at class distributions and decomposing them into balanced sets – an approach popular in traditional imbalanced classification. Zhou et al. [76] proposed a cumulative learning scheme from global data properties down to class-based features. Sharma et al. [77] suggests using a small ensemble of three classifiers, each focusing on majority, middle, or tail groups of classes. Meta-learning is also commonly used to improve the distribution estimation of tail classes [78]. 

## IV. DEEPSMOTE 

## _A. Motivation_ 

We propose DeepSMOTE - a novel and breakthrough oversampling algorithm dedicated to enhancing deep learning models and countering the learning bias caused by imbalanced classes. As discussed above, oversampling is a proven technique for combating class imbalance; however, it has traditionally been used with classical machine learning models. Several attempts have been made to extend oversampling methods, such as SMOTE, to deep learning models, although the results have been mixed [79], [80], [81]. Metric based oversampling methods, such as SMOTE, can also be computationally expensive because they require access to the full dataset during training and inference. Accessing the full dataset, especially when dealing with image or speech data, can be challenging when using deep learning systems that also require large amounts of memory to store gradients. 

In order for an oversampling method to be successfully applied to deep learning models, we believe that it should meet three essential criteria: 

- 1) It should operate in an end-to-end manner by accepting raw input, such as images (i.e., similar to VAEs, WAEs and GANs). 

~~A~~ | | | | | | > | | | C@O | | >> Algorithm| Data: B: batches1: DEEPSMOTE of imbalanced training data 7 _ Input: Model parameters: 0 = {O09,01,...,O;}: ON Learning Rate: a f ~~<mark>—</mark>~~ : Output: Balanced training set. 2 | <mark>cra</mark> f <u>A</u> | Train the Encoder / Decoder: | o <mark>o</mark> ee <u>nA</u> <mark>_</mark> for e < epochs do ° for m < B do ~~a~~ ee, Ep < encode(B) Dp « decode(Exg) Cp «+ sample(class data) Es «+ encode(Cp) Pr + permute_order(Es) Dp + decode(Pr) Pr = 450%, (Dpi-Coi)” T,=R,+Pr 0:=0- ag 

### Generate Samples: 

for i — number of minority classes do C + select(class data) E + encode(C) G+« SMOTE(E) S < decode(G) 

5 

random percentage (i.e., between 0 and 1) and added to the example instance in order to generate synthetic instances. We simulate SMOTE’s methodology during DeepSMOTE training by selecting a class sample and calculating a distance between the instance and its neighbors (in the embedding or feature space), except that the distance (MSE) during training is used as an implicit penalty on the reconstruction loss. As noted by Arjovsky et al. [13], many generative deep learning models effectively incorporate a penalty, or noise, term in their loss function, to impart diversity into the model distribution. For example, both VAEs and WAEs include penalty terms in their loss functions. We use permutation, instead of SMOTE, during training because it is more memory and computationally efficient. The use of the penalty term, and SMOTE’s fidelity in interpolating synthetic samples during the inference phase, allows us to avoid the use of a discriminator, which is typically used by GAN and WAE models. 

**Artificial image generation** . Once DeepSMOTE is trained, images can be generated with the encoder / decoder structure. The encoder reduces the raw input to a lower dimensional feature space, which is oversampled by SMOTE. The decoder then decodes the SMOTEd features into images, which can augment the training set of a deep learning classifier. 

The main difference between the DeepSMOTE training and generation phases is that during the data generation phase, SMOTE is substituted for the order permutation step. SMOTE is used during data generation to introduce variance; whereas, during training, variance is introduced by permuting the order of the training examples that are encoded and then decoded and also through the penalty loss. SMOTE itself does not require training because it is non-parametric. 

## V. EXPERIMENTAL STUDY 

We have designed the following experimental study in order to answer the following research questions: 

- RQ1: Is DeepSMOTE capable of outperforming state-of-the-art pixel-based oversampling algorithms? 

- RQ2: Is DeepSMOTE capable of outperforming state-of-the-art GAN-based resampling algorithms designed to work with complex and imbalanced data representations? 

- RQ3: What is the impact of the test set distribution on DeepSMOTE performance? 

- RQ4: What is the visual quality of artificial images generated by DeepSMOTE? 

- RQ5: Is DeepSMOTE robust to increasing class imbalance ratios? 

- RQ6: Can DeepSMOTE produce stable models under extreme class imbalance? 

## _A. Setup_ 

**Overview of the Datasets.** Five popular datasets were selected as benchmarks for evaluating imbalanced data oversampling: MNIST [83], Fashion-MNIST [84], CIFAR-10 [85], the Street View House Numbers (SVHN) [86], and Large-scale CelebFaces Attributes (CelebA) [87]. Below we discuss their details, while their class distributions are given in Table I. 

<u>MNIST and Fashion-MNIST.</u> The MNIST dataset consists of handwritten digits and the Fashion-MNIST dataset contains Zalando clothing article images. Both training sets have 60,000 images and the test sets have 10,000 examples. Both datasets contain gray-scale images (1 X 28 X 28), with 10 classes each. <u>CIFAR-10 and SVHN.</u> The CIFAR-10 dataset consists of images, such as automobiles, cats, dogs, frogs and birds, whereas the SVHN dataset consists of small, cropped digits from house numbers in Google Street View images. CIFAR-10 has 50,000 training images and 10,000 test images. SVHN has 73,257 digits for training and 26,032 digits for testing. Both datasets consist of color images (3 X 32 X 32), with 10 classes each. <u>CelebA.</u> The CelebA dataset contains 200,000 celebrity images, each with 40 attribute annotations (i.e., classes). The color images (3 X 178 X 218) in this dataset cover large pose variations and background clutter. For purposes of this study, the images were resized to 3 X 32 X 32 and 5 classes were selected: black hair, brown hair, blond, gray, and bald. 

TABLE I: Class distributions of five benchmark datasets used in experimental evaluation. 

||_M_|_NIST/FM_|_NIST_|_C_|_IFAR/S_|_V HN_||_CELEB_|_A_|
|---|---|---|---|---|---|---|---|---|---|
||Train|Bal. Test|Imbal. Test|Train|Bal. Test|Imbal. Test|Train|Bal. Test|Imbal. Test|
|_Class_||||||||||
|0|4000|1200|1000|4500|1000|1000|9000|900|1000|
|1|2000|1200|500|2000|1000|500|4500|900|500|
|2|1000|1200|250|1000|1000|250|1000|900|111|
|3|750|1200|187|800|1000|187|500|900|55|
|4|500|1200|125|600|1000|125|160|900|17|
|5|350|1200|87|500|1000|87||||
|6|200|1200|50|400|1000|50||||
|7|100|1200|25|250|1000|25||||
|8|60|1200|15|150|1000|15||||
|9|40|1200|10|80|1000|10||||



**Introducing class imbalance.** Imbalance was introduced by randomly selecting samples from each class in the training sets. For the MNIST and Fashion-MNIST datasets, the number of imbalanced examples were: [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]. For the CIFAR-10 and SVHN datasets, the number of imbalanced examples were: [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80]. For CelebA, the number of imbalanced examples were: [9000, 4500, 1000, 500, 160]. For MNIST and Fashion-MNIST, the imbalance ratio of the respective majority class compared to the smallest minority class was 100:1; and for CIFAR-10, SVHN, and CelebA the ratio was approx. 56:1. 

**Reference resampling methods.** In order to evaluate the effectiveness of DeepSMOTE, we compare it to state-of-the-art shallow and deep resampling methods. We have selected four pixel-based modern oversampling algorithms: SMOTE [8], Adaptive Mahalanobis Distance-based Oversampling (AMDO) [88], Combined Cleaning and Resampling (MC-CCR) [89], and Radial-Based Oversampling (MC-RBO) [90]. Additionally, we have chosen two of the top performing GAN-based oversampling approaches: Balancing GAN (BAGAN) [91] and Generative Adversarial Minority Oversampling (GAMO) [92]. BAGAN initializes its generator with the decoder portion of an autoencoder, which is trained on both minority and majority images. GAMO is based on a three-player adversarial game between a convex generator, a classifier network and a discriminator. 



<!-- Start of picture text -->
PoSN ACES aeSRS eeovine i" tO<br>“GaSeidCa AP<br>. Ne aa REee<br><!-- End of picture text -->



<!-- Start of picture text -->
eegee, ae ON GEeenabies<br>_| EO, Gar eee<br><!-- End of picture text -->



<!-- Start of picture text -->
iol %geet, Sa- aes,TS© 3 Bic:-<br>«| > fee, Salt Hae<br><!-- End of picture text -->



<!-- Start of picture text -->
Bel| Eee— epeeNER OHS Boas.eeDace ee<br>| es Ee<br><!-- End of picture text -->

7 

TABLE II: Performance of DeepSMOTE and reference methods on imbalanced test set 

|||_MNIST_||_F_|_MNIS_|_T_||_CIFAR_|||_SV HN_||_C_|_ELEBA_||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||ACSA|GM|F1|ACSA|GM|F1|ACSA|GM|F1|ACSA|GM|F1|ACSA|GM|F1|
|SMOTE|81.48|83.99|82.44|67.94|74.84|67.12|28.02|50.08|29.58|70.18|76.33|71.80|60.29|70.48|60.03|
|AMDO|84.29|88.73|84.88|74.90|80.89|75.39|31.19|53.99|32.44|71.94|78.52|73.06|63.54|72.86|62.94|
|MC-CCR|86.19|92.04|86.46|78.58|86.17|79.03|32.83|56.68|33.91|72.01|80.94|74.26|65.23|77.14|64.88|
|MC-RBO|87.25|94.46|88.69|80.06|88.02|80.14|33.01|59.15|35.83|74.20|82.97|74.91|67.11|80.52|65.37|
|BAGAN|92.56|96.11|93.85|82.50|90.51|82.96|42.41|64.12|43.01|75.81|86.44|77.02|68.62|80.84|**68.33**|
|GAMO|95.45|97.61|95.11|83.05|90.76|83.00|44.72|65.72|**45.93**|75.07|86.00|76.68|66.06|79.11|64.85|
|DeepSMOTE|**96.16**|**98.11**|**96.44**|**84.88**|**91.63**|**83.79**|**45.26**|**66.13**|44.86|**79.59**|**88.67**|**80.71**|**72.40**|**82.91**|66.99|



TABLE III: Performance of DeepSMOTE and reference methods on balanced test set (long-tailed recognition setup 

|||_MNIST_||_F_|_MNIS_|_T_||_CIFAR_|||_SV HN_||_C_|_ELEBA_||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||ACSA|GM|F1|ACSA|GM|F1|ACSA|GM|F1|ACSA|GM|F1|ACSA|GM|F1|
|SMOTE|87.98|89.99|85.02|70.58|76.39|68.06|27.93|42.81|25.10|68.19|74.48|64.28|48.19|56.39|42.19|
|AMDO|88.34|91.03|87.28|72.98|79.36|71.53|31.85|48.19|30.04|71.59|79.13|68.47|51.44|60.73|47.28|
|MC-CCR|90.83|93.18|91.22|75.78|81.04|74.39|33.48|51.18|32.88|74.29|81.62|72.49|58.46|65.39|57.91|
|MC-RBO|91.28|94.62|92.49|76.91|82.14|75.92|39.17|59.29|40.37|75.38|81.98|73.52|61.53|72.95|62.08|
|BAGAN|93.06|95.98|92.77|81.48|89.31|80.93|43.38|63.73|40.25|80.23|86.77|77.75|66.09|77.77|62.84|
|GAMO|95.52|97.47|95.47|83.03|90.26|82.50|44.89|65.30|43.35|80.53|87.17|78.21|66.00|77.71|63.01|
|DeepSMOTE|**96.09**|**97.80**|**96.03**|**83.63**|**90.61**|**83.27**|**45.38**|**65.30**|**43.35**|**80.94**|**87.39**|**78.73**|**69.88**|**80.38**|**69.19**|



TABLE IV: Results of Shaffer post-hoc tests and Bayesian Wilcoxon signed-rank tests with respect to _p_ -values for pairwise comparison between DeepSMOTE and the reference oversampling based methods for three performance metrics. We merged results from imbalanced and long-tailed recognition test scenarios. 

|DeepSMOT<br>vs.|E<br>Sha<br>ACSA|ffer post-<br>GM|hoc<br>F1|Bayesian <br>ACSA|Wilcoxon <br>GM|signed-rank<br>F1|
|---|---|---|---|---|---|---|
|SMOTE|0.00001|0.00000|0.00001|0.00001|0.00000|0.00001|
|AMDO|0.00316|0.00048|0.00329|0.00172|0.00026|0.00188|
|MC-CCR|0.01042|0.00072|0.01003|0.00099|0.00018|0.00083|
|MC-RBO|0.02141|0.01625|0.02331|0.02007|0.01002|0.02106|
|BAGAN|0.03148|0.01352|0.03319|0.02581|0.01039|0.02606|
|GAMO|0.03204|0.01488|0.03582|0.02620|0.01721|0.02938|



while providing an intuitive and easy to tune architecture and, according to both non-parametric and Bayesian tests presented in Table IV, outperforms all pixel-based approaches in a statistically significant manner ( **RQ1 answered** ). 

**Comparison with GAN-based oversampling.** Tables II and III show that regardless of the metric used, DeepSMOTE outperforms the baseline GAN-based models on all but two cases. Both of these situations are happening with F1 measure and for different models (BAGAN displays a slightly higher F1 value on CelebA, while GAMO on CIFAR). It is important to note that for the same benchmarks, DeepSMOTE offers significantly higher ACSA and GM values than any of these reference algorithms, allowing us to conclude that F1 performance variation is not reflective on how DeepSMOTE can handle minority classes. The success of DeepSMOTE can be attributed to better placement of artificial instances and empowering uncertainty areas because oversampling is driven by our penalized loss function. DeepSMOTE can enhance decision boundaries, effectively reducing the classifier bias 

towards the majority classes. Furthermore, DeepSMOTE does not share some of the limitations of GAN-based oversampling, such as mode collapse. As DeepSMOTE is driven by the SMOTE-based approach for selecting and placing artificial instances, we ensure that the minority classes are enriched with diverse training data of high discriminative quality. Table IV shows that DeepSMOTE outperforms all GANbased approaches in a statistically significant manner ( **RQ2 answered** ). This comes with an additional gain of directly generating higher-quality artificial images (as will be discussed in the following experiment). 

We note that the CIFAR-10 dataset was the most challenging benchmark for deep oversampling algorithms. We hypothesize that the reason why the models did not exhibit high accuracy on CIFAR-10 compared to the other datasets is because the CIFAR-10 classes do not have similar attributes. For example, in MNIST and SVHN, all classes are instances of digits and in the case of CelebA, all classes represent faces; whereas, in CIFAR-10, the classes are diverse (e.g., cat, dog, airplane, frog). Therefore, the models are not able to leverage information that they learn from the majority class (which has more examples) to the minority class (which contains fewer examples). In addition, we also noticed that, in some cases, there appears to be a significant overlap of CIFAR-10 class features. 

**Effects of test set distribution.** The final part of the first experiment focused on evaluating the role of class distributions in the test set. In the domain of learning from imbalanced data, the test set follows the distribution of the training set, in order to reflect the actual class disproportions [1]. This also impacts the calculation of several cost-sensitive measures that more severely penalize the errors on minority classes [2]. However, the recently emerging field of long-tailed recognition follows 

8 

a different testing protocol [73]. In this scenario of extreme multi-class imbalance, the training set is skewed, but test sets for most benchmarks are balanced. As DeepSMOTE aims to be a universal approach for imbalanced data preprocessing and resampling, we evaluated its performance in both scenarios. Table II reports results for the traditional imbalanced setup, while Table III reflects the long-tailed recognition setup. We can see that DeepSMOTE excels in both scenarios, confirming our previous observations on its benefits over pixel-based and GAN-based approaches. It is interesting to see that for the long-tailed setup, DeepSMOTE returns slightly better F1 performance on the CIFAR10 and CelebA datasets. This can be explained by the way the F1 measure is calculated, as it gives equal importance to precision and recall. When dealing with a balanced test set, DeepSMOTE was able to return even better performance on these two metrics. For all other metrics and datasets, DeepSMOTE showcases similar trends for imbalanced and balanced test sets. This allows us to conclude that DeepSMOTE is a suitable and effective solution for both imbalanced and long-tailed recognition scenarios ( **RQ3 answered** ). 

## _C. Experiment 2: Quality of artificially generated images_ 

Figures 3 to 7 presents the artificially generated images for all five benchmark datasets by BAGAN, GAMO, and the DeepSMOTE. We can see the quality of DeepSMOTEgenerated images. This can be attributed to DeepSMOTE using an efficient encoding/decoding architecture with an enhanced loss function, as well as preserving class topology via metricbased instance imputation. We note that in the case of GAMO, we present images that were used for classification purposes and not images generated by the GAMO2PIX method, so as to provide a direct comparison of GAMO training images to training images generated by BAGAN and DeepSMOTE. The outcomes of both experiments demonstrate that DeepSMOTE generates artificial images that are both information-rich (i.e., they improve the discriminative ability of deep classifiers and they counter majority bias) and are of high visual quality ( **RQ4 answered** ). 

## _D. Experiment 3: Robustness and stability under varied imbalance ratios_ 

**Robustness to varying imbalance ratios.** One of the most challenging aspects of learning from imbalanced data lies in creating robust algorithms that can manage various datalevel difficulties. Many existing resampling methods return very good results only under specific conditions or under a narrow range of imbalance ratios. Therefore, in order to obtain a complete picture of the performance of DeepSMOTE, we analyze its robustness to varying imbalance ratios in the range of [20,400]. Figure 8 depicts the relationship between the three performance metrics and increasing imbalance ratio on five used benchmarks. This experiment allows us not only to evaluate DeepSMOTE and the reference methods under various skewed scenarios, but also offers a bird-eye view on the characteristics of the performance curves displayed by each examined resampling method. An ideal resampling algorithm 

should be characterized by a high robustness to increasing imbalance ratios, display stable, or small, performance degradation with increased class disproportions. Sharp and significant performance declines indicate breaking points for resampling methods and show when a given algorithm stops being capable of generating useful instances and countering class imbalance. 

Analyzing Figure 8 allows us to draw several interesting conclusions. First, Experiment 1 shows that pixel-based solutions are inferior to their GAN-based counterparts. However, we can see that this observation does not hold for extreme values of imbalance ratios. When the disproportion among classes increases, pixels-based methods (especially MC-CCR and MC-RBO) start displaying increased robustness. On the contrary, the two GAN-based methods are more sensitive to an increased imbalance ratio and we can observe a more rapid decline in their predictive power. This can be explained by two factors: the method by which resampling approaches use the original instances and the issue of small sample size. The former factor shows the limitations of GAN-based methods. While they focus on instance generation and creating highquality images, they do not possess more sophisticated mechanisms on where to precisely inject new artificial instances. With higher imbalance ratios, this placement starts playing a crucial role, as the classifier needs to handle more and more difficult bias. Current GAN-based models use relatively simplistic mechanisms for this issue. On the contrary, pixelbased methods rely on more sophisticated mechanisms, (e.g., MC-CCR uses an energy-based function, while MC-RBO uses local optimization for positioning their artificial instances). With increasing imbalance ratios, such mechanisms start to dominate simpler GAN-based solutions, making pixel-based approaches more robust to extreme imbalance ratios. The latter factor of small sample size also strongly affects GAN-based algorithms. With extreme imbalance, we have less and less minority instances at our disposal, making it more difficult to train effective GANs. 

Compared to both pixel-based and GAN-based approaches, DeepSMOTE displays an excellent robustness even to the highest imbalance ratios. We can see that DeepSMOTE is able to effectively handle such a challenging scenario, displaying the lowest decline of performance on all evaluated metrics. This can be attributed to the fact that SMOTE generates artificial instances following class geometry, while using only nearest neighbors for instance generation. Hence, DeepSMOTE is not affected as strongly as GAN-based approaches by a small sample size andthe need for smart placement of artificial instances, leading to excellent robustness ( **RQ5 answered** ). 

**Model stability under varying imbalance ratios.** Another important aspect of evaluating modern resampling algorithms is their stability. We need to evaluate how a given model reacts to small perturbations in data, as we want to evaluate its generalization capabilities. Models that display high variance under such small changes cannot be treated as stable and thus should not be preferred. It is especially crucial in the learning from imbalanced data area, as we want to select a resampling algorithm that will generate information-rich 

9 









Fig. 3: MNIST minority class images, with rows corresponding to digit classes 









Fig. 4: Fashion MNIST minority class images: trouser / pullover / dress / coat / sandal / shirt / sneaker / bag / ankle boot 









Fig. 5: CIFAR-10 minority class images: automobile / bird / cat / deer / dog / frog / horse / ship / truck 

10 









Fig. 6: SVHN minority class images,with rows corresponding to digit classes 









Fig. 7: CELEBA minority class images: brown hair / blond hair / gray hair / bald 

artificial instances under any data permutations. 

In order to evaluate this, we have measured the spread of performance metrics for DeepSMOTE and GAN-based algorithms under 20 repetitions of 5-fold cross validation. During each CV repetition, minority classes were created randomly from the original balanced benchmarks. This ensured that we not only measure the stability to training data permutation within a single dataset instance, but we also measure the possibility of creating minority classes with instances of varying difficulties. Figure 9 shows the plots of three resampling methods with shaded regions denoting the variance of results. GAN-based approaches display increasing variance under higher imbalance ratios, showing that those approaches cannot be considered as stable models for challenging imbalanced data problems. DeepSMOTE returned the lowest variance within those metrics, showcasing the high stability of our resampling algorithm. This information enriches our previous observation regarding the robustness of DeepSMOTE. Joint analysis of Figures 8 and 9 allows us to conclude that DeepSMOTE can handle extreme imbalance among classes, while generating stable models under challenging conditions ( **RQ6 answered** ). 

## VI. DISCUSSION 

- **Simple design is effective.** DeepSMOTE is an effective approach for countering class imbalance and training skew-insensitive deep learning classifiers. It outperforms state-of-the-art solutions, and is able to work on raw image representations. DeepSMOTE is composed of three components: an encoder/decoder is combined with a dedicated loss function and SMOTE-based resampling. This simplicity makes it an easy to understand, transparent, yet very powerful method for handling class imbalance in deep learning. 

- **Dedicated data encoding for artificial instance generation.** DeepSMOTE uses a two-phase approach that first trains a dedicated encoder/decoder architecture and then uses it to obtain a high quality embedding for the oversampling procedure. This allows us to find the best possible data representations for oversampling, allowing SMOTE-based generation to enrich the training set of minority classes. 

- **Effective placement of artificial instances.** DeepSMOTE follows the geometric properties of minority classes, creating artificial instances on borders among classes. This leads to improved training of discriminative models on datasets balanced with DeepSMOTE. 



<!-- Start of picture text -->
MNIST FMNIST CIFAR10 SVHN CELEBA<br>100 en 60 os 75 So-o.<br>907 Pete 90 | Bi 807. SA nig\ eeoog<br>STS ag Eee 50 Se Nees wee<br>380 an 0 | Sis oes | Sanaa oo] HG bs<br>5 ne = eon Ray O00 404K ONS = ‘o~ = = =, Vg IG ER,<br>fe)<704Po sworewRPTRYqABSS.YES.| S)Ff< RagesASAbog\ \ 2 Nea:MS) g9<30 feg\,NeyeukLNorge, oo 3 ; zQ60= ‘boNn\o-0,\o-0 % WieAR O55= eeeo bogVece\ KeeAes FR| |<br>60 4-e-  MOMc-RBOa \va peRO 60 aren\y-0-0- 3-3Swen° 20 SIT Vonoro-o-Q von 50 “A‘ay 50 \roa weK<br>pa SAGAN aan 50 ‘p-aNs bo ey Ni 45 YN<br>50 {-S-_DeepSMOTE vs oN 10 40 40 “8<br>RISSSRISZRUTRRSISERSclass imbalance ratio RESSSRISSRHTRRSHSSESclass imbalance ratio RESBSRKISSRRTRRISESSclass imbalance ratio RESSSRISSRRTRRSSSESSclass imbalance ratio RSSSSKISSARTRRSISESSclass imbalance ratio<br>100 MNIST FMNIST CIFAR10 SVHN CELEBA<br>OOS 70 PP<br>* ; 907g00% a 9 CES a0 90 nite 80 SWkees |<br>90 obs. aia e eg = Ne. 60 SNe), ooo MERRRya Bw ; ne<br>a5 beeendoooan\, Be Aa ZB80 ala aner ; ‘ “oi\ Proce z50 a One:Pong,Oo.eeVY SS\ ==80 owSBEoe = tteA ; Fyz70 aAosa. Seen o<br>5, voaree e } \ F 0-0-0.[vena\. 340 2,| Fe \ee 3 60 BsEEK<br>© SMOTE \ 0% X, 70 ook wv Po a \ ‘oo, Wo Ve<br>70 4-—t- GAMOBAGAN ~ 60 \eooa!> 29 \\ ‘ 40 “<br>654 DeepSMOTE 10 ° 50<br>RESSSKISSRHTRRSHTESE RESSISISSRRTRRASTESS RESRSKISSRRTRRSSTESS RESRSRTSSRRLRRSSSERS RESSSKISSRRTRRSHSESS<br>class imbalance ratio class imbalance ratio class imbalance ratio class imbalance ratio class imbalance ratio<br>MNIST FMNIST CIFAR10 SVHN CELEBA<br>100 { 60 90 75<br>wy, Sbeto-c 90 + Bea,<br>907, Wy,+ vasKoo nwWRAL a 501356 80 1 Hy a POO 5 70 7 BAYRo-oR-0cone)<br>S008 80 ea NH ON ac Oo 654 °%.\" SRR<br>= \‘NyWHS VS Bya Pon En ame‘a Wes vy. BR on oy “ss Neg 7 ew ee<br>nS _ a an . Ste. x athe of ze hes wy ty Seat WN<br>roo aaa Nog WARS 60 Nate in RR 60 bo, aan s |<br>60 FE4-- toonMC-RBO ae\ \ 50 oeOA 20 BET oneer te]‘o. 5 Lotay<br>~y aw: an ae a) a<br>50 {8 _DeepsMoTe “ 40 . 10 b 40<br>RESRSRIBE RRS ERESSERS RSSSRIBELRERRESSERS RISRERISE RRL RERSSERE RISRERSSSAR GREATS RSE RSRSRSSE<br>class imbalance ratio class imbalance ratio class imbalance ratio class imbalance ratio class imbalanceRRS RRETSERSratio<br><!-- End of picture text -->



<!-- Start of picture text -->
- —YY S—\\ = SS<br>LS ~ \ \S SAY<br>X SS SSS S =~<br>XQ\ \\ S<br>\ \ N<br>XQ NN WS = NS<br><!-- End of picture text -->

12 

- **Superiority over pixel-based and GAN-based algorithms.** DeepSMOTE outperforms state-of-the-art resampling approaches. By being able to work on raw images and extracting features from them, DeepSMOTE can generate more meaningful artificial instances than pixelbased approaches, even while using relatively simpler rules for instance generation. By using efficient and dedicated data embeddings, DeepSMOTE can better enrich minority classes under varying imbalance ratios than GAN-based solutions. 

- **Easy to use.** One of the reasons behind the tremendous success of the original SMOTE algorithm was its easy and intuitive usage. DeepSMOTE follows these steps, as it is not only accurate, but also an attractive off-the-shelf solution. Our method is easy to tune and use on any data, both as a black-box solution and as a steppingstone for developing novel and robust deep learning architectures. As deep learning is being used by a wider and wider interdisciplinary audience, such a characteristic is highly sought after. 

- **High quality of generated images.** DeepSMOTE can return high quality artificial images that under visual inspection do not differ from real ones. This makes DeepSMOTE an all-around approach, since the generated images are both sharp and information rich. 

- **Excellent robustness and stability.** DeepSMOTE can handle extreme imbalance ratios, while being robust to small sample size and within-data variance. DeepSMOTE is less prone to variations in training data than any of the reference methods. It is a stable oversampling approach that is suitable for enhancing deep learning models deployed in real-world applications. 

## VII. CONCLUSION 

**Summary.** We proposed DeepSMOTE, a novel and transformative model that fuses the highly popular SMOTE algorithm with deep learning methods. This allows us to create an efficient oversampling solution for training deep architectures on imbalanced data distributions. DeepSMOTE can be seen as a data-level solution to class imbalance, as it creates artificial instances that balance the training set, which can then be used to train any deep classifier without suffering from bias. DeepSMOTE uniquely fulfilled three crucial characteristics of a successful resampling algorithm in this domain: the ability to operate on raw images, creation of efficient lowdimensional embeddings, and the generation of high-quality artificial images. This was made possible by our novel architecture that combined an encoder/decoder framework with SMOTE-based oversampling and an enhanced loss function. Extensive experimental studies show that DeepSMOTE not only outperforms state-of-the-art pixel-based and GAN-based oversampling algorithms, but also offers unparalleled robustness to varying imbalance ratios with high model stability, while generating artificial images of excellent quality. 

**Future work.** Our next efforts will focus on enhancing DeepSMOTE with information regarding class-level and instance-level difficulties, which will allow it to better tackle 

challenging regions of the feature space. We plan to enhance our dedicated loss function with instance-level penalties for focusing the encoder/decoder training on instances that display borderline / overlapping characteristics, while discarding outliers and noisy instances. Such a compound skew-insensitive loss function will bridge the worlds between data-level and algorithm-level approaches to learning from imbalanced data. Furthermore, we want to make DeepSMOTE suitable for continual and lifelong learning scenarios, where there is a need for handling dynamic class ratios and generating new artificial instances. We envision that DeepSMOTE may not only help to counter online class imbalance, but also help increase the robustness of lifelong learning models to catastrophic forgetting. Finally, we plan to extend DeepSMOTE to incorporate other data modalities, such as graphs. 

## REFERENCES 

- [1] B. Krawczyk, “Learning from imbalanced data: open challenges and future directions,” _Progress in Artificial Intelligence_ , vol. 5, no. 4, pp. 221–232, 2016. 

- [2] A. Fern´andez, S. Garc´ıa, M. Galar, R. C. Prati, B. Krawczyk, and F. Herrera, _Learning from Imbalanced Data Sets_ . Springer, 2018. [Online]. Available: https://doi.org/10.1007/978-3-319-98074-4 

- [3] F. Bao, Y. Deng, Y. Kong, Z. Ren, J. Suo, and Q. Dai, “Learning deep landmarks for imbalanced classification,” _IEEE Trans. Neural Networks Learn. Syst._ , vol. 31, no. 8, pp. 2691–2704, 2020. 

- [4] L. A. Bugnon, C. A. Yones, D. H. Milone, and G. Stegmayer, “Deep neural architectures for highly imbalanced data in bioinformatics,” _IEEE Trans. Neural Networks Learn. Syst._ , vol. 31, no. 8, pp. 2857– 2867, 2020. 

- [5] X. Jing, X. Zhang, X. Zhu, F. Wu, X. You, Y. Gao, S. Shan, and J. Yang, “Multiset feature learning for highly imbalanced data classification,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 43, no. 1, pp. 139–156, 2021. 

- [6] Z. Wang, X. Ye, C. Wang, Y. Wu, C. Wang, and K. Liang, “RSDNE: exploring relaxed similarity and dissimilarity from completelyimbalanced labels for network embedding,” in _Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018_ . AAAI Press, 2018, pp. 475–482. 

- [7] C. Wu and H. Li, “Conditional transferring features: Scaling gans to thousands of classes with 30% less high-quality data for training,” in _2020 International Joint Conference on Neural Networks, IJCNN 2020, Glasgow, United Kingdom, July 19-24, 2020_ . IEEE, 2020, pp. 1–8. 

- [8] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “Smote: synthetic minority over-sampling technique,” _Journal of artificial intelligence research_ , vol. 16, pp. 321–357, 2002. 

- [9] C. Huang, Y. Li, C. C. Loy, and X. Tang, “Deep imbalanced learning for face recognition and attribute prediction,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 42, no. 11, pp. 2781–2794, 2020. 

- [10] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, “Spectral normalization for generative adversarial networks,” _arXiv preprint arXiv:1802.05957_ , 2018. 

- [11] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” _arXiv preprint arXiv:1606.03498_ , 2016. 

- [12] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, “Improved training of wasserstein gans,” _arXiv preprint arXiv:1704.00028_ , 2017. 

- [13] M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein generative adversarial networks,” in _International conference on machine learning_ . PMLR, 2017, pp. 214–223. 

- [14] M. Koziarski, “Radial-based undersampling for imbalanced data classification,” _Pattern Recognit._ , vol. 102, p. 107262, 2020. 

- [15] W. Lin, C. Tsai, Y. Hu, and J. Jhang, “Clustering-based undersampling in class-imbalanced data,” _Inf. Sci._ , vol. 409, pp. 17–26, 2017. 

- [16] P. Vuttipittayamongkol and E. Elyan, “Neighbourhood-based undersampling approach for handling imbalanced and overlapped data,” _Inf. Sci._ , vol. 509, pp. 47–70, 2020. 

13 

- [17] G. Douzas and F. Bac¸˜ao, “Geometric SMOTE a geometrically enhanced drop-in replacement for SMOTE,” _Inf. Sci._ , vol. 501, pp. 118–135, 2019. 

- [18] H. He, Y. Bai, E. A. Garcia, and S. Li, “ADASYN: adaptive synthetic sampling approach for imbalanced learning,” in _Proceedings of the International Joint Conference on Neural Networks, IJCNN 2008, part of the IEEE World Congress on Computational Intelligence, WCCI 2008, Hong Kong, China, June 1-6, 2008_ . IEEE, 2008, pp. 1322– 1328. 

- [19] X. W. Liang, A. P. Jiang, T. Li, Y. Y. Xue, and G. Wang, “LR-SMOTE - an improved unbalanced data set oversampling based on k-means and SVM,” _Knowl. Based Syst._ , vol. 196, p. 105845, 2020. 

- [20] M. Koziarski, B. Krawczyk, and M. Wozniak, “Radial-based oversampling for noisy imbalanced data classification,” _Neurocomputing_ , vol. 343, pp. 19–33, 2019. 

- [21] M. Koziarski and M. Wozniak, “CCR: A combined cleaning and resampling algorithm for imbalanced data classification,” _Int. J. Appl. Math. Comput. Sci._ , vol. 27, no. 4, pp. 727–736, 2017. 

- [22] K. Boonchuay, K. Sinapiromsaran, and C. Lursinsap, “Decision tree induction based on minority entropy for the class imbalance problem,” _Pattern Anal. Appl._ , vol. 20, no. 3, pp. 769–782, 2017. 

- [23] D. A. Cieslak, T. R. Hoens, N. V. Chawla, and W. P. Kegelmeyer, “Hellinger distance decision trees are robust and skew-insensitive,” _Data Min. Knowl. Discov._ , vol. 24, no. 1, pp. 136–158, 2012. 

- [24] F. Li, X. Zhang, X. Zhang, C. Du, Y. Xu, and Y. Tian, “Cost-sensitive and hybrid-attribute measure multi-decision tree over imbalanced data sets,” _Inf. Sci._ , vol. 422, pp. 242–256, 2018. 

- [25] S. Datta and S. Das, “Multiobjective support vector machines: Handling class imbalance with pareto optimality,” _IEEE Trans. Neural Networks Learn. Syst._ , vol. 30, no. 5, pp. 1602–1608, 2019. 

- [26] Q. Fan, Z. Wang, D. Li, D. Gao, and H. Zha, “Entropy-based fuzzy support vector machine for imbalanced datasets,” _Knowl. Based Syst._ , vol. 115, pp. 87–99, 2017. 

- [27] K. Qi, H. Yang, Q. Hu, and D. Yang, “A new adaptive weighted imbalanced data classifier via improved support vector machines with high-dimension nature,” _Knowl. Based Syst._ , vol. 185, 2019. 

- [28] Q. Dong, S. Gong, and X. Zhu, “Imbalanced deep learning by minority class incremental rectification,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 41, no. 6, pp. 1367–1381, 2019. 

- [29] Y. Liu, C. Liu, and S. Tseng, “Deep discriminative features learning and sampling for imbalanced data problem,” in _IEEE International Conference on Data Mining, ICDM 2018, Singapore, November 1720, 2018_ . IEEE Computer Society, 2018, pp. 1146–1151. 

- [30] P. Wang, F. Su, Z. Zhao, Y. Guo, Y. Zhao, and B. Zhuang, “Deep classskewed learning for face recognition,” _Neurocomputing_ , vol. 363, pp. 35–45, 2019. 

- [31] C. Cao and Z. Wang, “Imcstacking: Cost-sensitive stacking learning with feature inverse mapping for imbalanced problems,” _Knowl. Based Syst._ , vol. 150, pp. 27–37, 2018. 

- [32] S. H. Khan, M. Hayat, M. Bennamoun, F. A. Sohel, and R. Togneri, “Cost-sensitive learning of deep feature representations from imbalanced data,” _IEEE Trans. Neural Networks Learn. Syst._ , vol. 29, no. 8, pp. 3573–3587, 2018. 

- [33] C. Zhang, K. C. Tan, H. Li, and G. S. Hong, “A cost-sensitive deep belief network for imbalanced classification,” _IEEE Trans. Neural Networks Learn. Syst._ , vol. 30, no. 1, pp. 109–122, 2019. 

- [34] D. Devi, S. K. Biswas, and B. Purkayastha, “Learning in presence of class imbalance and class overlapping by using one-class SVM and undersampling technique,” _Connect. Sci._ , vol. 31, no. 2, pp. 105–142, 2019. 

- [35] B. Krawczyk, M. Wozniak, and F. Herrera, “Weighted one-class classification for different types of minority class examples in imbalanced data,” in _2014 IEEE Symposium on Computational Intelligence and Data Mining, CIDM 2014, Orlando, FL, USA, December 9-12, 2014_ . IEEE, 2014, pp. 337–344. 

- [36] B. P´erez-S´anchez, O. Fontenla-Romero, and N. S´anchez-Maro˜no, “Selecting target concept in one-class classification for handling class imbalance problem,” in _2015 International Joint Conference on Neural Networks, IJCNN 2015, Killarney, Ireland, July 12-17, 2015_ . IEEE, 2015, pp. 1–8. 

- [37] M. Wozniak, M. Gra˜na, and E. Corchado, “A survey of multiple classifier systems as hybrid systems,” _Information Fusion_ , vol. 16, pp. 3–17, 2014. 

- [38] J. D´ıez-Pastor, J. J. R. Diez, C. I. Garc´ıa-Osorio, and L. I. Kuncheva, “Random balance: Ensembles of variable priors classifiers for imbalanced data,” _Knowl. Based Syst._ , vol. 85, pp. 96–111, 2015. 

- [39] J. Blaszczynski and J. Stefanowski, “Neighbourhood sampling in bagging for imbalanced data,” _Neurocomputing_ , vol. 150, pp. 529–542, 2015. 

- [40] S. Hido, H. Kashima, and Y. Takahashi, “Roughly balanced bagging for imbalanced data,” _Statistical Analysis and Data Mining_ , vol. 2, no. 5-6, pp. 412–426, 2009. 

- [41] S. E. Roshan and S. Asadi, “Improvement of bagging performance for classification of imbalanced datasets using evolutionary multi-objective optimization,” _Eng. Appl. Artif. Intell._ , vol. 87, 2020. 

- [42] S. Datta, S. Nag, and S. Das, “Boosting with lexicographic programming: Addressing class imbalance without cost tuning,” _IEEE Trans. Knowl. Data Eng._ , vol. 32, no. 5, pp. 883–897, 2020. 

- [43] B. Krawczyk, M. Galar, L. Jelen, and F. Herrera, “Evolutionary undersampling boosting for imbalanced classification of breast cancer malignancy,” _Appl. Soft Comput._ , vol. 38, pp. 714–726, 2016. 

- [44] X. Zhang, Y. Zhuang, W. Wang, and W. Pedrycz, “Transfer boosting with synthetic instances for class imbalanced object recognition,” _IEEE Trans. Cybern._ , vol. 48, no. 1, pp. 357–370, 2018. 

- [45] B. Krawczyk, M. Wozniak, and G. Schaefer, “Cost-sensitive decision tree ensembles for effective imbalanced classification,” _Appl. Soft Comput._ , vol. 14, pp. 554–562, 2014. 

- [46] X. Tao, Q. Li, W. Guo, C. Ren, C. Li, R. Liu, and J. Zou, “Self-adaptive cost weights-based support vector machine cost-sensitive ensemble for imbalanced data classification,” _Inf. Sci._ , vol. 487, pp. 31–56, 2019. 

- [47] Q. Zhou, H. Zhou, and T. Li, “Cost-sensitive feature selection using random forest: Selecting low-cost subsets of informative features,” _Knowl. Based Syst._ , vol. 95, pp. 1–11, 2016. 

- [48] J. D´ıez-Pastor, J. J. Rodr´ıguez, C. I. Garc´ıa-Osorio, and L. I. Kuncheva, “Diversity techniques improve the performance of the best imbalance learning ensembles,” _Inf. Sci._ , vol. 325, pp. 98–117, 2015. 

- [49] A. Roy, R. M. O. Cruz, R. Sabourin, and G. D. C. Cavalcanti, “A study on combining dynamic selection and data preprocessing for imbalance learning,” _Neurocomputing_ , vol. 286, pp. 179–192, 2018. 

- [50] P. Zyblewski, R. Sabourin, and M. Wozniak, “Preprocessed dynamic classifier ensemble selection for highly imbalanced drifted data streams,” _Information Fusion_ , vol. 66, pp. 138–154, 2021. 

- [51] M. A. Souza, G. D. C. Cavalcanti, R. M. O. Cruz, and R. Sabourin, “On evaluating the online local pool generation method for imbalance learning,” in _International Joint Conference on Neural Networks, IJCNN 2019 Budapest, Hungary, July 14-19, 2019_ . IEEE, 2019, pp. 1–8. 

- [52] C. Bellinger, R. Corizzo, and N. Japkowicz, “Remix: Calibrated resampling for class imbalance in deep learning,” _CoRR_ , vol. abs/2012.02312, 2020. [Online]. Available: https://arxiv.org/abs/2012. 02312 

- [53] V. A. Fajardo, D. Findlay, C. Jaiswal, X. Yin, R. Houmanfar, H. Xie, J. Liang, X. She, and D. B. Emerson, “On oversampling imbalanced data with deep conditional generative models,” _Expert Syst. Appl._ , vol. 169, p. 114463, 2021. 

- [54] C. Bellinger, C. Drummond, and N. Japkowicz, “Manifold-based synthetic oversampling with manifold conformance estimation,” _Mach. Learn._ , vol. 107, no. 3, pp. 605–637, 2018. 

- [55] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. WardeFarley, S. Ozair, A. Courville, and Y. Bengio, “Generative adversarial networks,” _arXiv preprint arXiv:1406.2661_ , 2014. 

- [56] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” _arXiv preprint arXiv:1312.6114_ , 2013. 

- [57] I. Tolstikhin, O. Bousquet, S. Gelly, and B. Schoelkopf, “Wasserstein auto-encoders,” _arXiv preprint arXiv:1711.01558_ , 2017. 

- [58] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in _Proceedings of the IEEE international conference on computer vision_ , 2017, pp. 2223–2232. 

- [59] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, “Analyzing and improving the image quality of stylegan,” in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2020, pp. 8110–8119. 

- [60] M. Watter, J. T. Springenberg, J. Boedecker, and M. Riedmiller, “Embed to control: A locally linear latent dynamics model for control from raw images,” _arXiv preprint arXiv:1506.07365_ , 2015. 

- [61] R. Bonatti, R. Madaan, V. Vineet, S. Scherer, and A. Kapoor, “Learning visuomotor policies for aerial navigation using cross-modal representations,” _arXiv preprint arXiv:1909.06993_ , 2019. 

- [62] X. Yi, E. Walia, and P. Babyn, “Generative adversarial network in medical imaging: A review,” _Medical image analysis_ , vol. 58, p. 101552, 2019. 

14 

- [63] Z. Hu, Z. Yang, R. Salakhutdinov, and E. P. Xing, “On unifying deep generative models,” _arXiv preprint arXiv:1706.00550_ , 2017. 

- [64] C. Doersch, “Tutorial on variational autoencoders,” _arXiv preprint arXiv:1606.05908_ , 2016. 

- [65] Y. Wu, J. Donahue, D. Balduzzi, K. Simonyan, and T. Lillicrap, “Logan: Latent optimisation for generative adversarial networks,” _arXiv preprint arXiv:1912.00953_ , 2019. 

- [66] X. Chen, Y. Duan, R. Houthooft, J. Schulman, I. Sutskever, and P. Abbeel, “Infogan: Interpretable representation learning by information maximizing generative adversarial nets,” _arXiv preprint arXiv:1606.03657_ , 2016. 

- [67] D. Pfau and O. Vinyals, “Connecting generative adversarial networks and actor-critic methods,” _arXiv preprint arXiv:1610.01945_ , 2016. 

- [68] S. Wang, W. Liu, J. Wu, L. Cao, Q. Meng, and P. J. Kennedy, “Training deep neural networks on imbalanced data sets,” in _2016 International Joint Conference on Neural Networks, IJCNN 2016, Vancouver, BC, Canada, July 24-29, 2016_ . IEEE, 2016, pp. 4368–4374. 

- [69] T. Lin, P. Goyal, R. B. Girshick, K. He, and P. Doll´ar, “Focal loss for dense object detection,” in _IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017_ . IEEE Computer Society, 2017, pp. 2999–3007. 

- [70] Y. S. Resheff, A. Mandelbom, and D. Weinshall, “Controlling imbalanced error in deep learning with the log bilinear loss,” in _First International Workshop on Learning with Imbalanced Domains: Theory and Applications, LIDTA@PKDD/ECML 2017, 22 September 2017, Skopje, Macedonia_ , ser. Proceedings of Machine Learning Research, vol. 74. PMLR, 2017, pp. 141–151. 

- [71] Z. Zhang and M. R. Sabuncu, “Generalized cross entropy loss for training deep neural networks with noisy labels,” in _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´eal, Canada_ , 2018, pp. 8792–8802. 

- [72] Y. Cui, M. Jia, T. Lin, Y. Song, and S. J. Belongie, “Class-balanced loss based on effective number of samples,” in _IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019_ . Computer Vision Foundation / IEEE, 2019, pp. 9268–9277. 

- [73] J. Tan, C. Wang, B. Li, Q. Li, W. Ouyang, C. Yin, and J. Yan, “Equalization loss for long-tailed object recognition,” in _2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020_ . IEEE, 2020, pp. 11 659–11 668. 

- [74] S. Abdelkarim, P. Achlioptas, J. Huang, B. Li, K. Church, and M. Elhoseiny, “Long-tail visual relationship recognition with a visiolinguistic hubless loss,” _CoRR_ , vol. abs/2004.00436, 2020. 

- [75] X. Zhang, Z. Fang, Y. Wen, Z. Li, and Y. Qiao, “Range loss for deep face recognition with long-tailed training data,” in _IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017_ . IEEE Computer Society, 2017, pp. 5419–5428. 

- [76] B. Zhou, Q. Cui, X. Wei, and Z. Chen, “BBN: bilateral-branch network with cumulative learning for long-tailed visual recognition,” in _2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020_ . IEEE, 2020, pp. 9716–9725. 

   - [84] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms,” _arXiv preprint arXiv:1708.07747_ , 2017. 

   - [85] A. Krizhevsky, G. Hinton _et al._ , “Learning multiple layers of features from tiny images,” 2009. 

   - [86] Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng, “Reading digits in natural images with unsupervised feature learning,” 2011. 

   - [87] Z. Liu, P. Luo, X. Wang, and X. Tang, “Deep learning face attributes in the wild,” in _Proceedings of the IEEE international conference on computer vision_ , 2015, pp. 3730–3738. 

   - [88] X. Yang, Q. Kuang, W. Zhang, and G. Zhang, “AMDO: an oversampling technique for multi-class imbalanced problems,” _IEEE Trans. Knowl. Data Eng._ , vol. 30, no. 9, pp. 1672–1685, 2018. 

   - [89] M. Koziarski, M. Wozniak, and B. Krawczyk, “Combined cleaning and resampling algorithm for multi-class imbalanced data with label noise,” _Knowl. Based Syst._ , vol. 204, p. 106223, 2020. 

   - [90] B. Krawczyk, M. Koziarski, and M. Wozniak, “Radial-based oversampling for multiclass imbalanced data classification,” _IEEE Trans. Neural Networks Learn. Syst._ , vol. 31, no. 8, pp. 2818–2831, 2020. 

   - [91] G. Mariani, F. Scheidegger, R. Istrate, C. Bekas, and C. Malossi, “Bagan: Data augmentation with balancing gan,” _arXiv preprint arXiv:1803.09655_ , 2018. 

   - [92] S. S. Mullick, S. Datta, and S. Das, “Generative adversarial minority oversampling,” in _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , 2019, pp. 1695–1704. 

   - [93] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in _Proceedings of the IEEE conference on computer vision and pattern recognition_ , 2016, pp. 770–778. 

   - [94] M. Sokolova and G. Lapalme, “A systematic analysis of performance measures for classification tasks,” _Information processing & management_ , vol. 45, no. 4, pp. 427–437, 2009. 

   - [95] K. Stapor, P. Ksieniewicz, S. Garcia, and M. Wozniak, “How to design the fair experimental classifier evaluation,” _Applied Soft Computing_ , vol. 104, p. 107219, 2021. 

   - [96] A. Benavoli, G. Corani, J. Demsar, and M. Zaffalon, “Time for a change: a tutorial for comparing multiple classifiers through bayesian analysis,” _Journal of Machine Learning Research_ , vol. 18, no. 1, pp. 2653–2688, 2017. 

   - [97] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in _International conference on machine learning_ . PMLR, 2015, pp. 448–456. 

   - [98] A. L. Maas, A. Y. Hannun, and A. Y. Ng, “Rectifier nonlinearities improve neural network acoustic models,” in _Proc. icml_ , vol. 30, no. 1. Citeseer, 2013, p. 3. 

   - [99] V. Nair and G. E. Hinton, “Rectified linear units improve restricted boltzmann machines,” in _Icml_ , 2010. 

   - [100] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” _arXiv preprint arXiv:1412.6980_ , 2014. 

- [77] S. Sharma, N. Yu, M. Fritz, and B. Schiele, “Long-tailed recognition using class-balanced experts,” _CoRR_ , vol. abs/2004.03706, 2020. [Online]. Available: https://arxiv.org/abs/2004.03706 

- [78] M. A. Jamal, M. Brown, M. Yang, L. Wang, and B. Gong, “Rethinking class-balanced methods for long-tailed visual recognition from a domain adaptation perspective,” in _2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020_ . IEEE, 2020, pp. 7607–7616. 

- [79] S. Ando and C. Y. Huang, “Deep over-sampling framework for classifying imbalanced data,” in _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ . Springer, 2017, pp. 770–785. 

- [80] A. Fern´andez, S. Garcia, F. Herrera, and N. V. Chawla, “Smote for learning from imbalanced data: progress and challenges, marking the 15-year anniversary,” _Journal of artificial intelligence research_ , vol. 61, pp. 863–905, 2018. 

- [81] J. M. Johnson and T. M. Khoshgoftaar, “Survey on deep learning with class imbalance,” _Journal of Big Data_ , vol. 6, no. 1, pp. 1–54, 2019. 

- [82] A. Radford, L. Metz, and S. Chintala, “Unsupervised representation learning with deep convolutional generative adversarial networks,” _arXiv preprint arXiv:1511.06434_ , 2015. 

- [83] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” _Proceedings of the IEEE_ , vol. 86, no. 11, pp. 2278–2324, 1998. 

