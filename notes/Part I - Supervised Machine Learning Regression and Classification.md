# Introduction to Machine Learning

## 1. Overview of Machine Learning

### 1.1 Understanding AI, Machine Learning, and Deep Learning

<div style="text-align:center">
    <img src="../img/ai-ml-dl.png" alt="Overview of Machine Learning" width="450"/>
</div>

a. **Artificial Intelligence (AI)**

> AI is the overarching discipline that encompasses the creation of intelligent machines capable of performing tasks that typically require human intellect.
> 
> > Example: An intelligent home automation system that adapts the home environment to the preferences of its residents without manual programming.

b. **Machine Learning (ML)**

> ML is a branch of AI focused on algorithms that enable machines to improve at tasks with experience.
> 
> > Example: A fraud detection system that learns to identify fraudulent transactions by analyzing patterns in transaction data.

c. **Deep Learning (DL)**

> DL, a subset of ML, leverages multi-layered neural networks to analyze data, allowing for the modeling of complex patterns.
> 
> > Example: Facial recognition technology used in security systems that can accurately identify individuals even in varying lighting conditions.

---

### 1.2 Technologies, Languages, and Libraries in Machine Learning

#### 1.2.1 Programming Languages

- **Python**
  
  > A versatile language with extensive support for ML through numerous libraries and frameworks.

- **R**
  
  > Focused on statistical analysis and visualization, R is particularly strong in academia and research settings.

#### 1.2.2 Key Libraries and Frameworks

- **NumPy & Pandas**
  
  > Essential Python libraries for numerical computing and data manipulation.

- **scikit-learn**
  
  > A comprehensive library providing simple and efficient tools for predictive data analysis.

- **TensorFlow & PyTorch**
  
  > Leading frameworks for building and training deep learning models with GPU acceleration.

- **Keras**
  
  > High-level neural networks API, written in Python and capable of running on top of TensorFlow.

- **Matplotlib & Seaborn**
  
  > Widely used Python libraries for creating static, interactive, and informative visualizations.

---

### 1.3 When to Use Machine Learning and Deep Learning

While traditional programming is effective for problems with clear rules and logic, machine learning and deep learning shine when such rules are hard to define.

- **Use `Traditional Programming` when:**
  
  > - The problem is well-understood and can be solved with explicit rules.
  > - Data is scarce or not available.
  > - The environment is static, and change is infrequent.
  > - You require 100% interpretability of the model's decisions.

- **Use `Machine Learning` when:**
  
  > - The problem involves complex patterns that are difficult for humans to articulate.
  > - There is an abundance of data available for training.
  > - The task involves making predictions or classifications based on past data.
  > - Adaptability is required as the model needs to evolve with data over time.

- **Use `Deep Learning` when:**
  
  > - The task involves interpreting highly complex and high-dimensional data, such as images, audio, and text.
  > - You have the computational resources to train large neural networks.
  > - You can leverage large labeled datasets for supervised training, or significant amounts of data for unsupervised learning.

Remember, the choice between these approaches depends not only on the nature of the problem and data but also on the available computational resources and the required transparency of the model's decisions.

---

## 2. Supervised Learning

### 2.1 What is Supervised Learning?

**Supervised learning**, also known as `supervised machine learning`, is a subcategory of machine learning and artificial intelligence.

**`It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.`**

As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately, which occurs as part of the cross validation process. 

**<u>Supervised learning helps organizations solve for a variety of real-world problems at scale, such as classifying spam in a separate folder from your inbox.</u>**

### 2.2 How Supervised Learning Works

Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time.

The algorithm measures its accuracy through the `loss function`, adjusting until the error has been sufficiently minimized.

Supervised learning can be separated into two types of problems when data mining:

1. <u>**Classification**</u>
   
   It uses an algorithm to accurately assign test data into specific categories. It recognizes specific entities within the dataset and attempts to draw some conclusions on how those entities should be labeled or defined.
   
   Common classification algorithms are <span style="color:aqua"> linear classifiers </span>, support vector machines (SVM), decision trees, k-nearest neighbor, and random forest, which are described in more detail below.

2. <u>**Regression**</u>

### 2.3 Key Algorithms in Supervised Learning

- **Neural Networks**: Mimic human brain interconnectivity, adjusting based on loss function and gradient descent.
- **Naive Bayes**: Classification approach using the principle of class conditional independence from Bayes' Theorem.
- **Linear Regression**: Predicts relationships between dependent and independent variables.
- **Logistic Regression**: For binary classification problems.
- **Support Vector Machines (SVM)**: Constructs a hyperplane for classification, maximizing distance between data point classes.
- **K-nearest Neighbor (KNN)**: Classifies data points based on proximity to others.
- **Random Forest**: An ensemble of decision trees for improved accuracy and reduced overfitting.

### 2.4 Applications of Supervised Learning

> - **Image and Object Recognition**: For computer vision techniques and imagery analysis.
> - **Predictive Analytics**: Providing insights for strategic decision-making in businesses.
> - **Customer Sentiment Analysis**: Analyzing large data volumes for customer interactions and brand engagement.
> - **Spam Detection**: Training models to filter out spam effectively.

### 2.5 Challenges in Supervised Learning

> - **Expertise Requirement**: Building models requires specialized knowledge.
> - **Time-Consuming Training**: Prolonged training periods for model accuracy.
> - **Data Quality Dependence**: The effectiveness of models relies on high-quality input data.
> - **Limited to Labeled Data**: Inefficiency in handling unlabeled data autonomously.

### 2.6 Comparing Learning Types

> - **Unsupervised Learning**: Focuses on unlabeled data to discover patterns.
> - **Semi-Supervised Learning**: Involves partially labeled input data.

---

## 3. Unsupervised Learning

### 3.1 What is Unsupervised Learning?

> Unsupervised learning, a branch of machine learning, uses algorithms to analyze and cluster unlabeled datasets. These algorithms uncover hidden patterns or data groupings without human intervention. Unsupervised learning is crucial for tasks like exploratory data analysis, cross-selling strategies, customer segmentation, and image recognition.

### 3.2 Common Unsupervised Learning Approaches

Unsupervised learning models are primarily used for clustering, association, and dimensionality reduction. Each approach involves different methodologies and algorithms:

#### 3.2.1 Clustering

- **Exclusive Clustering (Hard Clustering)**:
  
  > Example: K-means clustering groups data into K clusters based on distance from centroids.

- **Overlapping Clustering (Soft Clustering)**:
  
  > Example: Fuzzy k-means clustering allows data points to belong to multiple clusters.

- **Hierarchical Clustering**:
  
  > Agglomerative (bottom-up) or Divisive (top-down) methods are used. Common distance measures include Ward’s linkage, average linkage, complete linkage, and single linkage.

- **Probabilistic Clustering**:
  
  > Gaussian Mixture Models (GMM) cluster data based on distribution probabilities.

#### 3.2.2 Association Rules

- Utilized in market basket analysis for understanding relationships between products.
- Common algorithms: Apriori, Eclat, and FP-Growth.

#### 3.2.3 Dimensionality Reduction

- Techniques like Principal Component Analysis (PCA), Singular Value Decomposition (SVD), and Autoencoders reduce the number of features in a dataset while preserving as much information as possible.

### 3.3 Applications of Unsupervised Learning

Unsupervised learning is applied in various fields, including:

- **News Section Categorization**: Like Google News categorizing articles from various sources.
- **Computer Vision**: For tasks such as object recognition.
- **Medical Imaging**: Assisting in image detection, classification, and segmentation.
- **Anomaly Detection**: Identifying unusual data points in datasets.
- **Customer Persona Creation**: Helping businesses understand customer traits and purchasing habits.
- **Recommendation Engines**: Improving cross-selling strategies based on past purchase behavior.

### 3.4 Comparing with Other Learning Types

- **Supervised Learning**: Uses labeled data for predictive or categorical analysis.
- **Semi-Supervised Learning**: Involves a mix of labeled and unlabeled data, balancing the needs for human intervention and computational complexity.

### 3.5 Challenges in Unsupervised Learning

> - **Computational Complexity**: High volume of training data can increase complexity.
> - **Training Time**: Can require longer training periods.
> - **Risk of Inaccuracy**: Higher chances of inaccurate results without labeled data.
> - **Need for Human Validation**: Often requires human intervention for validating output variables.
> - **Transparency Issues**: Lack of clear understanding of how data was clustered.

---
