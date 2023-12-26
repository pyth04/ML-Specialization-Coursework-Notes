# Introduction to Machine Learning

## 1. Overview of Machine Learning

### 1.1 Understanding AI, Machine Learning, and Deep Learning

<div style="text-align:center">
    <img src="..sd/img/ai-ml-dl.png" alt="Overview of Machine Learning" width="450"/>
</div>

a. **Artificial Intelligence (AI)**
> AI is the overarching discipline that encompasses the creation of intelligent machines capable of performing tasks that typically require human intellect.
>> Example: An intelligent home automation system that adapts the home environment to the preferences of its residents without manual programming.

b. **Machine Learning (ML)**
> ML is a branch of AI focused on algorithms that enable machines to improve at tasks with experience.
>> Example: A fraud detection system that learns to identify fraudulent transactions by analyzing patterns in transaction data.

c. **Deep Learning (DL)**
> DL, a subset of ML, leverages multi-layered neural networks to analyze data, allowing for the modeling of complex patterns.
>> Example: Facial recognition technology used in security systems that can accurately identify individuals even in varying lighting conditions.

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
