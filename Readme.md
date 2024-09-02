# Resume Classification and Semantic Search using Machine Learning

This project implements a comprehensive solution for resume classification and semantic search. It leverages multiple machine learning models and techniques, including transformer-based architectures, for text classification and semantic search.

## Project Overview

This repository includes the following key components:

1. **Text Classification (`1-text-classification.ipynb`)**:
   - **Description**: This file contains a pipeline for classifying resumes into predefined categories.
   - **Technologies Used**: 
     - **Huggingface's Transformers**: Used for fine-tuning transformer models.
     - **Scikit-learn**: For data preprocessing and model evaluation metrics.
     - **Pandas & NumPy**: For data manipulation and analysis.
   - **Purpose**: This file is the backbone of the text classification pipeline, which assigns categories to resume data.

2. **Semantic Search (`2-semantic-search.ipynb`)**:
   - **Description**: Implements a semantic search engine using embeddings to match resumes to specific job descriptions or queries.
   - **Technologies Used**:
     - **Sentence-Transformers**: For generating semantic embeddings for each resume.
     - **Cosine Similarity**: To compare job queries with resume embeddings.
   - **Purpose**: This file provides a way to query resumes based on semantic meaning rather than simple keyword matching, improving search results.

3. **Resume Dataset (`resumes_train.csv`, `resumes_test.csv`, `resumes.csv`)**:
   - **Description**: These files contain raw and processed resume data, split into training and test sets.
   - **Technologies Used**:
     - **CSV**: Used as the format for data storage.
   - **Purpose**: Used for both training models and testing the effectiveness of the classification and search pipelines.

4. **Embeddings Dataset (`embeddings_train.csv`, `embeddings_test.csv`)**:
   - **Description**: Contains the embeddings generated from the resumes for both training and test purposes.
   - **Technologies Used**:
     - **Sentence-Transformers**: To convert the resume text into meaningful embeddings.
   - **Purpose**: These embeddings are used for semantic search to find resumes that match job descriptions.

5. **Requirements (`requirements.txt`)**:
   - **Description**: This file lists the required packages and dependencies for running the project.
   - **Key Technologies**:
     - **Transformers**, **Sentence-Transformers**: For handling large language models and semantic search.
     - **Scikit-learn**, **Pandas**, **Numpy**: For machine learning algorithms, data manipulation, and numerical computations.
     - **Torch & TorchVision**: For deep learning-based computations.
   - **Purpose**: Ensures the necessary dependencies are installed for seamless execution of the notebooks.

6. **Notebook for Resume Data Processing (`0-create-resumes.ipynb`)**:
   - **Description**: Provides an initial processing of resume data to format it appropriately for the classification and semantic search tasks.
   - **Technologies Used**:
     - **Pandas**: For reading, cleaning, and organizing resume data.
     - **Text Preprocessing**: Including techniques such as tokenization and stop-word removal.
   - **Purpose**: This notebook serves to preprocess and prepare raw resumes for use in both the classification and semantic search pipelines.

## How to Run

1. **Install Dependencies**:
   Ensure you have all necessary dependencies installed by running:
   ```bash
   pip install -r requirements.txt
2. **Run the Notebooks**: 
   If all the dependencies are installed correctly, you can run and test all lines with no problems at all. 

