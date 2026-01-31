🎬 Movie Recommendation System
==============================

**Item-based Collaborative Filtering with Popularity Fallback (MovieLens 100K)**

An end-to-end recommendation system that suggests movies to users based on their rating history using collaborative filtering and a cold-start fallback strategy. The project includes data processing, model building, evaluation, and an interactive Streamlit application.

🚀 Project Overview
-------------------

This project demonstrates how to build a real-world recommender system pipeline:

✔ Data ingestion and preprocessing✔ User-item interaction matrix construction✔ Item-based Collaborative Filtering (cosine similarity)✔ Cold-start handling with popularity fallback✔ Offline evaluation (Precision@K, Recall@K)✔ Interactive recommendation app (Streamlit)

The system uses the **MovieLens 100K** dataset, which contains anonymized user ratings for movies.

🧠 Recommendation Approach
--------------------------

### 1️⃣ Collaborative Filtering (Item-Based)

We use **Item-Item Collaborative Filtering**, where similarity between movies is computed based on user rating patterns.

**Steps:**

1.  Build a **user-item matrix** from training data
    
2.  Compute **cosine similarity** between item vectors
    
3.  For a target user, recommend movies similar to the ones they rated highly
    

**Why Item-based CF?**

*   Scales better than user-based for large systems
    
*   Stable item similarities
    
*   Widely used in production recommenders
    

### 2️⃣ Cold-Start Handling

Collaborative filtering fails when users have very little history.To solve this, we introduce a **popularity fallback**.

If a user has fewer than _N_ rated movies:➡ The system recommends globally popular movies instead.

This ensures:✔ Stable recommendations✔ Better UX for new users✔ Production-ready robustness

📊 Evaluation
-------------

We evaluate the recommender using **ranking metrics**:

MetricMeaning**Precision@K**How many of the top-K recommended movies are actually relevant**Recall@K**How many of the user’s relevant movies appear in the top-K list

### Example Results (MovieLens 100K)

ModelPrecision@10Recall@10Popularity baseline0.13790.1452ItemCF**0.21880.2555**ItemCF + Fallback**0.2704 (P@5)0.1680 (R@5)**

Collaborative filtering significantly outperforms the popularity baseline.

🏗 Project Structure
--------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   09-movie-recommender/  │  ├── app/  │   └── streamlit_app.py        # Interactive web app  │  ├── data/  │   ├── raw/                    # Original MovieLens files  │   └── processed/              # Train/test splits & artifacts  │  ├── notebooks/  │   └── analysis.ipynb          # EDA & metric visualization  │  ├── src/  │   ├── data_loader.py          # Load dataset  │   ├── preprocessing.py        # Train/test split & matrix creation  │   ├── model_itemcf.py         # Item similarity computation  │   ├── model_popularity.py     # Popularity baseline  │   └── recommend.py            # Recommendation logic  │  └── requirements.txt   `

🎯 Streamlit Application
------------------------

The project includes an interactive web app that allows you to:

👤 Select a user profile (anonymized)📊 View rating history and favorite movies🎯 Generate Top-K recommendations🧊 Automatically apply fallback for sparse users

The UI simulates how recommender systems are exposed in real products.

⚙️ How to Run
-------------

### 1️⃣ Install dependencies

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

### 2️⃣ Run the Streamlit app

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run app/streamlit_app.py   `

The app will automatically download the MovieLens dataset if not present.

🔮 Future Improvements
----------------------

If this system were extended toward production, next steps would include:

🔹 Matrix Factorization (SVD / ALS)🔹 Time-aware recommendations (recency weighting)🔹 Content-based features (genres, metadata)🔹 Online A/B testing🔹 Real-time user embeddings

📚 Dataset
----------

MovieLens 100K dataset[https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)

Contains:

*   100,000 ratings
    
*   943 users
    
*   1,682 movies
    
*   Ratings from 1 to 5
    

👩‍💻 Author
------------

Built as part of an Applied Machine Learning portfolio project demonstrating recommender system design, evaluation, and deployment.

🔍 How the Algorithm Works (Step-by-Step)
-----------------------------------------

This section explains the full recommendation pipeline from raw data to final suggestions.

### Step 1 — Load and Prepare Data

The MovieLens dataset provides user ratings for movies.Each record contains:

*   user\_id
    
*   item\_id (movie)
    
*   rating
    
*   timestamp
    

We split the dataset **by user** into train and test sets to simulate real-world unseen interactions.

### Step 2 — Build the User-Item Matrix

We transform the training data into a **user-item interaction matrix**:

Movie 1Movie 2Movie 3...User A503...User B040...

*   Rows = Users
    
*   Columns = Movies
    
*   Values = Ratings (understanding of preference strength)
    

This matrix is typically very **sparse** (most users rate only a few items).

### Step 3 — Compute Item Similarity

We compute similarity between movie vectors using **cosine similarity**:

sim(i,j)=vi⋅vj∥vi∥∥vj∥sim(i, j) = \\frac{v\_i \\cdot v\_j}{\\|v\_i\\|\\|v\_j\\|}sim(i,j)=∥vi​∥∥vj​∥vi​⋅vj​​

Where:

*   viv\_ivi​ and vjv\_jvj​ are rating vectors of two movies
    
*   Similar movies have similar rating patterns across users
    

This results in an **item-item similarity matrix**.

### Step 4 — Generate Recommendations

For a target user:

1.  Identify movies the user rated highly
    
2.  Find similar movies using the similarity matrix
    
3.  Aggregate similarity scores
    
4.  Remove movies already seen
    
5.  Return Top-K unseen items
    

### Step 5 — Cold-Start Strategy

If the user has fewer than a defined number of ratings:

➡ Collaborative filtering becomes unreliable➡ We recommend **globally popular movies** instead

This is a common strategy in production systems to ensure quality recommendations for new users.

📈 Evaluation Methodology
-------------------------

To evaluate the system, we simulate real-world recommendation:

1.  Hide part of each user's interactions (test set)
    
2.  Generate recommendations using training data
    
3.  Compare recommendations to held-out items
    

### Metrics Used

#### 🔹 Precision@K

How many of the recommended items were actually relevant.

Precision@K=Relevant items in Top-KKPrecision@K = \\frac{\\text{Relevant items in Top-K}}{K}Precision@K=KRelevant items in Top-K​

#### 🔹 Recall@K

How many of the user’s relevant items were successfully recommended.

Recall@K=Relevant items in Top-KTotal relevant itemsRecall@K = \\frac{\\text{Relevant items in Top-K}}{\\text{Total relevant items}}Recall@K=Total relevant itemsRelevant items in Top-K​

These are standard **ranking metrics** used in recommender systems.

🧩 Key Engineering Decisions
----------------------------

### Why Item-Based Instead of User-Based?

*   Item similarity is more stable
    
*   Scales better when number of users grows
    
*   Widely used in industry
    

### Why Cosine Similarity?

*   Handles sparse high-dimensional data well
    
*   Measures direction rather than magnitude of ratings
    

### Why Popularity Fallback?

*   Cold-start is unavoidable in collaborative filtering
    
*   Popular items provide a safe and strong baseline
    

💡 Lessons Learned
------------------

Building this project highlights real challenges in recommender systems:

✔ Data sparsity significantly impacts performance✔ Cold-start must always be handled✔ Offline metrics do not fully represent user satisfaction✔ UI explainability improves trust in recommendations

🖥 Demo Application Features
----------------------------

The Streamlit app allows:

🎬 Viewing user profile behavior⭐ Inspecting favorite (5-star) movies📊 Exploring rating history🎯 Generating real-time recommendations🧊 Automatic fallback for sparse users

The interface mimics how recommender systems are integrated into real products.

🌐 Live Demo

You can try the interactive movie recommender app here:

👉 [**Open the Streamlit App**](https://applied-scientist-portfolio-aeptjhnqf8vcnrwc8be5mq.streamlit.app/)

The app allows you to explore anonymized user profiles, view their rating history, and generate personalized movie recommendations in real time.