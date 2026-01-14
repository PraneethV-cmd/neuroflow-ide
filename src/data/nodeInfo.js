
export const nodeInfo = {
    // --- Data Input ---
    start: {
        title: 'Start Node',
        description: 'The starting point of your workflow. It initiates the execution flow.',
        usage: [
            'Always place this at the beginning of a pipeline',
            'Connects to your initial data loading nodes'
        ],
        inputs: [],
        outputs: ['CSV Reader Node', 'Database Reader Node'],
        notes: 'Every valid pipeline must effectively start with a data source, often triggered by this node.'
    },
    csvReader: {
        title: 'CSV Reader Node',
        description: 'Reads data from CSV or Excel files to bring external datasets into NeuroFlow.',
        usage: [
            'Importing raw data from local files',
            'Starting a new analysis pipeline'
        ],
        inputs: ['Start Node'],
        outputs: ['Data Cleaner Node', 'Encoder Node', 'Normalizer Node', 'Describe Node'],
        notes: 'Ensure your file has a header row for best results.'
    },
    databaseReader: {
        title: 'Database Reader Node',
        description: 'Fetches data from an external database using SQL queries.',
        usage: [
            'Importing data from SQL databases',
            'Selecting specific subsets of data via queries'
        ],
        inputs: ['Start Node'],
        outputs: ['Data Cleaner Node', 'Encoder Node', 'Normalizer Node'],
        notes: 'Requires valid connection credentials.'
    },

    // --- Data Preprocessing ---
    dataCleaner: {
        title: 'Data Cleaner Node',
        description: 'Cleaning data by handling missing values, duplicates, and outliers.',
        usage: [
            'Filling or dropping missing values (NaN)',
            'Removing duplicate rows',
            'Handling outliers'
        ],
        inputs: ['CSV Reader Node', 'Database Reader Node'],
        outputs: ['Encoder Node', 'Normalizer Node', 'Feature Selector Node'],
        notes: 'Perform cleaning before normalization or encoding.'
    },
    dataTypeConverter: {
        title: 'Data Type Converter Node',
        description: 'Converts columns between different data types (e.g., text to number).',
        usage: [
            'Fixing incorrect data types inferred during load',
            'Preparing columns for mathematical operations'
        ],
        inputs: ['CSV Reader Node', 'Data Cleaner Node'],
        outputs: ['Encoder Node', 'Normalizer Node'],
        notes: 'Verify conversions to avoid data loss.'
    },
    encoder: {
        title: 'Encoder Node',
        description: 'Converts categorical text data into numerical format using Label or One-Hot encoding.',
        usage: [
            'Preparing string categories for machine learning',
            'Converting "Yes"/"No" to 1/0'
        ],
        inputs: ['Data Cleaner Node', 'Data Type Converter Node'],
        outputs: ['Normalizer Node', 'Feature Selector Node', 'Regression Nodes'],
        notes: 'One-Hot encoding increases dimensionality.'
    },
    normalizer: {
        title: 'Normalizer Node',
        description: 'Scales numeric data to a standard range (e.g., 0-1) to improve model performance.',
        usage: [
            'Scaling features with different units',
            'Improving convergence speed of algorithms'
        ],
        inputs: ['Encoder Node', 'Data Cleaner Node'],
        outputs: ['Train/Test Split', 'Regression Nodes', 'Classification Nodes'],
        notes: 'Essential for distance-based algorithms like KNN and K-Means.'
    },
    featureSelector: {
        title: 'Feature Selector Node',
        description: 'Selects the most relevant features (columns) to use for training models.',
        usage: [
            'Reducing dimensionality',
            'Focusing on important variables',
            'Removing irrelevant noise'
        ],
        inputs: ['Normalizer Node', 'Encoder Node'],
        outputs: ['Regression Nodes', 'Classification Nodes', 'Clustering Nodes'],
        notes: 'Removing too many features may lose information.'
    },
    pca: {
        title: 'PCA Node',
        description: 'Performs Principal Component Analysis to reduce dataset dimensionality while preserving variance.',
        usage: [
            'Visualizing high-dimensional data in 2D/3D',
            'Reducing features to speed up training'
        ],
        inputs: ['Normalizer Node'],
        outputs: ['K-Means Node', 'Data Visualizer Node'],
        notes: 'Data should be normalized before PCA.'
    },
    svd: {
        title: 'SVD Node',
        description: 'Performs Singular Value Decomposition for dimensionality reduction.',
        usage: [
            'Matrix decomposition',
            'Latent semantic analysis'
        ],
        inputs: ['Normalizer Node'],
        outputs: ['Regression Nodes', 'Data Visualizer Node'],
        notes: 'Similar to PCA but works on the data matrix directly.'
    },

    // --- Regression Models ---
    linearRegression: {
        title: 'Linear Regression Node',
        description: 'Predicts a continuous target variable based on a linear relationship with one feature.',
        usage: [
            'Predicting prices, scores, or trends',
            'Understanding simple relationships'
        ],
        inputs: ['Normalizer Node', 'Feature Selector Node'],
        outputs: ['Model Evaluator Node', 'Model Visualizer Node'],
        notes: 'Assumes a linear relationship between input and output.'
    },
    multiLinearRegression: {
        title: 'Multi-Linear Regression Node',
        description: 'Predicts a continuous target variable based on multiple input features.',
        usage: [
            'Predicting complex outcomes with many factors',
            'Sales forecasting, risk assessment'
        ],
        inputs: ['Normalizer Node', 'Feature Selector Node'],
        outputs: ['Model Evaluator Node'],
        notes: 'Watch out for multicollinearity (highly correlated features).'
    },
    polynomialRegression: {
        title: 'Polynomial Regression Node',
        description: 'Models non-linear relationships by fitting a polynomial equation to the data.',
        usage: [
            'Modeling curved trends',
            'When linear models underfit the data'
        ],
        inputs: ['Normalizer Node', 'Feature Selector Node'],
        outputs: ['Model Evaluator Node', 'Model Visualizer Node'],
        notes: 'High degrees can lead to overfitting.'
    },
    knnRegression: {
        title: 'KNN Regression Node',
        description: 'Predicts values by averaging the targets of the K nearest neighbors.',
        usage: [
            'Non-linear prediction without assuming a formula',
            'Localized predictions'
        ],
        inputs: ['Normalizer Node', 'Select Features'],
        outputs: ['Model Evaluator Node'],
        notes: 'Computationally expensive on large datasets.'
    },

    // --- Classification Models ---
    logisticRegression: {
        title: 'Logistic Regression Node',
        description: 'Predicts the probability of a categorical outcome (e.g., Yes/No).',
        usage: [
            'Binary classification (Spam/Not Spam)',
            'Probability estimation'
        ],
        inputs: ['Normalizer Node', 'Encoder Node'],
        outputs: ['Model Evaluator Node', 'Model Visualizer Node'],
        notes: 'Ideally suited for binary classification.'
    },
    knnClassification: {
        title: 'KNN Classification Node',
        description: 'Classifies a data point based on the majority class of its neighbors.',
        usage: [
            'Simple classification tasks',
            'Multiclass classification'
        ],
        inputs: ['Normalizer Node'],
        outputs: ['Model Evaluator Node'],
        notes: 'Sensitive to the choice of K and distance metric.'
    },
    naiveBayes: {
        title: 'Naive Bayes Node',
        description: 'Probabilistic classifier based on Bayes\' theorem with an assumption of independence.',
        usage: [
            'Text classification',
            'Spam filtering',
            'Baseline classification'
        ],
        inputs: ['Encoder Node', 'Feature Selector Node'],
        outputs: ['Model Evaluator Node'],
        notes: 'Assumes features are independent of each other.'
    },

    // --- Clustering Models ---
    kMeans: {
        title: 'K-Means Clustering Node',
        description: 'Groups unlabeled data into K distinct clusters based on similarity.',
        usage: [
            'Customer segmentation',
            'Image compression',
            'Pattern discovery'
        ],
        inputs: ['Normalizer Node', 'PCA Node'],
        outputs: ['Data Visualizer Node', 'Cluster Analysis'],
        notes: 'You must specify the number of clusters (K).'
    },
    dbscan: {
        title: 'DBSCAN Node',
        description: 'Density-based clustering that finds core samples and expands clusters from them.',
        usage: [
            'finding clusters of arbitrary shape',
            'Detecting outliers (noise)'
        ],
        inputs: ['Normalizer Node'],
        outputs: ['Data Visualizer Node'],
        notes: 'Great for noisy data, does not require specifying K.'
    },
    hierarchicalClustering: {
        title: 'Hierarchical Clustering Node',
        description: 'Builds a hierarchy of clusters using agglomerative or divisive approaches.',
        usage: [
            'Taxonomy creation',
            'Understanding data hierarchy'
        ],
        inputs: ['Normalizer Node'],
        outputs: ['Dendrogram View'],
        notes: 'Can be computationally intensive for large data.'
    },

    // --- Visualization & Misc ---
    dataVisualizer: {
        title: 'Data Visualizer Node',
        description: 'Creates charts and graphs (Scatter, Bar, Line) to explore your dataset.',
        usage: [
            'Exploratory data analysis',
            'Presenting results'
        ],
        inputs: ['Any Data Node'],
        outputs: [],
        notes: 'Click "Visualize" to open the plotting modal.'
    },
    modelVisualizer: {
        title: 'Model Visualizer Node',
        description: 'Visualizes the learned decision boundaries or regression lines of a model.',
        usage: [
            'Debugging models',
            'Understanding model behavior'
        ],
        inputs: ['Regression/Classification Models'],
        outputs: [],
        notes: 'Currently supports 2D visualizations.'
    },
    heatmap: {
        title: 'Heatmap Node',
        description: 'Displays a correlation matrix or 2D density plot of the data.',
        usage: [
            'Identifying correlated features',
            'Visualizing density'
        ],
        inputs: ['Data Cleaner Node', 'Normalizer Node'],
        outputs: [],
        notes: 'Useful for feature selection.'
    },
    modelEvaluator: {
        title: 'Model Evaluator Node',
        description: 'Calculates performance metrics (MSE, Accuracy, R2) for trained models.',
        usage: [
            'Comparing model performance',
            'Validating results on test data'
        ],
        inputs: ['Regression/Classification/Clustering Models'],
        outputs: [],
        notes: 'Crucial for verifying model quality.'
    },
    describeNode: {
        title: 'Describe Node',
        description: 'Provides statistical summaries (mean, std, min, max) of the dataset.',
        usage: [
            'Quick statistical overview',
            'Checking data distribution'
        ],
        inputs: ['Any Data Node'],
        outputs: [],
        notes: 'Similar to pandas .describe().'
    },

    // --- Placeholders/Basics ---
    mlp: { title: 'MLP Node', description: 'Multi-Layer Perceptron neural network.', usage: [], inputs: [], outputs: [], notes: 'Placeholder.' },
    cnn: { title: 'CNN Node', description: 'Convolutional Neural Network.', usage: [], inputs: [], outputs: [], notes: 'Placeholder.' },
    rnn: { title: 'RNN Node', description: 'Recurrent Neural Network.', usage: [], inputs: [], outputs: [], notes: 'Placeholder.' },
    transformer: { title: 'Transformer Node', description: 'Transformer architecture.', usage: [], inputs: [], outputs: [], notes: 'Placeholder.' },
    visualizer: { title: 'Visualizer Node', description: 'General purpose visualizer.', usage: [], inputs: [], outputs: [], notes: 'Placeholder.' },
    exporter: { title: 'Exporter Node', description: 'Exports data or models.', usage: [], inputs: [], outputs: [], notes: 'Placeholder.' },
    evaluator: {
        title: 'Model Evaluator Node',
        description: 'Calculates performance metrics (MSE, Accuracy, R2) for trained models.',
        usage: [
            'Comparing model performance',
            'Validating results on test data'
        ],
        inputs: ['Regression/Classification/Clustering Models'],
        outputs: [],
        notes: 'Crucial for verifying model quality.'
    },
};
