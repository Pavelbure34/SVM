<h1>Support Vector Machine</h1>
<p>Made by Dongbin Alistaire Suh</p>

<h2>About this Project</h2>
<div>
    <p>
        This project is a linear support vector machine for binary classification, built with NumPy.
    </p>
</div>

<h2>Requirement</h2>
<div>
    <p>Input and output data should be processed. If discrete, they should be normalized.</p>
</div>

<h2>Implementation</h2>
<div>
    <ol>
        <li>Initiate the parameters</li>
        <ul>
            <li>W = vector of weight coefficients</li>
            <li>b = bias</li>
            <li>M = number of fields in the input data</li>
            <li>X = training dataset</li>
            <li>T = ground truth</li>
        </ul>
        <li>Train the model based on the given data, utilizing the stochastic gradient descent and slack variable.</li>
        <li>Give the test data set and get the classification result</li>
    </ol>
</div>

<h2>How to use it</h2>
<div>
    <ol>
        <li>model = SGD_SVC()</li>
        <li>model.fit(xvalues, yvalues, c_value, k_value)</li>
        <li>result = model.predict(test_xvalues)</li>
    </ol>
</div>