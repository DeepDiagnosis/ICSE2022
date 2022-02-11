# DeepDiagnosis: Automatically Diagnosing Faults and Recommending Actionable Fixes in Deep Learning Programs


## DeepDiagnosis
To use DeepDiagnosis, you need to add our callback as subclass in your keras.callbacks.py file.

The core principle of our callback to get a view on internal states and statistics of the model during training.

Then you can pass our callback `DeepDiagnosis()` to the `.fit()` method of a model as following:

```python
callback = keras.callbacks.DeepLocalize(inputs, outputs, layer_number, batch_size, startTime)
model = keras.models.Sequential()
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation(activations.relu))
model.compile(keras.optimizers.SGD(), loss='mse')
model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10, batch_size=1, 
...                     callbacks=[callback], verbose=0)
```



## Prerequisites

Version numbers below are of confirmed working releases for this project.

    python 3.6.5
    Keras  2.2.0
    Keras-Applications  1.0.2
    Keras-Preprocessing 1.0.1  
    numpy 1.19.2
    pandas 1.1.5
    scikit-learn 0.21.2
    scipy 1.6.0
    tensorflow 1.14.0

## BibTeX Reference
If you find this [paper](https://conf.researchr.org/details/icse-2022/icse-2022-papers/35/DeepDiagnosis-Automatically-Diagnosing-Faults-and-Recommending-Actionable-Fixes-in-D) useful in your research, please consider citing:

    @inproceedings{wardat2021deepdiagnosis,
    author={Mohammad Wardat and Breno Dantas Cruz and Wei Le and Hridesh Rajan},
    title={DeepDiagnosis: Automatically Diagnosing Faults and Recommending Actionable Fixes in Deep Learning Programs}, 
    booktitle = {ICSE'22: The 44th International Conference on Software Engineering},
    location = {Pittsburgh, PA, USA},
    month = {May 21-May 29, 2022},
    year = {2022},
    entrysubtype = {conference}
    }
    
## This repository contains the reproducibility package of DeepDiagnosis
#### [Extractor Folder](https://github.com/DeepDiagnosis/ICSE2022/tree/main/Extractor): 
* Contains the source code to extract (.h5) to source code
#### [AUTOTRAINER Model Code](https://github.com/DeepDiagnosis/ICSE2022/tree/main/AUTOTRAINER%20Model%20Code):
* Contains the source code of all AUTOTRAINER Models
#### [DeepDiagnosis Result](https://github.com/DeepDiagnosis/ICSE2022/tree/main/DeepDiagnosis%20Result):
* Contains the results of DeepDiagnosis from AUTOTRAINER dataset
#### [MOTIVATING EXAMPLE](https://github.com/DeepDiagnosis/ICSE2022/tree/main/MOTIVATING%20EXAMPLE/31880720):
* Contains the results of the motivating example using AUTOTRAINER
#### [SaturatedActivation](https://github.com/DeepDiagnosis/ICSE2022/tree/main/SaturatedActivation):
* Contains the experiments of saturated activation for Sigmoid and Tanh
#### [difference threshold](https://github.com/DeepDiagnosis/ICSE2022/tree/main/difference%20threshold):
* Contains the result of AUTOTRAINER on normal models with different threshold (accuracy =100%)
#### [Table 6](https://github.com/DeepDiagnosis/ICSE2022/tree/main/Table%206)
* Complete result of Table 6
#### [Normal model With 20 % Accuracy](https://github.com/DeepDiagnosis/ICSE2022/tree/main/MNIST_Normal/double_random_99fe3625-3c58-4766-968e-7d40401237fe)
* this model(MNIST_Normal/double_random_99fe3625-3c58-4766-968e-7d40401237fe) is detected by DeepDiagnosis and the accuracy = 20%
#### [DeepLocalize Result](https://github.com/DeepDiagnosis/ICSE2022/tree/main/DeepLocalize%20Result)
* Contains the results of DeepLocalize from AUTOTRAINER dataset
#### [UMLUAT Result](https://github.com/DeepDiagnosis/ICSE2022/tree/main/UMLUAT%20Result)
* Contains the results of UMLUAT from AUTOTRAINER dataset





