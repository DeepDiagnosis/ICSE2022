import tensorflow as tf 
import ast
import os
import pickle
import difflib

def kernel_initializer(AST):
     KIstr = 'tf.keras.initializers.'
     if AST['kernel_initializer']['class_name'] == 'VarianceScaling':
         return KIstr+"VarianceScaling(scale={0}, mode='{1}', distribution='{2}')".format(AST['kernel_initializer']['config']['scale'], AST['kernel_initializer']['config']['mode'], AST['kernel_initializer']['config']['distribution'])
     if AST['kernel_initializer']['class_name'] == 'Zeros':
         return KIstr+"Zeros()"
     if AST['kernel_initializer']['class_name'] == 'Constant':
         return KIstr+"Constant()"
     if AST['kernel_initializer']['class_name'] == 'Ones':
         return KIstr+"Ones()"
     if AST['kernel_initializer']['class_name'] == 'RandomUniform':
         return KIstr+"RandomUniform()"
     if AST['kernel_initializer']['class_name'] == 'TruncatedNormal':
         return KIstr+"TruncatedNormal()"

def embeddings_initializer(AST):
     KIstr = 'tf.keras.initializers.'
     if AST['embeddings_initializer']['class_name'] == 'VarianceScaling':
         return KIstr+"VarianceScaling(scale={0}, mode='{1}', distribution='{2}')".format(AST['embeddings_initializer']['config']['scale'], AST['embeddings_initializer']['config']['mode'], AST['embeddings_initializer']['config']['distribution'])
     if AST['embeddings_initializer']['class_name'] == 'Zeros':
         return KIstr+"Zeros()"
     if AST['embeddings_initializer']['class_name'] == 'Constant':
         return KIstr+"Constant()"
     if AST['embeddings_initializer']['class_name'] == 'Ones':
         return KIstr+"Ones()"
     if AST['embeddings_initializer']['class_name'] == 'RandomUniform':
         return KIstr+"RandomUniform()"
     if AST['embeddings_initializer']['class_name'] == 'TruncatedNormal':
         return KIstr+"TruncatedNormal()"
     
        
def bias_initializer(AST):
     KIstr = 'tf.keras.initializers.'
     if AST['bias_initializer']['class_name'] == 'VarianceScaling':
         return KIstr+"VarianceScaling(scale={0}, mode='{1}', distribution='{2}')".format(AST['bias_initializer']['config']['scale'], AST['bias_initializer']['config']['mode'], AST['bias_initializer']['config']['distribution'])
     if AST['bias_initializer']['class_name'] == 'Zeros':
         return KIstr+"Zeros()"
     if AST['bias_initializer']['class_name'] == 'Constant':
         return KIstr+"Constant()"
     if AST['bias_initializer']['class_name'] == 'Ones':
         return KIstr+"Ones()"
     if AST['bias_initializer']['class_name'] == 'RandomUniform':
         return KIstr+"RandomUniform()"
     if AST['bias_initializer']['class_name'] == 'TruncatedNormal':
         return KIstr+"TruncatedNormal()"

def kernel_regularizer(AST):
    KRstr ="tf.keras.regularizers."
    if AST['kernel_regularizer']['class_name'] == 'L1L2':
        l2 = AST['kernel_regularizer']['config']['l2']
        if str(AST['kernel_regularizer']['config']['l2']) == "0.009999999776482582":
            l2 = 0.01
        if str(AST['kernel_regularizer']['config']['l2']) == "9.999999747378752e-05":
            l2 = 1e-4
        return KRstr+"l1_l2(l1={0}, l2={1} )".format(AST['kernel_regularizer']['config']['l1'], str(l2))
        
    
    
# In[]

def createModel(buggyLines,name):
    f = open("{0}.txt".format(name), "w")
    
    f.write("model = tf.keras.Sequential()")
    f.write("\n")
    for line in buggyLines:
        print(line)
        #f.write(line)
        if line.find('batch_normalization') != -1:
            line = line.replace('ListWrapper([1])', 'RandomFun([1])')
        buggyAST = ast.literal_eval(line)
        if 'batch_input_shape' in buggyAST !=  -1:
            f.write("model.add(tf.keras.layers.Input(batch_shape={0}, dtype=tf.{1}, name=\"{2}\"))".format(buggyAST['batch_input_shape'],buggyAST['dtype'],buggyAST['name']))
            f.write("\n")
            
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('dense') != -1:
            KI = None
            BI = None
            KR = None
            if buggyAST['kernel_initializer'] is not None:
                KI= kernel_initializer(buggyAST)
            if buggyAST['bias_initializer'] is not None:
                BI =bias_initializer(buggyAST)
            if buggyAST['kernel_regularizer'] is not None:
                KR =kernel_regularizer(buggyAST)
            f.write("model.add(tf.keras.layers.Dense(units={0}, activation='{1}', kernel_initializer ={2}, bias_initializer = {3} , kernel_regularizer= {4},  name =\"{5}\"))".format(buggyAST['units'], buggyAST['activation'], KI, BI,KR, buggyAST['name']))
            f.write("\n")
            
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('embedding') != -1:
            KI = None
            if buggyAST['embeddings_initializer'] is not None:
                KI= embeddings_initializer(buggyAST)
            f.write("model.add(tf.keras.layers.Embedding(input_dim = {0}, output_dim = {1}, embeddings_initializer={2}, input_length = {4}, name =\"{5}\"))".format(buggyAST['input_dim'], buggyAST['output_dim'], KI, buggyAST['name']))
            f.write("\n")
            
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('batch_normalization') != -1:
            f.write("model.add(tf.keras.layers.BatchNormalization())")
            f.write("\n")
            
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('lstm') != -1:
            KI = None
            BI = None
            KR = None
            if buggyAST['kernel_initializer'] is not None:
                KI= kernel_initializer(buggyAST)
            if buggyAST['bias_initializer'] is not None:
                BI =bias_initializer(buggyAST)
            if buggyAST['kernel_regularizer'] is not None:
                KR =kernel_regularizer(buggyAST)
            f.write("model.add(tf.keras.layers.LSTM({0}, return_sequences ={1}, recurrent_activation = '{2}', activation = '{3}', kernel_initializer ={4}, bias_initializer = {5} , kernel_regularizer= {6},dropout= {7}, recurrent_dropout= {8}, implementation ={9},  name =\"{10}\"))"
                  .format(buggyAST['units'],  buggyAST['return_sequences'],  buggyAST['recurrent_activation'], buggyAST['activation'], KI, BI,KR, buggyAST['dropout'],buggyAST['recurrent_dropout'], buggyAST['implementation'],buggyAST['name']))
            f.write("\n")
           
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('max_pooling2d') != -1:
            f.write("tf.keras.layers.MaxPooling2D(pool_size={0}, strides={1}, padding=\"{2}\", data_format='{3}', name =\"{4}\")".format(buggyAST['pool_size'], buggyAST['strides'], buggyAST['padding'], buggyAST['data_format'],buggyAST['name']))
            f.write("\n")       
            
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('Conv2D') != -1:
            KI = None
            BI = None
            KR = None
            if buggyAST['kernel_initializer'] is not None:
                KI= kernel_initializer(buggyAST)
            if buggyAST['bias_initializer'] is not None:
                BI =bias_initializer(buggyAST)
            if buggyAST['bias_initializer'] is not None:
                KR =kernel_regularizer(buggyAST)
            f.write("model.add(tf.keras.layers.Conv2D(filters={0}, kernel_size={1},strides={2}, padding=\"{3}\", data_format='{4}', activation='{5}', kernel_initializer = {6}, bias_initializer = {7}, kernel_regularizer= {8}, name =\"{9}\"))"
                                           .format(buggyAST['filters'], buggyAST['kernel_size'], buggyAST['strides'], buggyAST['padding'], buggyAST['data_format'], buggyAST['activation'], KI, BI, KR , buggyAST['name']))
            f.write("\n")
            
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('activation') != -1:
            f.write("model.add(tf.keras.layers.Activation('{0}'))".format(buggyAST['activation']))
            print("\n")
        
        if 'name' in buggyAST !=  -1 and buggyAST['name'].find('flatten') != -1:
            f.write("model.add(tf.keras.layers.Flatten(data_format ='{0}'))".format(buggyAST['data_format']))
            f.write("\n")
        
    f.close()







model = tf.keras.models.load_model("best_model.h5")

fModel = open("file.txt", "w")

for layer in model.layers:
    fModel.write(str(layer.get_config()))
    fModel.write("\n")

fModel.close()      
buggyFile = open('file.txt', 'r')
buggyLines = buggyFile.readlines()
createModel(buggyLines,"fileM")
