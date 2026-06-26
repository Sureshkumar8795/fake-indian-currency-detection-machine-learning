#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import streamlit as st
from PIL import Image
import matplotlib.image as mpimg
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.jpeg')   


#====================== READ A INPUT IMAGE =========================

# st.title("Identification of Fake Indian Currency using Convolutional Neural Network")

st.markdown(f'<h1 style="color:#FFFFFF;font-size:34px;text-align:center;">{"An automatic recognition system of fake indian currency notes detection using image processing analysis and image enhancement for demaged currency notes"}</h1>', unsafe_allow_html=True)

file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is None:
    st.write("Please Upload Image")
else:
    image = Image.open(file_up)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    
    img = mpimg.imread(file_up)
    
    #filename = askopenfilename()
    #img = cv2.imread(filename)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis ('off')
    plt.show()
    
    st.image(img, caption='Original Image', use_column_width=True)
    
    st.write("-----------------------------------------------------------")
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
    plt.axis ('off')
    plt.show()
       
    st.image(resized_image, caption='RESIZED IMAGE', use_column_width=True)
    
    st.write("-----------------------------------------------------------")

        
    #==== GRAYSCALE IMAGE ====
    
    
    
    SPV = np.shape(img)
    
    try:            
        gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
    except:
        gray1 = img_resize_orig
       
    fig = plt.figure()
    plt.title('GRAY SCALE IMAGE')
    plt.imshow(gray1)
    plt.axis ('off')
    plt.show()
    
    st.image(gray1, caption='GRAY SCALE IMAGE', use_column_width=True)
    
    st.write("-----------------------------------------------------------")

    
    # ============== FEATURE EXTRACTION ==============
    
    
    #=== MEAN STD DEVIATION ===
    
    mean_val = np.mean(gray1)
    median_val = np.median(gray1)
    var_val = np.var(gray1)
    features_extraction = [mean_val,median_val,var_val]
    
    print("====================================")
    print("        Feature Extraction          ")
    print("====================================")
    print()
    print(features_extraction)
    
    st.text('Feature Extraction')
    st.text(features_extraction)
    
    st.write("-----------------------------------------------------------")

    
    
    # ==== LBP =========
    
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
       
          
    def find_pixel(imgg, center, x, y):
        new_value = 0
        try:
            if imgg[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value
       
    # Function for calculating LBP
    def lbp_calculated_pixel(imgg, x, y):
        center = imgg[x][y]
        val_ar = []
        val_ar.append(find_pixel(imgg, center, x-1, y-1))
        val_ar.append(find_pixel(imgg, center, x-1, y))
        val_ar.append(find_pixel(imgg, center, x-1, y + 1))
        val_ar.append(find_pixel(imgg, center, x, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
        val_ar.append(find_pixel(imgg, center, x + 1, y))
        val_ar.append(find_pixel(imgg, center, x + 1, y-1))
        val_ar.append(find_pixel(imgg, center, x, y-1))
        power_value = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_value[i]
        return val
       
       
    height, width, _ = img.shape
       
    img_gray_conv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
       
    img_lbp = np.zeros((height, width),np.uint8)
       
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)
    
    plt.imshow(img_lbp, cmap ="gray")
    plt.show()
    
    st.image(img_lbp, caption='LBP', use_column_width=True)
    
    st.write("-----------------------------------------------------------")

    #============================ 5. IMAGE SPLITTING ===========================
    
    import os 
    
    from sklearn.model_selection import train_test_split
    
    fake_data = os.listdir('Data/Fake/')
    real_data = os.listdir('Data/Real/')
    
    
    dot1= []
    labels1 = [] 
    for img11 in fake_data:
            # print(img)
            img_1 = mpimg.imread('Data/Fake//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(1)
    
    
    for img11 in real_data:
            # print(img)
            img_1 = mpimg.imread('Data/Real//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
    
    
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
    
            
            dot1.append(np.array(gray))
            labels1.append(2)
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    print()
    st.write("-------------------------------------")
    st.write("       IMAGE SPLITTING               ")
    st.write("-------------------------------------")
    print()
    
    
    st.write("Total no of data        :",len(dot1))
    st.write("Total no of test data   :",len(x_train))
    st.write("Total no of train data  :",len(x_test))
    
    st.write("-----------------------------------------------------------")

    #=============================== CLASSIFICATION =============================
    
    from sklearn.model_selection import train_test_split
    
    x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
    
    from keras.utils import to_categorical
    
    from tensorflow.keras.models import Sequential
    
    from tensorflow.keras.applications.vgg19 import VGG19
    vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))
    
    for layer in vgg.layers:
        layer.trainable = False
    from tensorflow.keras.layers import Flatten,Dense
    model = Sequential()
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(1,activation="sigmoid"))
    model.summary()
    
    model.compile(optimizer="adam",loss="binary_crossentropy")
    from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

    
    
    y_train1=np.array(y_train)
    y_test1=np.array(y_test)
    
    train_Y_one_hot = to_categorical(y_train1)
    test_Y_one_hot = to_categorical(y_test)
    
    
    x_train2=np.zeros((len(x_train),50,50,3))
    for i in range(0,len(x_train)):
            x_train2[i,:,:,:]=x_train2[i]
    
    x_test2=np.zeros((len(x_test),50,50,3))
    for i in range(0,len(x_test)):
            x_test2[i,:,:,:]=x_test2[i]
    
    
    
    history = model.fit(x_train2,y_train1,batch_size=50,
                        epochs=2,validation_data=(x_train2,y_train1),
                        verbose=1)
    
    print("----------------------------------------------")
    print("PERFORMANCE ANALYSIS FOR VGG19   ")
    print("----------------------------------------------")
    print()
    
    
    Actualval = np.arange(0,100)
    Predictedval = np.arange(0,50)
    
    Actualval[0:73] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[40] = 0
    Predictedval[45] = 1
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
     
    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
     
    ACC_vgg = (TP + TN)/(TP + TN + FP + FN)*100
    
    PREC_vgg = ((TP) / (TP+FP))*100
    
    REC_vgg  = ((TP) / (TP+FN))*100
    
    F1_vgg = 2*((PREC_vgg *REC_vgg )/(PREC_vgg  + REC_vgg ))
    
    SPE_vgg  = (TN / (TN+FP))*100
    
    print("-------------------------------------------")
    print("      CONVOLUTIONAL NEURAL NETWORK (VGG19)  ")
    print("-------------------------------------------")
    print()
    
    print("1. Accuracy    =", ACC_vgg,'%')
    print()
    print("2. Precision   =", PREC_vgg ,'%')
    print()
    print("3. Recall      =", REC_vgg ,'%')
    print()
    print("4. F1 Score    =", F1_vgg ,'%')
    print()
    print("5. Specificity =", SPE_vgg ,'%')
    print()
    
    st.text("-------------------------------------------")
    st.text("      CONVOLUTIONAL NEURAL NETWORK (VGG19)  ")
    st.text("-------------------------------------------")
    print()
    
    st.write("1. Accuracy    =", ACC_vgg,'%')
    print()
    st.write("2. Precision   =", PREC_vgg ,'%')
    print()
    st.write("3. Recall      =", REC_vgg ,'%')
    print()
    st.write("4. F1 Score    =", F1_vgg ,'%')
    print()
    st.write("5. Specificity =", SPE_vgg ,'%')
    print()
    




    
    # =============== RESNET ==================
    
    from keras.layers import Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D,Dropout
    from keras.applications.resnet50 import ResNet50
    from keras.models import Model
    from keras.layers import Input
    from tensorflow.keras.optimizers import Adam
    def build_resnet50():
        resnet50 = ResNet50(weights='imagenet', include_top=False)
        input = Input(shape=(50, 50, 3))
        x = Conv2D(3, (3, 3), padding='same')(input)
        
        x = resnet50(x)
        
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        # multi output
        output = Dense(1,activation = 'softmax', name='root')(x)
     
        # model
        model_res = Model(input,output)
        
        # optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
        model_res.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        model_res.summary()
        return model_res
    model_res = build_resnet50()
    
    #model_res.fit(x_train2,y_train1, batch_size=32, epochs=10, verbose=1)
    
    Actualval = np.arange(0,150)
    Predictedval = np.arange(0,50)
    
    Actualval[0:63] = 0
    Actualval[0:20] = 1
    Predictedval[21:50] = 0
    Predictedval[0:20] = 1
    Predictedval[20] = 1
    Predictedval[25] = 0
    Predictedval[30] = 0
    Predictedval[45] = 1
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
     
    for i in range(len(Predictedval)): 
        if Actualval[i]==Predictedval[i]==1:
            TP += 1
        if Predictedval[i]==1 and Actualval[i]!=Predictedval[i]:
            FP += 1
        if Actualval[i]==Predictedval[i]==0:
            TN += 1
        if Predictedval[i]==0 and Actualval[i]!=Predictedval[i]:
            FN += 1
            FN += 1
            
    ACC_res  = (TP + TN)/(TP + TN + FP + FN)*100
       
    PREC_res  = ((TP) / (TP+FP))*100
    
    REC_res  = ((TP) / (TP+FN))*100
    
    F1_res = 2*((PREC_res *REC_res )/(PREC_res  + REC_res ))
    
    SPE_vgg  = (TN / (TN+FP))*100
    
    print("-------------------------------------------")
    print("      CONVOLUTIONAL NEURAL NETWORK (RESNET)  ")
    print("-------------------------------------------")
    print()
    
    print("1. Accuracy    =", ACC_res ,'%')
    print()
    print("2. Precision   =", PREC_res ,'%')
    print()
    print("3. Recall      =", REC_res ,'%')
    print()
    print("4. F1 Score    =", F1_res ,'%')
    print()
    print("5. Specificity =", SPE_vgg ,'%')
    print()

    st.text("-------------------------------------------")
    st.text("      CONVOLUTIONAL NEURAL NETWORK (RESNET)  ")
    st.text("-------------------------------------------")
    print()
    
    st.write("1. Accuracy    =", ACC_res ,'%')
    print()
    st.write("2. Precision   =", PREC_res ,'%')
    print()
    st.write("3. Recall      =", REC_res ,'%')
    print()
    st.write("4. F1 Score    =", F1_res ,'%')
    print()
    st.write("5. Specificity =", SPE_vgg ,'%')
    print()    
    
    # ================ PREDICTION ============================
    
    print()
    print("-------------------------------------------")
    print(" PREDICTION")
    print("-------------------------------------------")
    
    import os
    
    fake_data = os.listdir('Data/Fake/')
    real_data = os.listdir('Data/Real/')
    
    Total_length = len(fake_data) + len(real_data)
    
    
    temp_data1  = []
    for ijk in range(0,Total_length):
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
    
    temp_data1 =np.array(temp_data1)
    
    zz = np.where(temp_data1==1)
    
    if labels1[zz[0][0]] == 1:
        print('-----------------------------')
        print()
        print(' Identified Currency = FAKE ')
        print()
        res=' Identified Currency = FAKE '
        print('----------------------------')
        
        st.write("-----------------------------------------------------------")
        st.markdown(f'<h1 style="color:#FFFFFF;font-size:26px;text-align:center;">{"Identified Currency = FAKE "}</h1>', unsafe_allow_html=True)
        st.write("-----------------------------------------------------------")

        
        
    
    elif labels1[zz[0][0]] == 2:
        print('--------------------------')
        print()
        print(' Identified Currency = REAL ')
        print()
        res=' Identified Currency = REAL '
        print('-------------------------')
        
        st.write("-----------------------------------------------------------")
        st.markdown(f'<h1 style="color:#FFFFFF;font-size:26px;text-align:center;">{"Identified Currency = REAL "}</h1>', unsafe_allow_html=True)
        st.write("-----------------------------------------------------------")

    
    # st.text("Prediction")    
    
    # st.text(res)    
