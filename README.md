"# FR_Pretrained_Test" 

Data Directory
--
```
|--/IJB-A_protocols: IJB-A Protocols。
|--/Model: The network structures of the pre-trained model.
|--/Pretrained: The pre-trained weights. 
|--/Protocol: CFP Dataset Protocols.
```
Pre-trained Models: [Download Link](https://drive.google.com/drive/folders/1EKHzQ4Q6PRl61mV_ZhtR7WPP1jJSLaFA?usp=sharing).

Codes
--
1. Check_List.py: Check the image name of the generated images. 
2. Main.py: Feature extraction using the STOA models, and store the features into .txt format. 
3. Main_HashTable.py: Similar to 2), which is used to load the POE models but store the features in the dictionary.
4. Main_InsightFace.py: ```to be filled```.
5. Main_Manual.py: Similar to 3), but is used to test the POE models. 
6. Main_IJBA.py: Evaluate the IJB-A by template pooling. The features are stored in dictionary, where the keys are the image name, and the values are the features. 
7. Main_MPIE.py: Evaluate the Multi-PIE
8. Main_PON_Test.py: It is used to extract the features of the generated images. 

